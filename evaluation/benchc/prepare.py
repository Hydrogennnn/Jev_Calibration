"""Convert a Bench-C TSV (including embedded/multiple images) to native Valen JSONL."""
import argparse
import ast
import base64
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import quote

from PIL import Image

from valen.data.schema import validate_record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def list_field(value):
    value = (value or '').strip()
    if not value or value.lower() == 'nan':
        return []
    if value.startswith('['):
        result = ast.literal_eval(value)
        if not isinstance(result, list) or not all(isinstance(v, str) and v for v in result):
            raise ValueError('Expected a nonempty-string list in an image field')
        return result
    return [value]


def relative_image_path(value):
    # Released TSVs use '/5.jpg' as a root-relative image name, not a host path.
    path = PurePosixPath(value.lstrip('/'))
    if not path.parts or '..' in path.parts or ':' in value or '\\' in value:
        raise ValueError(f'Unsafe image reference: {value}')
    return Path(*path.parts)


def checked_text(value, name):
    if not isinstance(value, str) or not value.strip() or '<|' in value:
        raise ValueError(f'{name} must be nonempty text without control tokens')
    return value


def image_bytes(payload):
    if payload.startswith('data:'):
        header, payload = payload.split(',', 1)
        if ';base64' not in header:
            raise ValueError('Only base64 data URLs are supported')
    return base64.b64decode(''.join(payload.split()), validate=True)


def save_image(raw, folder, stem):
    # Keep original encoded bytes: no extra JPEG compression, resizing or cropping.
    with Image.open(BytesIO(raw)) as image:
        kind, size = image.format, image.size
        image.verify()
    extensions = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'BMP': '.bmp', 'GIF': '.gif', 'TIFF': '.tiff'}
    if kind not in extensions:
        raise ValueError(f'Unsupported image format: {kind}')
    path = folder/(stem + extensions[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, dict(sha256=hashlib.sha256(raw).hexdigest(), width=size[0], height=size[1])


def conditions(root, corruptions=None, severities=None):
    if root is None:
        if corruptions or severities:
            raise ValueError('--corruptions/--severities require --corrupted-root')
        return [('clean', 0)]
    root = Path(root).resolve(strict=True)
    names = corruptions or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    result = [('clean', 0)]
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'clean':
            raise ValueError(f'Invalid corruption name: {name}')
        levels = severities or sorted(int(p.name) for p in (root/name).iterdir() if p.is_dir() and p.name.isdigit())
        if not levels:
            raise ValueError(f'No severity directories for {name}')
        for level in levels:
            if level not in range(1, 6) or not (root/name/str(level)).is_dir():
                raise ValueError(f'Missing/invalid corruption severity: {name}/{level}')
            if (name, level) in result:
                raise ValueError(f'Duplicate condition: {name}/{level}')
            result.append((name, level))
    if len(result) == 1:
        raise ValueError('No corruptions found under --corrupted-root')
    return result


def prepare(data, output, image_root=None, corrupted_root=None, corruptions=None, severities=None):
    data, output = Path(data).resolve(), Path(output).resolve()
    construction = None
    if data.is_dir() and (data/'construction.json').is_file():
        construction = json.loads((data/'construction.json').read_text())
        if construction.get('status') != 'complete':
            raise ValueError('Construction is incomplete; finish construct --resume before preparing')
        if image_root or corrupted_root:
            raise ValueError('A constructed dataset discovers its own clean/corrupted roots')
        image_root, corrupted_root = data/'clean/images', data/'corrupted'
        if (sha256(data/'clean/bench_c.tsv') != construction['clean_tsv_sha256']
                or sha256(corrupted_root/'image_manifest.json') != construction['image_manifest_sha256']):
            raise ValueError('Constructed manifest hash mismatch')
        data = data/'clean/bench_c.tsv'
    if data.is_dir():
        candidates = list(data.glob('*.tsv'))
        if len(candidates) != 1:
            raise ValueError('Pass a specific TSV when the directory does not contain exactly one TSV')
        data = candidates[0]
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    image_root = Path(image_root).resolve() if image_root else None
    corrupted_root = Path(corrupted_root).resolve() if corrupted_root else None
    variants = conditions(corrupted_root, corruptions, severities)
    image_map, corrupted_hashes = {}, {}
    if construction:
        image_map = json.loads((corrupted_root/'image_manifest.json').read_text())
        for name, level in variants[1:]:
            ledger = corrupted_root/name/str(level)/'images.jsonl'
            if sha256(ledger) != construction['ledger_sha256'][f'{name}/{level}']:
                raise ValueError(f'Corruption ledger changed: {name}/{level}')
            entries = [json.loads(line) for line in ledger.read_text().splitlines()]
            if len(entries)!=len(image_map) or {e['identity'] for e in entries}!=set(image_map):
                raise ValueError(f'Incomplete image coverage: {name}/{level}')
            corrupted_hashes[(name,level)] = {e['identity']:e['sha256'] for e in entries}
    csv.field_size_limit(sys.maxsize)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A bad label/image leaves no partial dataset at the requested output path.
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    seen, images_per_question, categories, level2 = set(), Counter(), Counter(), Counter()
    stats = dict(questions=0, images=0, embedded_images=0, file_images=0)
    source_digest = sha256(data)
    try:
        with ExitStack() as stack:
            source = stack.enter_context(data.open(encoding='utf-8-sig', newline=''))
            reader = csv.DictReader(source, delimiter='\t')
            if not {'index', 'question', 'A', 'B', 'answer'}.issubset(reader.fieldnames or []):
                raise ValueError('TSV requires index, question, A, B, and answer columns')
            streams = {variant: stack.enter_context((stage/('clean.jsonl' if variant[0]=='clean' else f'{variant[0]}_{variant[1]}.jsonl')).open('w', encoding='utf-8'))
                       for variant in variants}
            all_stream = stack.enter_context((stage/'eval.jsonl').open('w', encoding='utf-8'))
            for row_number, row in enumerate(reader, 2):
                try:
                    base_id = checked_text(row['index'].strip(), 'index')
                    if base_id in ('.', '..'):
                        raise ValueError('index cannot be a directory traversal component')
                    if base_id in seen:
                        raise ValueError(f'Duplicate index: {base_id}')
                    seen.add(base_id)
                    question = checked_text(row['question'], 'question')
                    options = {key: checked_text(row[key], key) for key in 'ABCDE'
                               if row.get(key, '').strip() and row[key].strip().lower() != 'nan'}
                    answer = row['answer'].strip()
                    if len(options) < 2 or answer not in options:
                        raise ValueError('Expected at least two options and a matching answer letter')
                    hint = row.get('hint', '') or ''
                    if hint.strip().lower() == 'nan':
                        hint = ''
                    if hint:
                        checked_text(hint, 'hint')
                    payloads, paths = list_field(row.get('image')), list_field(row.get('image_path'))
                    if payloads and paths and len(payloads) != len(paths):
                        raise ValueError('Embedded image count differs from image_path count')
                    if not payloads and not paths:
                        raise ValueError('No images for this question')
                    raw_images = []
                    for i in range(len(payloads or paths)):
                        if payloads:
                            raw_images.append(image_bytes(payloads[i]))
                            stats['embedded_images'] += 1
                        else:
                            if image_root is None:
                                raise ValueError('Missing embedded image: supply --image-root')
                            raw_images.append((image_root/relative_image_path(paths[i])).read_bytes())
                            stats['file_images'] += 1
                    if corrupted_root and len(paths) != len(raw_images):
                        raise ValueError('Corruption import requires image_path for each image')
                    if construction:
                        for path, raw in zip(paths,raw_images,strict=True):
                            if hashlib.sha256(raw).hexdigest()!=image_map[path]['clean_sha256']:
                                raise ValueError(f'Constructed clean image changed: {path}')
                    category, l2 = row.get('category') or 'unknown', row.get('l2-category') or 'unknown'
                    safe_id = quote(base_id, safe='')
                    for corruption, severity in variants:
                        content = [dict(type='text', text=f'Hint: {hint}')] if hint else []
                        assets = []
                        for i, clean_bytes in enumerate(raw_images):
                            corrupt_path = image_map[paths[i]]['corrupted_path'] if construction else paths[i] if paths else None
                            raw = (clean_bytes if corruption=='clean' else
                                   (corrupted_root/corruption/str(severity)/relative_image_path(corrupt_path)).read_bytes())
                            if construction and corruption!='clean' and hashlib.sha256(raw).hexdigest()!=corrupted_hashes[(corruption,severity)][paths[i]]:
                                raise ValueError(f'Constructed corrupted image changed: {corruption}/{severity}/{paths[i]}')
                            path, properties = save_image(raw, stage/'assets'/corruption/str(severity)/safe_id, f'image_{i}')
                            relative = path.relative_to(stage).as_posix()
                            if len(raw_images) > 1:
                                content.append(dict(type='text', text=f'Image {i+1}:'))
                            content.append(dict(type='image_url', image_url=dict(url=relative)))
                            assets.append(dict(path=relative, **properties))
                        subset = 'clean' if corruption=='clean' else f'{corruption}/{severity}'
                        record = dict(
                            group_id=f'benchc:{safe_id}',
                            request=dict(state=dict(messages=[dict(role='user', content=content)]),
                                         questions={'q': dict(type='choice', instructions=question, criteria=options)}),
                            targets={'q': dict(probabilities={k: float(k==answer) for k in options})}, assets=assets,
                            meta=dict(benchmark='Bench-C', benchmark_record_id=f'benchc:{safe_id}/{subset}',
                                      benchmark_subset=subset, base_id=base_id, corruption=corruption, severity=severity,
                                      modality='image', language_bucket='unknown', domain=category,
                                      category=category, l2_category=l2, source_index=base_id, source_row=row_number,
                                      source_image_paths=paths, image_count=len(raw_images)))
                        validate_record(record)
                        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'
                        streams[(corruption,severity)].write(serialized)
                        all_stream.write(serialized)
                    stats['questions'] += 1
                    stats['images'] += len(raw_images)
                    images_per_question[len(raw_images)] += 1
                    categories[category] += 1
                    level2[l2] += 1
                except Exception as error:
                    raise ValueError(f'{data}: TSV row {row_number}, index={row.get("index")!r}: {error}') from error
        if not stats['questions']:
            raise ValueError('Empty TSV')
        if sha256(data) != source_digest:
            raise ValueError('TSV changed during conversion')
        report = dict(benchmark='Bench-C', source_tsv=str(data), source_sha256=source_digest,
                      clean=stats, conditions=[dict(corruption=c, severity=s, questions=stats['questions']) for c,s in variants],
                      records=stats['questions']*len(variants), images_per_question=dict(images_per_question),
                      categories=dict(categories), l2_categories=dict(level2),
                      image_policy='Embedded image bytes preferred; original encodings and multi-image order preserved.',
                      corruption_policy='Import existing images only; no synthetic corruptions generated.',
                      official_split_verified=False,
                      split_note='All supplied TSV rows retained. The currently documented official 849-row split is not assumed.',
                      manifests={p.name:sha256(p) for p in sorted(stage.glob('*.jsonl'))})
        if construction:
            report['construction'] = construction
        (stage/'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if output.exists():
            raise FileExistsError(f'Output appeared during conversion: {output}')
        stage.rename(output)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='TSV file or directory containing exactly one TSV')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--image-root', help='Fallback image root when TSV has no embedded images')
    parser.add_argument('--corrupted-root', help='Optional existing <corruption>/<severity>/<source image_path> tree')
    parser.add_argument('--corruptions', nargs='+', help='Names to import; default: all directories')
    parser.add_argument('--severities', nargs='+', type=int, choices=range(1,6), help='Severities to import; default: existing levels')
    args = parser.parse_args()
    report = prepare(**vars(args))
    print(json.dumps({k:report[k] for k in ('source_tsv','clean','records','images_per_question','conditions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
