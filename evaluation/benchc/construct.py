"""Build raw clean/corrupted Bench-C TSVs with the official corruption processors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from PIL import Image

from .prepare import prepare, sha256
from . import CORRUPTIONS, OFFICIAL_COMMIT


SNOW_FIX_FROM = 'snow_blurred = gaussian_filter(snow_layer, sigma=blur_sigma)'
SNOW_FIX_TO = (SNOW_FIX_FROM + '\n            snow_blurred = cv2.resize(snow_blurred, (w, h), interpolation=cv2.INTER_LINEAR)')


def compatible_source(path):
    source = path.read_text(encoding='utf-8')
    if source.count(SNOW_FIX_FROM)!=1:
        raise ValueError('Pinned snow compatibility patch no longer matches upstream')
    # Upstream's Wand branch already resizes the enlarged snow layer before
    # blending; its no-Wand branch omitted this same required operation.
    return source.replace(SNOW_FIX_FROM,SNOW_FIX_TO)


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_tsv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@lru_cache(maxsize=4)
def load_processors(repo):
    # Keep the upstream checkout unchanged; patch only the missing fallback resize.
    path = Path(repo)/'src/corruption/base.py'
    spec = importlib.util.spec_from_file_location('_benchc_official_corruptions', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(compatible_source(path),str(path),'exec'), module.__dict__)
    if set(module.CORRUPTION_PROCESSORS) != set(CORRUPTIONS):
        raise ValueError('Official corruption registry differs from the supported 19 types')
    return module.CORRUPTION_PROCESSORS


def provenance(repo):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f'Use the verified official revision: git -C {repo} checkout {OFFICIAL_COMMIT}')
    files = ['src/corruption/base.py', 'src/corruption/pipeline.py', 'src/metrics.py']
    hashes = {}
    for name in files:
        raw = subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{name}'])
        hashes[name] = sha256(repo/name)
        if hashes[name] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'Upstream source has local changes: {name}')
    # The pinned upstream Wand motion-blur branch has no return value. Use the
    # upstream OpenCV fallback supplied by the official dependency environment.
    if importlib.util.find_spec('wand') is not None:
        raise ValueError('Use a generation environment without Wand: the pinned official motion_blur Wand branch is incomplete')
    frost_dir = repo/'images_c/Weather/Frost/frost'
    textures = sorted(p for p in frost_dir.glob('*') if p.is_file())
    versions = {name:importlib.metadata.version(name) for name in ('numpy','Pillow','scipy','scikit-image')}
    import cv2
    versions['opencv'] = cv2.__version__
    return dict(repository='https://github.com/xiangjieSui/Bench-C', commit=commit, files=hashes,
                versions=versions, python=sys.version.split()[0],
                compatibility_patch='snow SciPy fallback: resize enlarged snow layer to (w,h) using INTER_LINEAR before stacking, matching the upstream Wand branch',
                loaded_source_sha256=hashlib.sha256(compatible_source(repo/'src/corruption/base.py').encode()).hexdigest(),
                motion_blur='official OpenCV fallback (Wand absent)',
                snow='official SciPy fallback (Wand absent)',
                frost='official texture lookup with built-in fallback when unavailable',
                frost_textures={p.name:sha256(p) for p in textures})


def generate_image(job):
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    repo, source, destination, corruption, severity, seed, identity = job
    # Per-image seeds are independent of process scheduling and resume order.
    digest = hashlib.sha256(json.dumps([seed,corruption,severity,identity]).encode()).digest()
    image_seed = int.from_bytes(digest[:4], 'big')
    np.random.seed(image_seed)
    random.seed(image_seed)
    processor = load_processors(repo)[corruption](input_folder=str(Path(source).parent),
                                                 output_root=str(Path(destination).parent),
                                                 severity_level=severity)
    with Image.open(source) as image:
        array = np.array(image.convert('RGB'))
    # Preserve the official pipeline's RGB input and final JPEG quality=95.
    result = processor.apply_to_image(array, severity)
    if not isinstance(result, np.ndarray) or result.shape != array.shape or result.dtype != np.uint8:
        raise ValueError(f'Invalid official processor output: {corruption}/{severity}/{identity}')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.tmp')
    Image.fromarray(result).save(temporary, format='JPEG', quality=95)
    temporary.replace(destination)
    return dict(identity=identity, sha256=sha256(destination), seed=image_seed)


def construct(data, output, official_repo, corruptions=None, severities=None, seed=0, workers=1, resume=False):
    data, output, official_repo = Path(data).resolve(), Path(output).resolve(), Path(official_repo).resolve()
    if data.is_dir():
        candidates = list(data.glob('*.tsv'))
        if len(candidates) != 1:
            raise ValueError('Pass a specific TSV when the input directory does not contain exactly one TSV')
        data = candidates[0]
    names, levels = list(corruptions or CORRUPTIONS), list(severities or range(1,6))
    if (not names or len(set(names))!=len(names) or set(names)-set(CORRUPTIONS)
            or not levels or len(set(levels))!=len(levels) or set(levels)-set(range(1,6)) or workers<1):
        raise ValueError('Invalid corruption names, severities or worker count')
    upstream = provenance(official_repo)
    config = dict(source_sha256=sha256(data), corruptions=names, severities=levels, seed=seed,
                  upstream=upstream, adapter_sha256=sha256(__file__), prepare_sha256=sha256(Path(__file__).with_name('prepare.py')))
    state_path = output/'construction.json'
    if output.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError('Use a new output directory, or --resume for a matching construction')
        state = json.loads(state_path.read_text())
        if state['config'] != config:
            raise ValueError('Source, code, environment, seed or requested conditions changed; use a new output')
    else:
        output.mkdir(parents=True)
        state = dict(status='initializing', config=config, source_tsv=str(data), official_split_verified=False,
                     split_note='All local TSV rows retained; not claimed to be the documented official 849-question split.')
        write_json(state_path, state)
    csv.field_size_limit(sys.maxsize)
    with data.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields, source_rows = reader.fieldnames, list(reader)
    if sha256(data)!=config['source_sha256']:
        raise ValueError('Source TSV changed during construction')
    if 'image' not in fields or 'image_path' not in fields:
        raise ValueError('Construction requires image and image_path TSV fields')
    if state['status']=='initializing':
        # Recreate interrupted materialization, only within this owned construction.
        for name in ('clean', '.materialize'):
            if (output/name).exists():
                shutil.rmtree(output/name)
        native = output/'.materialize'
        prepared = prepare(data, native)
        records = [json.loads(line) for line in (native/'clean.jsonl').read_text().splitlines()]
        (output/'clean').mkdir()
        (native/'assets/clean/0').rename(output/'clean/images')
        clean_rows, image_map = [], {}
        for original, record in zip(source_rows, records, strict=True):
            paths = [str(Path(a['path']).relative_to('assets/clean/0')) for a in record['assets']]
            clean_rows.append(dict(original, image='', image_path=json.dumps(paths)))
            for path, asset in zip(paths,record['assets'],strict=True):
                image_map[path] = dict(clean_sha256=asset['sha256'], corrupted_path=str(Path(path).with_suffix('.jpg')))
        write_tsv(output/'clean/bench_c.tsv', fields, clean_rows)
        (output/'corrupted').mkdir(exist_ok=True)
        write_json(output/'corrupted/image_manifest.json', image_map)
        shutil.rmtree(native)
        state.update(status='generating', questions=len(clean_rows), images=len(image_map),
                     clean_tsv_sha256=sha256(output/'clean/bench_c.tsv'),
                     image_manifest_sha256=sha256(output/'corrupted/image_manifest.json'),
                     expected_corrupted_instances=len(clean_rows)*len(names)*len(levels),
                     expected_corrupted_images=len(image_map)*len(names)*len(levels), clean_summary=prepared['clean'])
        write_json(state_path,state)
    if (sha256(output/'clean/bench_c.tsv') != state['clean_tsv_sha256']
            or sha256(output/'corrupted/image_manifest.json') != state['image_manifest_sha256']):
        raise ValueError('Constructed clean manifest changed')
    image_map = json.loads((output/'corrupted/image_manifest.json').read_text())
    for path, entry in image_map.items():
        if sha256(output/'clean/images'/path) != entry['clean_sha256']:
            raise ValueError(f'Clean image changed: {path}')
    with (output/'clean/bench_c.tsv').open(newline='') as stream:
        clean_rows = list(csv.DictReader(stream, delimiter='\t'))
    state['status'] = 'generating'
    write_json(state_path,state)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for name in names:
            for level in levels:
                folder = output/'corrupted'/name/str(level)
                folder.mkdir(parents=True, exist_ok=True)
                ledger = folder/'images.jsonl'
                completed = {}
                if ledger.exists():
                    # Only a truncated last line is recoverable after interruption.
                    lines = ledger.read_text().splitlines()
                    for i,line in enumerate(lines):
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            if i != len(lines)-1:
                                raise
                            break
                        completed[entry['identity']] = entry
                jobs = []
                valid = {}
                for path, entry in image_map.items():
                    dest = folder/entry['corrupted_path']
                    old = completed.get(path)
                    if old and dest.is_file() and sha256(dest)==old['sha256']:
                        valid[path] = old
                    else:
                        jobs.append((str(official_repo),str(output/'clean/images'/path),str(dest),name,level,seed,path))
                # Rewrite a clean ledger before appending new completions.
                with ledger.open('w',encoding='utf-8') as stream:
                    for entry in valid.values():
                        stream.write(json.dumps(entry)+'\n')
                    stream.flush()
                    for entry in executor.map(generate_image,jobs):
                        stream.write(json.dumps(entry)+'\n')
                        stream.flush()
                        valid[entry['identity']] = entry
                corrupted_rows = [dict(row, image_path=json.dumps([image_map[p]['corrupted_path']
                                   for p in json.loads(row['image_path'])])) for row in clean_rows]
                write_tsv(folder/'bench_c.tsv',fields,corrupted_rows)
                print(json.dumps(dict(event='condition_complete',corruption=name,severity=level,
                                      images=len(valid),generated=len(jobs))),flush=True)
    state.update(status='complete', ledger_sha256={f'{name}/{level}':sha256(output/'corrupted'/name/str(level)/'images.jsonl')
                 for name in names for level in levels})
    write_json(state_path,state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--official-repo',required=True,help=f'Official checkout at {OFFICIAL_COMMIT}')
    parser.add_argument('--corruptions',nargs='+',choices=CORRUPTIONS)
    parser.add_argument('--severities',nargs='+',type=int,choices=range(1,6))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    result = construct(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('status','questions','images','expected_corrupted_instances','expected_corrupted_images')},indent=2))


if __name__=='__main__':
    main()
