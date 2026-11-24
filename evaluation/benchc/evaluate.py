"""Evaluate released Valen models on prepared Bench-C data, with calibration metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from valen.data.schema import validate_record, target_distribution
from .metrics import distribution_metrics, summarize, paired, paired_details


def select_records(records, corruptions=None, severities=None, require_full_grid=False):
    """Always retain clean controls when filtering corrupted conditions."""
    available = {r['meta']['corruption'] for r in records if r['meta']['corruption']!='clean'}
    if corruptions and set(corruptions)-available:
        raise ValueError(f'Requested corruptions absent from data: {set(corruptions)-available}')
    selected = [r for r in records if r['meta']['corruption']=='clean' or
                ((not corruptions or r['meta']['corruption'] in corruptions) and
                 (not severities or r['meta']['severity'] in severities))]
    present = {r['meta']['severity'] for r in selected if r['meta']['corruption']!='clean'}
    if severities and set(severities)-present:
        raise ValueError('Requested severities absent from selected corruptions')
    if corruptions and set(corruptions)-{r['meta']['corruption'] for r in selected}:
        raise ValueError('Requested corruption has no selected severity')
    if not selected:
        raise ValueError('No records match the selected conditions')
    if require_full_grid:
        from . import CORRUPTIONS
        clean = {r['meta']['base_id'] for r in selected if r['meta']['corruption']=='clean'}
        groups = {}
        for r in selected:
            meta = r['meta']
            groups.setdefault((meta['corruption'],meta['severity']),set()).add(meta['base_id'])
        expected = {('clean',0)} | {(name,level) for name in CORRUPTIONS for level in range(1,6)}
        if not clean or set(groups)!=expected or any(ids!=clean for ids in groups.values()):
            raise ValueError('--require-full-grid requires every clean question in all 19 x 5 conditions')
    return selected


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_path(data):
    data = Path(data)
    return data/'eval.jsonl' if data.is_dir() else data


def read_records(data, verify_media=True):
    """Validate schema/pair identity and optionally image hashes without loading a model."""
    source = source_path(data)
    records, seen, signatures, verified = [], set(), {}, {}
    with source.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line))
                questions = record['request']['questions']
                if len(questions)!=1 or set(record['targets'])!=set(questions):
                    raise ValueError('Each Bench-C record requires exactly one labeled question')
                question = next(iter(questions.values()))
                if question['type']!='choice' or len(question['criteria'])<2:
                    raise ValueError('Bench-C requires a Choice question with at least two candidates')
                target = next(iter(record['targets'].values()))['probabilities']
                if target is None or list(target.values()).count(1.)!=1 or sum(p!=0 for p in target.values())!=1:
                    raise ValueError('Bench-C requires one-hot labels')
                meta = record['meta']
                if meta['benchmark']!='Bench-C' or not isinstance(meta['base_id'],str) or not meta['base_id']:
                    raise ValueError('Missing Bench-C identity metadata')
                if (not isinstance(meta['corruption'],str) or not meta['corruption'] or type(meta['severity']) is not int
                        or (meta['corruption']=='clean' and meta['severity']!=0)
                        or (meta['corruption']!='clean' and meta['severity'] not in range(1,6))):
                    raise ValueError('clean requires severity=0; corruptions require severity=1..5')
                identity = (meta['base_id'],meta['corruption'],meta['severity'])
                if identity in seen:
                    raise ValueError(f'Duplicate condition/question: {identity}')
                seen.add(identity)
                assets = {a['path']:a['sha256'] for a in record['assets']}
                if not assets or len(assets)!=len(record['assets']):
                    raise ValueError('Missing/duplicate assets')
                state = record['request']['state']
                context, references = [], []
                for message in state['messages']:
                    context.append(message['role'])
                    for item in message['content']:
                        if item['type']=='text':
                            context.append(item['text'])
                        elif item['type']=='image_url':
                            references.append(item['image_url']['url'])
                            context.append('<image>')
                        else:
                            raise ValueError('Bench-C state only supports text and images')
                if not references or set(references)!=set(assets):
                    raise ValueError('State images and assets must match')
                signature = (question, target, context)
                if signatures.setdefault(meta['base_id'],signature)!=signature:
                    raise ValueError('Paired variants must retain the same question, candidates, target, and text context')
                if verify_media:
                    for relative, expected in assets.items():
                        path = (source.parent/relative).resolve(strict=True)
                        if Path(relative).is_absolute() or not path.is_relative_to(source.parent.resolve()):
                            raise ValueError('Prepared images must be inside the dataset directory')
                        if path not in verified:
                            verified[path] = file_hash(path)
                        if verified[path]!=expected:
                            raise ValueError(f'Media hash mismatch: {relative}')
                records.append(record)
            except (ValueError,KeyError,TypeError,AttributeError,OSError) as error:
                raise ValueError(f'{source}:{line_number}: {error}') from error
    if not records:
        raise ValueError('No labeled Bench-C records')
    return records


def evaluate(model, data, output, device='cuda', execution='shared_state', bins=10, high_confidence=.9,
             corruptions=None, severities=None, require_full_grid=False, _records=None):
    # Lazy import keeps --validate-only usable with just Python and dataset files.
    import torch

    if bins<1 or not 0<=high_confidence<=1:
        raise ValueError('Invalid confidence bin count or threshold')
    source, output = source_path(data), Path(output)
    records = select_records(read_records(source) if _records is None else _records,
                             corruptions,severities,require_full_grid)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    t = time.perf_counter()
    compiler = model.make_compiler(media_root=source.parent, execution=execution)
    processor_seconds = time.perf_counter()-t
    model.eval()

    def synchronize():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    rows = []
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream, torch.inference_mode():
        for index, record in enumerate(records):
            # Target, corruption identity and categories are scoring-only information.
            compiled = compiler.compile({'request':record['request'],'assets':record['assets']})
            if {q.qid for q in compiled.questions}!=set(record['request']['questions']):
                raise ValueError(f'Record {index}: compiled questions differ from data')
            logits = model.forward_state(compiled) if compiled.inputs is not None else [model(q) for q in compiled.questions]
            for question, values in zip(compiled.questions,logits,strict=True):
                spec = record['request']['questions'][question.qid]
                if question.kind!='choice' or set(question.keys)!=set(spec['criteria']) or len(question.keys)!=len(spec['criteria']):
                    raise ValueError('Compiled candidate keys/type differ from dataset')
                values = values.detach().double().cpu()
                if values.ndim!=1 or len(values)!=len(question.keys) or not torch.isfinite(values).all():
                    raise ValueError('Expected one finite logit per candidate')
                target = target_distribution(record['targets'][question.qid],question.keys)
                probability = values.softmax(-1)
                p = dict(zip(question.keys, probability.tolist()))
                y = dict(zip(question.keys,target))
                metrics = distribution_metrics(p,y)
                # Stable log-softmax preserves NLL even if probabilities underflow.
                metrics['nll'] = float(-(torch.tensor(target,dtype=torch.float64)*values.log_softmax(-1)).sum())
                meta = record['meta']
                predicted = question.keys[int(values.argmax())]
                row = dict(record_index=index, record_id=meta['benchmark_record_id'], base_id=meta['base_id'],
                           qid=question.qid, task='choice', modality='image', corruption=meta['corruption'],
                           severity=meta['severity'], category=meta.get('category','unknown'),
                           l2_category=meta.get('l2_category','unknown'), target=y, probabilities=p,
                           prediction=predicted, confidence=metrics['confidence'], metrics=metrics)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if (index+1)%50==0:
                stream.flush()
                print(json.dumps(dict(event='progress',records=index+1,questions=len(rows))),flush=True)
    synchronize()
    seconds = time.perf_counter()-started
    pair_rows = list(paired_details(rows))
    with (output/'paired_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in pair_rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    report = dict(benchmark='Bench-C', data=str(source), data_sha256=file_hash(source), records=len(records),
                  questions=len(rows), base_questions=len({r['base_id'] for r in rows}), execution=execution,world_size=1,
                  device=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else str(device),
                  media_kwargs=compiler.media_kwargs,processor_load_seconds=processor_seconds,
                  evaluation_seconds=seconds,questions_per_second=len(rows)/seconds,
                  timing_scope='Compile/media processing, forward, scoring and prediction writes; excludes preflight JSONL/hash validation, model/processor loading, final aggregation and downloads.',
                  calibration=dict(confidence='Maximum candidate softmax probability; no temperature scaling',
                                   bins=bins,binning='Equal-width [lower, upper), last bin includes 1',
                                   high_confidence_threshold=high_confidence),
                  metrics=summarize(rows,bins,high_confidence),paired_changes=paired(rows),
                  selection=dict(corruptions=corruptions,severities=severities,require_full_grid=require_full_grid),
                  paired_overall=dict(pairs=len(pair_rows), **{key:sum(r[key] for r in pair_rows)/len(pair_rows) if pair_rows else None
                                                             for key in ('dAcc','dS','dC','RAS')}),
                  paired_note='dAcc/dS/dC/RAS match Bench-C src/metrics.py at 9ff3ea6279bf8576ad9720944176202134e914d7. dC is the change of per-sample absolute confidence/correctness error, not a change of binned ECE. RAS is computed per pair then averaged. Other deltas are descriptive corrupted-minus-clean diagnostics.',
                  brier_definition='Mean over questions of SUM over candidate squared errors (Valen convention); official Bench-C brier_score instead divides each question by K.',
                  evaluation_note='Overall pools every evaluated condition. Use condition/clean/0 for the clean baseline; do not confuse pooled condition accuracy with clean accuracy.')
    preparation = source.parent/'preparation.json'
    if preparation.is_file():
        report['preparation'] = json.loads(preparation.read_text(encoding='utf-8'))
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='Valen-Team/Valen-2B')
    parser.add_argument('--data',required=True,help='Prepared directory or native JSONL; all conditions in eval.jsonl by default')
    parser.add_argument('--output',help='New or empty result directory; required except with --validate-only')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--execution',choices=('question','shared_state'),default='shared_state')
    parser.add_argument('--attn-implementation',choices=('eager','sdpa','flash_attention_2'),default='sdpa')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--lora',action='store_true')
    parser.add_argument('--base-model')
    parser.add_argument('--bins',type=int,default=10)
    parser.add_argument('--high-confidence',type=float,default=.9)
    parser.add_argument('--validate-only',action='store_true',help='Check schema, labels, pairing, media paths and hashes without loading a model')
    parser.add_argument('--corruptions',nargs='+',help='Evaluate these corruption types, always retaining clean controls')
    parser.add_argument('--severities',nargs='+',type=int,choices=range(1,6))
    parser.add_argument('--require-full-grid',action='store_true',help='Require clean plus all 19 x 5 conditions on identical question IDs')
    args = parser.parse_args()
    if args.bins<1 or not 0<=args.high_confidence<=1:
        parser.error('--bins must be positive; --high-confidence must be in [0,1]')
    if args.base_model and not args.lora:
        parser.error('--base-model requires --lora')
    if args.validate_only:
        records = select_records(read_records(args.data),args.corruptions,args.severities,args.require_full_grid)
        print(json.dumps(dict(event='validated', records=len(records),
                              base_questions=len({r['meta']['base_id'] for r in records}),
                              conditions=sorted({r['meta']['benchmark_subset'] for r in records}))))
        return
    if not args.output:
        parser.error('--output is required for evaluation')
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error('--output must be new or empty')
    records = read_records(args.data)  # Fail on malformed data before model download/load.
    select_records(records,args.corruptions,args.severities,args.require_full_grid)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    folder = Path(args.model)
    if not folder.is_dir():
        folder = Path(snapshot_download(args.model,revision=args.revision,local_dir=args.model,
                                        allow_patterns='unmerged/*' if args.lora else None))
    options = dict(trust_remote_code=True,dtype=torch.bfloat16 if args.lora else 'auto',
                   attn_implementation=args.attn_implementation,local_files_only=False)
    if args.lora:
        folder /= 'unmerged'
        if args.base_model:
            options['base_model_path'] = args.base_model
    started = time.perf_counter()
    model = AutoModel.from_pretrained(folder,**options).to(args.device).eval()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter()-started
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    report = evaluate(model,args.data,args.output,args.device,args.execution,args.bins,args.high_confidence,
                      args.corruptions,args.severities,args.require_full_grid,_records=records)
    report.update(model=args.model,revision=args.revision,format='unmerged' if args.lora else 'complete',
                  attention=args.attn_implementation,load_seconds=load_seconds)
    path = Path(args.output)/'metrics.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(event='complete',questions=report['questions'],
                         accuracy=report['metrics']['overall']['accuracy'],ece=report['metrics']['overall']['ece'],
                         metrics=str(path))),flush=True)


if __name__ == '__main__':
    main()
