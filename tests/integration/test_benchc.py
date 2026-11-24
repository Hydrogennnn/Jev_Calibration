"""Bench-C data integrity, probability calibration, and released-model API tests."""
import base64
import csv
from io import BytesIO
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from evaluation.benchc.prepare import prepare
from evaluation.benchc.evaluate import evaluate, read_records, select_records
from evaluation.benchc.metrics import aggregate, distribution_metrics, paired, official_pair_metrics


def encoded_image(color):
    stream = BytesIO()
    Image.new('RGB',(12,8),color).save(stream,format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


def write_tsv(path, rows):
    fields = ['index','question','hint','A','B','C','D','E','answer','category','l2-category','image','image_path']
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def sample(index='1', multi=False):
    images = [encoded_image('red'),encoded_image('blue')] if multi else [encoded_image('red')]
    return dict(index=index,question='Which image is red?',hint='Use the supplied evidence.',
                A='First',B='Second',C='',D='',E='',answer='A',category='color',
                **{'l2-category':'perception'},image=repr(images) if multi else images[0],
                image_path=repr([f'/{index}_{i}.png' for i in range(len(images))]))


def prepared(tmp_path, corrupted=False):
    tsv = tmp_path/'source.tsv'
    write_tsv(tsv,[sample('1',multi=True),sample('2')])
    kwargs = {}
    if corrupted:
        root = tmp_path/'corrupted'
        folder = root/'blur'/'2'
        folder.mkdir(parents=True)
        for name in ('1_0.png','1_1.png','2_0.png'):
            (folder/name).write_bytes(base64.b64decode(encoded_image('gray')))
        kwargs['corrupted_root'] = root
    output = tmp_path/'prepared'
    report = prepare(tsv,output,**kwargs)
    return output,report


def test_prepare_preserves_multi_image_order_context_and_labels(tmp_path):
    output,report = prepared(tmp_path)
    records = read_records(output)
    assert report['clean']==dict(questions=2,images=3,embedded_images=3,file_images=0)
    assert report['images_per_question']=={2:1,1:1}
    row = records[0]
    assert list(row['request']['questions']['q']['criteria'])==['A','B']
    assert row['targets']['q']['probabilities']=={'A':1.,'B':0.}
    content = row['request']['state']['messages'][0]['content']
    assert content[0]['text']=='Hint: Use the supplied evidence.'
    images=[output/item['image_url']['url'] for item in content if item['type']=='image_url']
    assert [Image.open(p).getpixel((0,0)) for p in images]==[(255,0,0),(0,0,255)]
    assert len(row['assets'])==2
    assert (output/'clean.jsonl').read_bytes()==(output/'eval.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(tmp_path/'source.tsv',output)


def test_file_images_fallback_and_corruption_pairing(tmp_path):
    output,report = prepared(tmp_path,corrupted=True)
    records = read_records(output)
    assert len(records)==4 and report['records']==4
    assert len({r['group_id'] for r in records})==2
    assert len({r['meta']['benchmark_record_id'] for r in records})==4
    assert (output/'blur_2.jsonl').is_file()
    source = tmp_path/'fallback.tsv'
    row = sample('7')
    row['image']=''
    write_tsv(source,[row])
    (tmp_path/'7_0.png').write_bytes(base64.b64decode(encoded_image('blue')))
    report=prepare(source,tmp_path/'fallback',image_root=tmp_path)
    assert report['clean']['file_images']==1
    read_records(tmp_path/'fallback')


@pytest.mark.parametrize('problem',['invalid_answer','duplicate','bad_image','path_escape','missing_variant','missing_target'])
def test_bad_inputs_do_not_publish_partial_datasets(tmp_path,problem):
    row=sample()
    rows=[row]
    kwargs={}
    if problem=='invalid_answer':
        row['answer']='C'
    elif problem=='duplicate':
        rows.append(dict(row))
    elif problem=='bad_image':
        row['image']='not base64'
    elif problem=='path_escape':
        row['index']='..'
    elif problem=='missing_target':
        row['answer']=''
    else:
        root=tmp_path/'corrupt'
        (root/'blur'/'1').mkdir(parents=True)
        kwargs['corrupted_root']=root
    source=tmp_path/'source.tsv'
    write_tsv(source,rows)
    with pytest.raises(ValueError):
        prepare(source,tmp_path/'output',**kwargs)
    assert not (tmp_path/'output').exists()
    assert not list(tmp_path.glob('.output-*'))


def test_media_hash_and_pair_label_changes_are_detected(tmp_path):
    output,_=prepared(tmp_path,corrupted=True)
    rows=read_records(output)
    rows[1]['targets']['q']['probabilities']={'A':0.,'B':1.}
    bad=output/'changed.jsonl'
    bad.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Paired variants'):
        read_records(bad)
    asset=output/rows[0]['assets'][0]['path']
    asset.write_bytes(b'corrupted file')
    with pytest.raises(ValueError,match='hash mismatch'):
        read_records(output)


class FakeModel:
    def __init__(self):
        self.seen=[]

    def eval(self):
        return self

    def make_compiler(self,media_root,execution):
        def compile(record):
            self.seen.append(record)
            questions=[SimpleNamespace(qid=qid,kind=q['type'],keys=list(reversed(q['criteria'])))
                       for qid,q in record['request']['questions'].items()]
            return SimpleNamespace(questions=questions,inputs={} if execution=='shared_state' else None)
        return SimpleNamespace(compile=compile,media_kwargs={})

    def __call__(self,question):
        # Reversed keys are B, A. This is a correct prediction with confidence .8.
        return torch.tensor([.2,.8]).log()

    def forward_state(self,compiled):
        return [self(q) for q in compiled.questions]


@pytest.mark.parametrize('execution',['question','shared_state'])
def test_evaluation_and_clean_pairing_without_label_leakage(tmp_path,execution):
    data,_=prepared(tmp_path,corrupted=True)
    model=FakeModel()
    output=tmp_path/'results'
    report=evaluate(model,data,output,device='cpu',execution=execution)
    m=report['metrics']['overall']
    assert report['questions']==4 and report['base_questions']==2
    assert m['accuracy']==1 and m['mean_confidence']==pytest.approx(.8)
    assert m['nll']==pytest.approx(-math.log(.8))
    assert m['brier']==pytest.approx(.08)
    assert m['ece']==pytest.approx(.2)
    assert m['confidence_minus_accuracy']==pytest.approx(-.2)
    assert report['paired_changes']['blur/2']['pairs']==2
    assert report['paired_changes']['blur/2']['delta_accuracy']==0
    assert all(set(row)=={'request','assets'} for row in model.seen)
    rows=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    assert all(row['prediction']=='A' and row['target']=={'B':0.,'A':1.} for row in rows)
    assert json.loads((output/'metrics.json').read_text())['questions']==4
    with pytest.raises(FileExistsError):
        evaluate(model,data,output,device='cpu')


def test_calibration_endpoints_high_confidence_errors_and_signed_pair_deltas():
    rows=[]
    for base_id,p in [('1',1.),('2',.9)]:
        probabilities={'A':p,'B':1-p}
        target={'A':float(base_id=='1'),'B':float(base_id=='2')}
        m=distribution_metrics(probabilities,target)
        m['nll']=0. if base_id=='1' else -math.log(.1)
        rows.append(dict(base_id=base_id,corruption='clean',severity=0,probabilities=probabilities,
                         target=target,prediction='A',metrics=m))
    result=aggregate(rows)
    assert result['accuracy']==.5 and result['mean_confidence']==pytest.approx(.95)
    assert result['ece']==pytest.approx(.45)
    assert result['reliability_bins'][-1]['questions']==2
    assert result['high_confidence_error_rate']==.5
    corrupted=dict(rows[0],corruption='blur',severity=1,probabilities={'A':.1,'B':.9},prediction='B',
                   metrics=dict(distribution_metrics({'A':.1,'B':.9},rows[0]['target']),nll=-math.log(.1)))
    change=paired(rows+[corrupted])['blur/1']
    assert change['delta_accuracy']==-1 and change['correct_to_wrong']==1
    assert change['delta_confidence_minus_accuracy']==pytest.approx(.9)
    missing=paired([corrupted])['blur/1']
    assert missing['pairs']==0 and missing['missing_clean']==1


def test_nll_is_stable_for_extreme_logits(tmp_path):
    data,_=prepared(tmp_path)
    class Extreme(FakeModel):
        def __call__(self,q):
            return torch.tensor([1000.,-1000.])
    report=evaluate(Extreme(),data,tmp_path/'out',device='cpu')
    assert report['metrics']['overall']['nll']==2000.
    assert report['metrics']['overall']['accuracy']==0


def test_duplicate_conditions_and_soft_labels_rejected(tmp_path):
    data,_=prepared(tmp_path)
    rows=read_records(data)
    bad=data/'bad.jsonl'
    bad.write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='Duplicate condition'):
        read_records(bad)
    rows[0]['targets']['q']['probabilities']={'A':.7,'B':.3}
    bad.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='one-hot'):
        read_records(bad)


def test_condition_selection_retains_clean_and_rejects_incomplete_grid(tmp_path):
    data,_=prepared(tmp_path,corrupted=True)
    rows=read_records(data)
    assert len(select_records(rows,['blur'],[2]))==4
    with pytest.raises(ValueError,match='absent'):
        select_records(rows,['noise'])
    with pytest.raises(ValueError,match='absent'):
        select_records(rows,severities=[5])
    with pytest.raises(ValueError,match='19 x 5'):
        select_records(rows,require_full_grid=True)
    from copy import deepcopy
    from evaluation.benchc.construct import CORRUPTIONS
    clean=[r for r in rows if r['meta']['corruption']=='clean']
    complete=list(clean)
    for name in CORRUPTIONS:
        for severity in range(1,6):
            for r in clean:
                variant=deepcopy(r)
                variant['meta'].update(corruption=name,severity=severity)
                complete.append(variant)
    assert len(select_records(complete,require_full_grid=True))==2*96
    with pytest.raises(ValueError,match='19 x 5'):
        select_records(complete[:-1],require_full_grid=True)


def test_official_dc_is_not_signed_confidence_gap(tmp_path):
    m=official_pair_metrics({'A':.9,'B':.1},{'A':.2,'B':.8},{'A':1.,'B':0.})
    assert m['dAcc']==-1
    assert m['dC']==pytest.approx(.7)
    assert m['RAS']==pytest.approx(-.7)
    data,_=prepared(tmp_path,corrupted=True)
    report=evaluate(FakeModel(),data,tmp_path/'result',device='cpu')
    assert report['paired_overall']['pairs']==2
    assert report['paired_overall']['RAS']==0
    assert report['metrics']['subset/corrupted']['questions']==2
    details=[json.loads(l) for l in (tmp_path/'result/paired_predictions.jsonl').read_text().splitlines()]
    assert len(details)==2 and all(r['dC']==0 and r['RAS']==0 for r in details)


@pytest.fixture
def official_repo():
    path=os.environ.get('BENCHC_OFFICIAL_REPO')
    if not path:
        pytest.skip('Set BENCHC_OFFICIAL_REPO to the pinned upstream checkout for real-processor tests')
    return Path(path)


def test_metric_parity_with_official_repository(official_repo):
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location('official_metrics',official_repo/'src/metrics.py')
    upstream=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng=np.random.default_rng(42)
    for k in (2,3,4,5):
        cases=[(rng.dirichlet(np.ones(k)),rng.dirichlet(np.ones(k))) for _ in range(25)]
        cases.extend([(np.eye(k)[0],np.eye(k)[1]),(np.ones(k)/k,np.ones(k)/k)])
        for a,b in cases:
            keys=list(map(str,range(k)))
            got=official_pair_metrics(dict(zip(keys,a)),dict(zip(keys,b)),{key:float(key=='0') for key in keys})
            expected=upstream.compute_metrics(a,b,0)
            assert got==pytest.approx(expected,abs=1e-12)


def test_all_official_corruptions_and_severities_smoke(tmp_path,official_repo):
    import numpy as np
    from evaluation.benchc.construct import generate_image, CORRUPTIONS, provenance
    provenance(official_repo)
    rng=np.random.default_rng(1)
    source=tmp_path/'source.png'
    Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(source)
    for name in CORRUPTIONS:
        for level in range(1,6):
            out=tmp_path/name/f'{level}.jpg'
            generate_image((str(official_repo),str(source),str(out),name,level,42,'example'))
            with Image.open(out) as image:
                assert image.size==(64,64) and image.format=='JPEG'


def test_construct_convert_resume_multimage_and_provenance(tmp_path,official_repo):
    from evaluation.benchc.construct import construct
    from evaluation.benchc.prepare import sha256
    source=tmp_path/'input.tsv'
    write_tsv(source,[sample('1',multi=True),sample('2')])
    raw=tmp_path/'raw'
    args=dict(data=source,output=raw,official_repo=official_repo,
              corruptions=['gaussian_noise','brightness'],severities=[1],seed=12,workers=2)
    result=construct(**args)
    assert result['status']=='complete'
    assert result['questions']==2 and result['images']==3
    assert result['expected_corrupted_instances']==4 and result['expected_corrupted_images']==6
    target=raw/'corrupted/gaussian_noise/1/1/image_1.jpg'
    digest=sha256(target)
    target.unlink()
    construct(**args,resume=True)
    assert sha256(target)==digest
    converted=tmp_path/'native'
    report=prepare(raw,converted)
    assert report['construction']['config']['seed']==12
    rows=read_records(converted)
    assert len(rows)==6
    assert all(len(r['assets'])==2 for r in rows if r['meta']['base_id']=='1')
    # PNG originals and JPEG corruptions retain a one-to-one multi-image mapping.
    assert rows[0]['assets'][0]['path'].endswith('.png')
    assert rows[1]['assets'][0]['path'].endswith('.jpg')
    eval_report=evaluate(FakeModel(),converted,tmp_path/'results',device='cpu',corruptions=['brightness'])
    assert eval_report['questions']==4 and eval_report['paired_overall']['pairs']==2
    with pytest.raises(ValueError,match='changed'):
        construct(**dict(args,seed=99),resume=True)
    target.write_bytes(b'broken')
    with pytest.raises(ValueError,match='changed'):
        prepare(raw,tmp_path/'broken-native')
    assert not (tmp_path/'broken-native').exists()
