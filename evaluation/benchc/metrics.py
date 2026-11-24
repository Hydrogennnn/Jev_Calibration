"""Top-label calibration and Bench-C's official paired diagnostic definitions."""
from collections import defaultdict
import math


def official_pair_metrics(clean, corrupted, target):
    """Match src/metrics.py at official commit 9ff3ea6279bf8576ad9720944176202134e914d7.

    dC uses per-example absolute confidence/correctness error, NOT binned ECE.
    The official EPS clipping is confined to these four compatibility metrics.
    """
    if set(clean)!=set(corrupted) or set(clean)!=set(target):
        raise ValueError('Paired candidate sets differ')
    keys = list(clean)
    def normalized(values):
        p = [max(float(x),1e-12) for x in values]
        return [x/sum(p) for x in p]
    def entropy(p):
        p = normalized(p)
        return -sum(x*math.log(x) for x in p)/math.log(len(p))
    p = normalized([clean[k] for k in keys])
    q = normalized([corrupted[k] for k in keys])
    gold = next(i for i,k in enumerate(keys) if target[k]==1.)
    a = float(max(range(len(p)),key=p.__getitem__)==gold)
    b = float(max(range(len(q)),key=q.__getitem__)==gold)
    ds = entropy(q)-entropy(p)
    dc = abs(max(q)-b)-abs(max(p)-a)
    return dict(dAcc=b-a, dS=ds, dC=dc,
                RAS=-dc-max(0.,dc)*max(0.,-ds)-max(0.,ds)*max(0.,-dc))


def paired_details(rows):
    clean = {r['base_id']:r for r in rows if r['corruption']=='clean'}
    for row in rows:
        if row['corruption']=='clean' or row['base_id'] not in clean:
            continue
        original = clean[row['base_id']]
        if original['target']!=row['target']:
            raise ValueError('Paired labels differ')
        yield dict(base_id=row['base_id'],corruption=row['corruption'],severity=row['severity'],
                   **official_pair_metrics(original['probabilities'],row['probabilities'],row['target']))


def distribution_metrics(probabilities, target):
    keys = list(probabilities)
    if set(keys) != set(target) or len(keys) < 2:
        raise ValueError('Expected at least two matching candidates')
    p, y = [probabilities[k] for k in keys], [target[k] for k in keys]
    if any(not math.isfinite(v) or v < 0 for v in p+y) or not math.isclose(sum(p), 1., abs_tol=1e-6):
        raise ValueError('Invalid candidate probabilities')
    if y.count(1.) != 1 or sum(v != 0 for v in y) != 1:
        raise ValueError('Bench-C requires one-hot targets')
    predicted, gold = max(range(len(p)), key=p.__getitem__), y.index(1.)
    return dict(accuracy=float(predicted==gold), confidence=max(p),
                brier=sum((a-b)**2 for a,b in zip(p,y)),
                normalized_entropy=-sum(v*math.log(v) for v in p if v)/math.log(len(p)))


def aggregate(rows, bins=10, high_confidence=.9):
    if type(bins) is not int or bins < 1 or not 0 <= high_confidence <= 1:
        raise ValueError('Invalid confidence bin count or threshold')
    if not rows:
        raise ValueError('No scored questions')
    groups = [[] for _ in range(bins)]
    for row in rows:
        m = row['metrics']
        groups[min(int(m['confidence']*bins), bins-1)].append(m)
    reliability = []
    for i, members in enumerate(groups):
        reliability.append(dict(lower=i/bins, upper=(i+1)/bins, questions=len(members),
                                confidence=sum(m['confidence'] for m in members)/len(members) if members else None,
                                accuracy=sum(m['accuracy'] for m in members)/len(members) if members else None))
    n = len(rows)
    result = {'questions': n}
    for key in ('accuracy','nll','brier','confidence','normalized_entropy'):
        result['mean_confidence' if key=='confidence' else key] = sum(r['metrics'][key] for r in rows)/n
    result['correct'] = sum(int(r['metrics']['accuracy']) for r in rows)
    result['confidence_minus_accuracy'] = result['mean_confidence']-result['accuracy']
    result['ece'] = sum(b['questions']*abs(b['confidence']-b['accuracy']) for b in reliability if b['questions'])/n
    result['mce'] = max(abs(b['confidence']-b['accuracy']) for b in reliability if b['questions'])
    high = [r for r in rows if r['metrics']['confidence'] >= high_confidence]
    wrong = [r for r in rows if not r['metrics']['accuracy']]
    result.update(high_confidence_threshold=high_confidence, high_confidence_questions=len(high),
                  high_confidence_errors=sum(not r['metrics']['accuracy'] for r in high),
                  high_confidence_error_rate=sum(not r['metrics']['accuracy'] for r in high)/len(high) if high else None,
                  mean_confidence_on_errors=sum(r['metrics']['confidence'] for r in wrong)/len(wrong) if wrong else None,
                  reliability_bins=reliability)
    return result


def paired(rows):
    clean, corrupted = {}, defaultdict(list)
    for row in rows:
        if row['corruption']=='clean':
            if row['base_id'] in clean:
                raise ValueError('Duplicate clean base_id')
            clean[row['base_id']] = row
        else:
            corrupted[f"{row['corruption']}/{row['severity']}"].append(row)
    result = {}
    for name, members in sorted(corrupted.items()):
        pairs, missing = [], []
        for row in members:
            original = clean.get(row['base_id'])
            if original is None:
                missing.append(row['base_id'])
                continue
            if original['target'] != row['target'] or set(original['probabilities']) != set(row['probabilities']):
                raise ValueError(f'Labels/candidates changed in pair {name}/{row["base_id"]}')
            pairs.append((original,row))
        output = dict(pairs=len(pairs), missing_clean=len(missing), missing_clean_base_ids=missing)
        if pairs:
            n = len(pairs)
            official = [official_pair_metrics(a['probabilities'],b['probabilities'],a['target']) for a,b in pairs]
            for key in ('dAcc','dS','dC','RAS'):
                output[key] = sum(r[key] for r in official)/n
            for key in ('accuracy','confidence','normalized_entropy','nll','brier'):
                output[f'delta_{key}'] = sum(b['metrics'][key]-a['metrics'][key] for a,b in pairs)/n
            output.update(
                correct_to_wrong=sum(bool(a['metrics']['accuracy']) and not b['metrics']['accuracy'] for a,b in pairs),
                wrong_to_correct=sum(not a['metrics']['accuracy'] and bool(b['metrics']['accuracy']) for a,b in pairs),
                top1_consistency=sum(a['prediction']==b['prediction'] for a,b in pairs)/n,
                delta_confidence_minus_accuracy=output['delta_confidence']-output['delta_accuracy'])
        result[name] = output
    return result


def summarize(rows, bins=10, high_confidence=.9):
    buckets = defaultdict(list)
    for row in rows:
        for key in ('overall', f"condition/{row['corruption']}/{row['severity']}",
                    'subset/clean' if row['corruption']=='clean' else 'subset/corrupted',
                    f"corruption/{row['corruption']}", f"severity/{row['severity']}",
                    f"category/{row['category']}", f"l2_category/{row['l2_category']}"):
            buckets[key].append(row)
    return {name:aggregate(members,bins,high_confidence) for name,members in sorted(buckets.items())}
