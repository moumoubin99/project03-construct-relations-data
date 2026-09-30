"""Prespecified descriptive scoring; resampling/sign-flips are conditional sensitivity only."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=tuple(f'C{i:02}' for i in range(1,14))
HIERARCHY={'C02','C03','C04','C05'}
OPPOSITE={'C02':'C03','C03':'C02','C04':'C05','C05':'C04','C06':'C07','C07':'C06'}

def score_labels(truth, predictions):
    if len(truth)!=len(predictions) or any(y not in LABELS for y in truth):
        raise ValueError('Invalid reference alignment')
    n=len(truth)
    if not n:return {'n':0,'agreement':None}
    valid=[p in LABELS for p in predictions]
    success=np.array([int(v and p==y) for y,p,v in zip(truth,predictions,valid)])
    substantive=[p in LABELS[:11] for p in predictions]
    reference_resolvable=[y in LABELS[:11] for y in truth]
    resolved_n=sum(reference_resolvable)
    f1={}
    for c in sorted(set(truth)):
        tp=sum(y==c and p==c for y,p in zip(truth,predictions))
        fp=sum(y!=c and p==c for y,p in zip(truth,predictions))
        fn=sum(y==c and p!=c for y,p in zip(truth,predictions))
        f1[c]=2*tp/(2*tp+fp+fn)
    conf={c:{p:0 for p in LABELS+('INVALID',)} for c in LABELS}
    for y,p in zip(truth,predictions):conf[y][p if p in LABELS else 'INVALID']+=1
    h_n=sum(y in HIERARCHY for y in truth)
    eq_n=truth.count('C01')
    return {'n':n,'correct':int(sum(success)),'agreement':float(np.mean(success)),
            'valid':sum(valid),'invalid':n-sum(valid),'abstentions':predictions.count('C13'),
            'multiple_responses':predictions.count('C12'),'substantive_coverage':sum(substantive)/n,
            'reference_resolvable_n':resolved_n,
            'substantive_correct_resolution':sum(s for s,r in zip(success,reference_resolvable) if r)/resolved_n if resolved_n else None,
            'substantive_conditional_agreement':sum(s for s,r in zip(success,substantive) if r)/sum(substantive) if sum(substantive) else None,
            'macro_f1_present_reference_classes':float(np.mean(list(f1.values()))),
            'f1_by_present_class':f1,'reference_class_counts':dict(Counter(truth)),
            'equivalence_positive_n':eq_n,
            'equivalence_true_positives':sum(y==p=='C01' for y,p in zip(truth,predictions)),
            'false_equivalence':sum(y!='C01' and p=='C01' for y,p in zip(truth,predictions)),
            'non_equivalence_reference_n':n-eq_n,
            'hierarchy_reference_n':h_n,
            'hierarchy_exact_hits':sum(y in HIERARCHY and p==y for y,p in zip(truth,predictions)),
            'false_hierarchy':sum(y not in HIERARCHY and p in HIERARCHY for y,p in zip(truth,predictions)),
            'non_hierarchy_reference_n':n-h_n,
            'opposite_direction_errors':sum(OPPOSITE.get(y)==p and p is not None for y,p in zip(truth,predictions)),
            'confusion':conf}

def cluster_effect(a,b,groups,seed=20260915,bootstrap_draws=10000,sign_draws=100000):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or len(a)!=len(groups) or not len(a):raise ValueError('bad paired arrays')
    levels=sorted(set(groups));d=a-b
    sums=np.array([d[np.array(groups)==g].sum() for g in levels])
    sizes=np.array([groups.count(g) for g in levels])
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,len(levels),(bootstrap_draws,len(levels)))
    bootstrap=sums[selected].sum(axis=1)/sizes[selected].sum(axis=1)
    nonzero=sums[np.abs(sums)>1e-12];k=len(nonzero);observed=abs(sums.sum())
    if k<=18:
        patterns=np.arange(2**k,dtype=np.uint32)[:,None]
        signs=2*((patterns>>np.arange(k,dtype=np.uint32))&1).astype(np.int8)-1
        statistics=np.abs(signs@nonzero)
        p=float(np.mean(statistics>=observed-1e-12));method='exhaustive';draws=2**k
        minimum_grid=1/(2**k)
    else:
        signs=rng.integers(0,2,(sign_draws,k),dtype=np.int8)*2-1
        statistics=np.abs(signs@nonzero)
        p=(1+int(sum(statistics>=observed-1e-12)))/(sign_draws+1);method='monte_carlo_plus_one';draws=sign_draws
        minimum_grid=1/(sign_draws+1)
    leave=[float((sums.sum()-s)/(len(d)-n)) for s,n in zip(sums,sizes) if len(d)>n]
    return {'n':len(d),'components':len(levels),'component_sizes':sizes.tolist(),
            'delta':float(np.mean(d)), 'component_equal_delta':float(np.mean(sums/sizes)),
            'conditional_cluster_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'leave_one_component_out_range':[min(leave),max(leave)] if leave else None,
            'conditional_sign_flip_p':p,'nonzero_component_sums':k,'sign_method':method,
            'sign_draws':draws,'p_grid_lower_bound':minimum_grid,
            'inference_status':'exploratory conditional sensitivity; not confirmatory significance',
            'assumptions':'Sign-flip requires joint sign symmetry/exchangeability of component effects, not merely zero mean; component independence and population sampling are unverified.'}

def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);out=[None]*len(pvalues);running=0
    for rank,i in enumerate(order):
        running=max(running,(len(pvalues)-rank)*pvalues[i]);out[i]=min(1.,running)
    return out

def analyze_records(records, references, components, subsets):
    refs={r['item_id']:r['option_id'] for r in references if r['partition']=='main'}
    ids=sorted(refs);assert len(ids)==66
    cells={}
    for r in records:
        if r['phase']!='formal':continue
        key=(r['provider'],r['arm'],r['item_id'])
        if key in cells:raise ValueError('duplicate initial formal cell')
        if r['item_id'] not in refs:raise ValueError('unexpected formal ID')
        cells[key]=r['label'] if r['label'] in LABELS and r.get('scorable',True) else None
    expected={(p,a,i) for p in ('qwen','glm','deepseek') for a in ('D','F','L') for i in ids}
    if set(cells)!=expected:raise ValueError(f'Formal set incomplete or extra: missing={len(expected-set(cells))},extra={len(set(cells)-expected)}')
    groups={i:c['component_id'] for c in components for i in c['members_all96'] if i in refs}
    assert set(groups)==set(ids) and len(set(groups.values()))==33
    systems=[];contrasts=[];subresults=[];rows=[]
    for provider in ('qwen','glm','deepseek'):
        successes={}
        for arm in ('D','F','L'):
            predictions=[cells[(provider,arm,i)] for i in ids]
            m=score_labels([refs[i] for i in ids],predictions)
            systems.append({'provider':provider,'arm':arm,**m})
            successes[arm]=[int(p==refs[i]) for i,p in zip(ids,predictions)]
            for i,p in zip(ids,predictions):rows.append({'provider':provider,'arm':arm,'item_id':i,'reference':refs[i],
                'prediction':p,'correct':int(p==refs[i]),'component':groups[i]})
        for baseline in ('D','L'):
            contrasts.append({'provider':provider,'comparison':'F-'+baseline,
                              **cluster_effect(successes['F'],successes[baseline],[groups[i] for i in ids])})
        for key,subset in subsets.items():
            if key=='practice6':continue
            selected=sorted(set(subset['ids'])&set(ids))
            for arm in ('D','F','L'):
                predictions=[cells[(provider,arm,i)] for i in selected]
                m=score_labels([refs[i] for i in selected],predictions)
                subresults.append({'provider':provider,'arm':arm,'subset':key,
                                   **{k:m.get(k) for k in ('n','correct','agreement','invalid','false_equivalence','false_hierarchy')}})
    for row,p in zip(contrasts,holm([r['conditional_sign_flip_p'] for r in contrasts])):
        row['holm_six_contrasts_conditional_p']=p
    result={'complete_formal_initial_calls':len(cells),'unique_pairs':len(ids),'systems':systems,'contrasts':contrasts,
            'subsets':subresults,'main_class_counts':dict(Counter(refs.values())),
            'dummy_label':'C10','dummy_correct':sum(y=='C10' for y in refs.values()),
            'interpretation':'Agreement with one user-reported independent reviewer under the same material version. Not expert interrater reliability, objective construct truth, externally held-out accuracy or confirmatory efficacy.'}
    return result,rows

def save_outputs(result, rows):
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,data in [('pair_predictions.csv',rows),('subset_results.csv',result['subsets'])]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    compact=[]
    for r in result['systems']:
        compact.append({k:v for k,v in r.items() if not isinstance(v,dict)})
    with (out/'main_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)

