"""Deterministic checks on the distributed numeric records, with no API calls."""
from pathlib import Path
import csv
import json
import re
import unicodedata
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def rows(name): return list(csv.DictReader((ROOT/name).open(encoding='utf-8-sig')))
def norm(s):
    s=unicodedata.normalize('NFKC',s).casefold().replace('behaviour','behavior').replace('organisation','organization')
    return ' '.join(re.sub('[^a-z0-9]+',' ',s).split())

expected={'initial':[36,33,37,36,35,32,28,32,31], 'recovered':[36,33,37,36,35,33,28,32,31]}
checks=[]
for role in expected:
    data=rows(f'followup/{role}/pair_predictions.csv')
    assert len(data)==594
    assert len({(r['provider'],r['arm'],r['item_id']) for r in data})==594
    assert len({r['item_id'] for r in data})==66
    counts=[]
    for provider in ['qwen','glm','deepseek']:
        for arm in ['D','F','L']:
            subset=[r for r in data if (r['provider'],r['arm'])==(provider,arm)]
            assert len(subset)==66
            assert all(int(r['correct'])==int(r['prediction']==r['reference']) for r in subset)
            counts.append(sum(r['prediction']==r['reference'] for r in subset))
    assert counts==expected[role]
    assert sum(r['operational_no_response']=='True' for r in data)==(1 if role=='initial' else 0)
    by_id=defaultdict(list)
    for r in data:by_id[r['item_id']].append(r)
    unmatched=[rs for rs in by_id.values() if all(r['reference']!=r['prediction'] for r in rs)]
    assert len(unmatched)==12
    assert sum(len({r['prediction'] for r in rs})==1 for rs in unmatched)==4
    for contrast in read(f'followup/{role}/metrics.json')['contrasts']:
        a,b=contrast['comparison'].split('-'); provider=contrast['provider']
        arows={r['item_id']:r for r in data if r['provider']==provider and r['arm']==a}
        brows={r['item_id']:r for r in data if r['provider']==provider and r['arm']==b}
        delta=sum(int(arows[i]['correct'])-int(brows[i]['correct']) for i in arows)/66
        assert abs(delta-contrast['delta'])<1e-12
        assert len({r['component'] for r in arows.values()})==33
    checks.append({'analysis_role':role,'matches':counts,'paired_contrasts':6})

strict=read('coverage/pair_candidates.json'); expanded=read('coverage/expanded_search_leads.json')
assert len(strict)==len(expanded)==96
strict_both=lambda rs:sum(all(r['sides'][s]['candidates'] for s in ['a','b']) for r in rs)
expanded_both=lambda rs:sum(all(r['sides'][s]['expanded_leads'] for s in ['a','b']) for r in rs)
assert strict_both(strict)==6
assert strict_both([r for r in strict if r['in_formal_66']])==5
assert expanded_both(expanded)==17
assert expanded_both([r for r in expanded if r['in_formal_66']])==12
edges=read('coverage/bcio_positive_edges.json')
assert len(edges)==len({(r['child_id'],r['parent_id']) for r in edges})==93
assert len({r[k] for r in edges for k in ['child_id','parent_id']})==98
assert sum(any(norm(r['parent_label']) in norm(d) for d in r['child_definition']) for r in edges)==85
changes=rows('followup/paired_error_changes.csv')
assert len(changes)==12
glm_fl=next(r for r in changes if (r['analysis_role'],r['provider'],r['comparison'])==('initial','glm','F-L'))
assert (int(glm_fl['changed_outcomes']),int(glm_fl['both_returned_valid_n']),int(glm_fl['returned_label_changes']))==(8,65,7)
assert 'changed_labels' not in glm_fl
report={'passed':True,'scope':'numeric and retrieval record consistency, not semantic validation',
        'followup':checks,'strict_both':[6,5],'expanded_both':[17,12],
        'distinct_edges':93,'distinct_nodes':98,'parent_name_cues':85,
        'initial_glm_FL_changed_outcomes':8,'initial_glm_FL_returned_changes':'7/65','model_calls':0}
print(json.dumps(report,ensure_ascii=False,indent=2))
