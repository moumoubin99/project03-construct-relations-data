"""Public-source snapshot and lexical candidate audit; never labels research truth.

No model endpoint, credential, or private upload is used. Existing study inputs are
read-only. All generated outputs remain under this new audit directory.
"""
import csv
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RAW = ROOT / '.aris/feasibility'
RAW.mkdir(parents=True, exist_ok=True)
NS = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns',
      'owl': 'http://www.w3.org/2002/07/owl',
      'rdfs': 'http://www.w3.org/2000/01/rdf-schema',
      'obo': 'http://purl.obolibrary.org/obo/',
      'oi': 'http://www.geneontology.org/formats/oboInOwl'}
ABOUT, RESOURCE = '{'+NS['rdf']+'}about', '{'+NS['rdf']+'}resource'

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fetch(url, name, accept='application/json'):
    path = RAW/name
    meta_path = RAW/(name+'.meta.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text('utf-8'))
        assert digest(path) == meta['sha256'], f'Cached snapshot changed: {name}'
        return path, meta
    request = urllib.request.Request(url, headers={'User-Agent':'construct-coverage-research/1.0', 'Accept':accept})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(12_000_001)
        if len(data)>12_000_000:
            raise ValueError('Public-source size limit exceeded')
        meta = {'requested_url':url, 'final_url':response.url, 'status':response.status,
                'content_type':response.headers.get('Content-Type'), 'bytes':len(data),
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'sha256':hashlib.sha256(data).hexdigest()}
    path.write_bytes(data)
    save(meta_path, meta)
    return path, meta

def get_public():
    commit_path, _ = fetch('https://api.github.com/repos/HumanBehaviourChangeProject/ontologies/commits?path=MechanismOfAction%2Fbcio_moa.owl&per_page=1','bcio_commit.json')
    commit = json.loads(commit_path.read_text('utf-8'))[0]['sha']
    jobs = [
        ('bcio_moa.owl', f'https://raw.githubusercontent.com/HumanBehaviourChangeProject/ontologies/{commit}/MechanismOfAction/bcio_moa.owl', 'application/rdf+xml'),
        ('bcio_license.txt', f'https://raw.githubusercontent.com/HumanBehaviourChangeProject/ontologies/{commit}/LICENSE', 'text/plain'),
        ('cognitive_atlas.json', 'https://www.cognitiveatlas.org/api/v-alpha/concept?format=json', 'application/json'),
    ]
    def run(job):
        name,url,accept=job
        try:
            _,meta=fetch(url,name,accept)
            return name,{'found':'yes',**meta}
        except Exception as exc:
            return name,{'found':'unknown','url':url,'error':str(exc)[:300]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshots=dict(pool.map(run,jobs))
    snapshots['bcio_commit']=commit
    save(RAW/'snapshots.json',snapshots)
    print(json.dumps(snapshots,ensure_ascii=False,indent=2))

def normalize(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=text.replace('behaviour','behavior').replace('organisation','organization')
    return ' '.join(re.sub(r'[^a-z0-9]+',' ',text).split())

def original_variants(label):
    # No synonym guessing. Removing parentheses is only a broad lexical search,
    # never an assertion that the qualifier is semantically dispensable.
    raw=[('full_name',label),('qualifier_removed',re.sub(r'\([^)]*\)','',label))]
    seen=set()
    for kind,value in raw:
        n=normalize(value)
        if n and n not in seen:
            seen.add(n)
            yield kind,n

def parse_bcio():
    tree=ET.parse(RAW/'bcio_moa.owl')
    classes={}
    for node in tree.findall('owl:Class',NS):
        uri=node.get(ABOUT)
        if not uri: continue
        def texts(tag):
            return list(dict.fromkeys((n.text or '').strip() for n in node.findall(tag,NS) if (n.text or '').strip()))
        labels=texts('rdfs:label')
        if not labels: continue
        classes[uri]={'id':uri,'source':'BCIO_MoA','labels':labels,
            'definitions':texts('obo:IAO_0000115'),
            'synonyms':texts('obo:IAO_0000118')+texts('oi:hasExactSynonym'),
            'examples':texts('obo:IAO_0000112'),
            'parents':[n.get(RESOURCE) for n in node.findall('rdfs:subClassOf',NS) if n.get(RESOURCE)],
            'status':texts('obo:IAO_0000114'),
            'deprecated':texts('owl:deprecated')}
    version=tree.find('owl:Ontology/owl:versionIRI',NS)
    return classes,version.get(RESOURCE) if version is not None else None

def parse_cogat():
    data=json.loads((RAW/'cognitive_atlas.json').read_text('utf-8'))
    rows=data if isinstance(data,list) else data.get('results',[])
    classes={}
    for row in rows:
        aliases=row.get('alias') or []
        if isinstance(aliases,str): aliases=[a.strip() for a in re.split(r'[,;]',aliases) if a.strip()]
        aliases=[a for a in aliases if isinstance(a,str)]
        classes[row['id']]={'id':row['id'],'source':'Cognitive_Atlas','labels':[row['name']],
            'definitions':[row.get('definition_text') or ''], 'synonyms':aliases,
            'examples':[], 'parents':[], 'status':[], 'raw_keys':list(row)}
    return classes,{'response_type':type(data).__name__,'count':len(rows),'next':data.get('next') if isinstance(data,dict) else None}

def candidates(label, index, nodes):
    found={}
    for variant,normalized in original_variants(label):
        for source,node_id,field in index.get(normalized,[]):
            key=(source,node_id)
            if key not in found:
                node=nodes[key]
                found[key]={'source':source,'id':node_id,'label':node['labels'][0],
                    'match_field':field,'name_variant':variant,'matched_text':normalized,
                    'definitions':node['definitions'], 'parents':node['parents'],
                    'status':node['status'],'interpretation':'lexical_candidate_not_validated_mapping'}
    return list(found.values())

def ancestors(node_id,bcio):
    todo=list(bcio.get(node_id,{}).get('parents',[])); seen=set()
    while todo:
        current=todo.pop()
        if current in seen: continue
        seen.add(current)
        todo.extend(bcio.get(current,{}).get('parents',[]))
    return seen

def audit():
    bcio,version=parse_bcio()
    cogat,cmeta=parse_cogat()
    nodes={(n['source'],n['id']):n for n in list(bcio.values())+list(cogat.values())}
    index=defaultdict(list)
    for (source,node_id),n in nodes.items():
        if n.get('deprecated') and 'true' in n['deprecated']:continue
        for field in ('labels','synonyms'):
            for text in n[field]:
                index[normalize(text)].append((source,node_id,field))
        # Examples are kept separate, not counted in label/synonym coverage.
    template=BASE/'project03_experiments_20260914/reviewer_1/reference_template_96_pairs.csv'
    mapping_path=BASE/'project03_experiments_20260914/private_mapping/DO_NOT_GIVE_TO_REVIEWERS.csv'
    prepared_path=BASE/'supplementary_20260915/source_context/combined.json'
    predictions_path=BASE/'supplementary_20260915/analysis/recovered/pair_predictions.csv'
    with template.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    with mapping_path.open(encoding='utf-8-sig',newline='') as f:mapping={r['blind_id']:r for r in csv.DictReader(f)}
    prepared={r['id']:r for r in json.loads(prepared_path.read_text('utf-8'))['records']}
    with predictions_path.open(encoding='utf-8-sig',newline='') as f:
        # Read IDs ONLY, never human/reference/prediction labels in mapping decisions.
        formal={r['item_id'] for r in csv.DictReader(f)}
    assert len(original)==96 and len(formal)==66 and len(mapping)==96
    rows=[]
    for pair in original:
        rid=pair['blind_id']; row={'id':rid,**mapping[rid],'in_formal_66':rid in formal,'sides':{}}
        for number,letter in ((1,'a'),(2,'b')):
            label=pair[f'construct_{number}_label']
            row['sides'][letter]={'original_label':label,'original_discipline':pair[f'construct_{number}_discipline'],
                'prepared_material':prepared[rid].get(letter), 'candidates':candidates(label,index,nodes)}
        # Candidate graph checks only; scope mismatch may invalidate them.
        graph=[]
        for a in row['sides']['a']['candidates']:
            for b in row['sides']['b']['candidates']:
                if a['source']!=b['source']:continue
                relation=None
                if a['id']==b['id']:relation='same_resource_node_not_construct_equivalence'
                elif a['source']=='BCIO_MoA':
                    if b['id'] in ancestors(a['id'],bcio):relation='a_subclass_of_b'
                    elif a['id'] in ancestors(b['id'],bcio):relation='b_subclass_of_a'
                if relation:graph.append({'a':a['id'],'b':b['id'],'source':a['source'],'relation':relation})
        row['candidate_graph_links']=graph
        rows.append(row)
    def counts(subset):
        result={'pairs':len(subset),'side_occurrences':2*len(subset)}
        for source in ('BCIO_MoA','Cognitive_Atlas','either'):
            def hit(side):return any(source=='either' or c['source']==source for c in side['candidates'])
            flags=[[hit(r['sides'][s]) for s in ('a','b')] for r in subset]
            result[source]={'sides_with_name_candidates':sum(sum(f) for f in flags),
                'pairs_with_both_name_candidates':sum(all(f) for f in flags),
                'pairs_with_any_name_candidate':sum(any(f) for f in flags)}
        result['pairs_with_candidate_graph_links']=sum(bool(r['candidate_graph_links']) for r in subset)
        result['pairs_with_candidate_subclass_links']=sum(any('subclass' in g['relation'] for g in r['candidate_graph_links']) for r in subset)
        return result
    native={k:v for k,v in bcio.items() if k.startswith('http://humanbehaviourchange.org/ontology/BCIO_') and not v.get('deprecated')}
    native_edges=[{'child':k,'child_label':v['labels'][0],'parent':p,'parent_label':native[p]['labels'][0]}
                  for k,v in native.items() for p in v['parents'] if p in native and v['definitions'] and native[p]['definitions']]
    report={'scope':'candidate retrieval, not verified relation coverage or gold-label creation',
        'bcio_version':version,'bcio_labeled_classes_including_imports':len(bcio),
        'bcio_native_classes':len(native),'native_edges_with_two_defined_endpoints':len(native_edges),
        'cognitive_atlas':cmeta,'original_96':counts(rows),
        'formal_66_original_name_screen':counts([r for r in rows if r['in_formal_66']]),
        'matching':'NFKC/case/punctuation and behaviour spelling normalization; full or parenthesis-removed name equals label/alias; examples excluded',
        'limitations':['Aliases can be broad; qualifiers must be reviewed.','Counts are occurrences, not independent constructs.',
            'Cognitive Atlas parent edges not available in the concept response.','No model outputs or human ratings used to choose mappings.',
            'Formal66 screen uses original names as retrieval keys; prepared names/definitions must be reviewed separately.'],
        'input_hashes':{str(p):digest(p) for p in (template,mapping_path,prepared_path,predictions_path,
          BASE/'project03_revision/plos_revision_20260914/replication/data/BENCHMARK_v0.4.yaml',
          BASE/'project03_revision/plos_revision_20260914/paper/manuscript/main.tex')}}
    save(ROOT/'coverage_summary.json',report)
    save(ROOT/'pair_candidates.json',rows)
    save(RAW/'parsed_nodes.json',list(nodes.values()))
    save(RAW/'bcio_native_edges.json',native_edges)
    lines=['# 96对原构念的名称候选清单','',
      '这不是答案表。命中仅指名称/别名检索结果；未命中不表示公开资料一定不存在。66题沿用原名称检索，实际已修订的材料保存在JSON中供语境审核。','',
      '| 原题 / 现编号 | 正式66题 | 构念A及候选 | 构念B及候选 | 资源图线索 |',
      '|---|---|---|---|---|']
    def fmt(side):
        hit='；'.join(f"{c['source']}:{c['label']} ({c['id'].split('/')[-1]};{c['match_field']};{c['name_variant']})" for c in side['candidates'])
        return (side['original_label']+' → '+(hit or '无名称命中')).replace('|','/')
    for r in rows:
        lines.append('| '+' | '.join([r['pair_id']+' / '+r['id'],'是' if r['in_formal_66'] else '否',fmt(r['sides']['a']),fmt(r['sides']['b']),'; '.join(g['relation'] for g in r['candidate_graph_links']) or '未取得关系'])+' |')
    (ROOT/'96对构念逐项候选清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('GRAPH_CANDIDATES')
    for r in rows:
        if r['candidate_graph_links']:
            print(r['id'],r['pair_id'],r['sides']['a']['original_label'],'|',r['sides']['b']['original_label'],r['candidate_graph_links'])

def expand():
    """Example phrases and broadened names are search leads, not validated aliases."""
    bcio,_=parse_bcio(); cogat,_=parse_cogat()
    nodes={(n['source'],n['id']):n for n in list(bcio.values())+list(cogat.values())}
    index=defaultdict(list)
    for (source,nid),node in nodes.items():
        for field in ('labels','synonyms'):
            for text in node[field]:index[normalize(text)].append((source,nid,field,text))
        for text in node['examples']:
            for phrase in re.findall(r'"([^"]+)"',text):
                index[normalize(phrase)].append((source,nid,'example_not_equivalence',text))
    rows=json.loads((ROOT/'pair_candidates.json').read_text('utf-8'))
    for row in rows:
        for side in row['sides'].values():
            variants=list(original_variants(side['original_label']))
            stripped=re.sub(r'\([^)]*\)','',side['original_label']).strip()
            if stripped.lower().startswith('big five '):variants.append(('big_five_prefix_removed',normalize(stripped[9:])))
            if ' / ' in stripped or '/' in stripped:
                for v in re.split(r'\s*/\s*',stripped):
                    if len(v.split())>1:variants.append(('slash_alternative_search_only',normalize(v)))
            expanded=[];seen=set()
            for variant,norm in variants:
                for source,nid,field,text in index.get(norm,[]):
                    key=source,nid,field,text
                    if key in seen:continue
                    seen.add(key); node=nodes[(source,nid)]
                    expanded.append({'source':source,'id':nid,'label':node['labels'][0],
                        'match_field':field,'name_variant':variant,'source_text':text,
                        'definitions':node['definitions'],'parents':node['parents'],
                        'status':node['status'],'interpretation':'search_lead_not_relation_label'})
            side['expanded_leads']=expanded
        graph=[]
        for a in row['sides']['a']['expanded_leads']:
            for b in row['sides']['b']['expanded_leads']:
                if a['source']!=b['source']:continue
                rel=None
                if a['id']==b['id']:rel='same_resource_node_not_construct_equivalence'
                elif a['source']=='BCIO_MoA':
                    if b['id'] in ancestors(a['id'],bcio):rel='a_subclass_of_b'
                    elif a['id'] in ancestors(b['id'],bcio):rel='b_subclass_of_a'
                if rel:
                    g={'source':a['source'],'a':a['id'],'b':b['id'],'a_label':a['label'],'b_label':b['label'],'relation':rel}
                    if g not in graph:graph.append(g)
        row['expanded_graph_leads']=graph
    save(ROOT/'expanded_search_leads.json',rows)
    selected=[r for r in rows if all(r['sides'][s]['expanded_leads'] for s in ('a','b'))]
    save(ROOT/'both_endpoint_leads.json',selected)
    counts={}
    for group,subset in [('original_96',rows),('formal_66',[r for r in rows if r['in_formal_66']])]:
        counts[group]={'pairs':len(subset),'sides_with_leads':sum(bool(r['sides'][s]['expanded_leads']) for r in subset for s in ('a','b')),
            'both_endpoint_leads':sum(all(r['sides'][s]['expanded_leads'] for s in ('a','b')) for r in subset),
            'graph_lead_pairs':sum(bool(r['expanded_graph_leads']) for r in subset)}
    save(ROOT/'expanded_summary.json',counts)
    print(json.dumps(counts,ensure_ascii=False,indent=2))
    for r in selected:
        print('\n',r['id'],r['pair_id'],'FORMAL',r['in_formal_66'])
        for s in ('a','b'):
            print(s,r['sides'][s]['original_label'])
            for c in r['sides'][s]['expanded_leads']:
                print(' ',c['id'],c['label'],c['match_field'],c['source_text'])
        print('GRAPH',r['expanded_graph_leads'])

def detail():
    urls=[('cogat_selective_attention_detail.json','https://www.cognitiveatlas.org/api/v-alpha/concept?id=trm_4a3fd79d0b043&format=json'),
          ('cogat_attention_detail.json','https://www.cognitiveatlas.org/api/v-alpha/concept?id=trm_4a3fd79d09902&format=json')]
    for name,url in urls:
        try:
            path,meta=fetch(url,name)
            data=json.loads(path.read_text('utf-8'))
            print(name,json.dumps(data,ensure_ascii=False)[:20000])
        except Exception as exc:print(name,type(exc).__name__,str(exc)[:200])

def family_inventory():
    bcio,version=parse_bcio()
    prefix='http://humanbehaviourchange.org/ontology/'
    roots=[('自我效能','BCIO_050898'),('行为动机','BCIO_006133'),('目标','BCIO_006049'),
           ('感知规范','BCIO_006039'),('行为意向','BCIO_006016'),('态度','BCIO_050328'),
           ('结果信念','BCIO_050217')]
    eligible={k:v for k,v in bcio.items() if k.startswith(prefix+'BCIO_') and
              'Published' in v['status'] and v['definitions'] and 'true' not in v['deprecated']}
    members={name:{k for k in eligible if k==prefix+rid or prefix+rid in ancestors(k,bcio)} for name,rid in roots}
    records={}
    counts={}
    for family,ids in members.items():
        edges=[]
        for child in sorted(ids):
            for parent in eligible[child]['parents']:
                if parent not in ids:continue
                key=(child,parent)
                edges.append(key)
                if key not in records:
                    records[key]={'id':f'BCIO_EDGE_{len(records)+1:03d}','child_id':child,'parent_id':parent,
                        'child_label':eligible[child]['labels'][0],'parent_label':eligible[parent]['labels'][0],
                        'child_definition':eligible[child]['definitions'],'parent_definition':eligible[parent]['definitions'],
                        'source_assertion':'rdfs:subClassOf','families':[],
                        'definition_names_parent':any(normalize(eligible[parent]['labels'][0]) in normalize(d) for d in eligible[child]['definitions']),
                        'use_status':'candidate_positive_edges_not_complete_classification_benchmark'}
                records[key]['families'].append(family)
        counts[family]={'defined_published_nodes':len(ids),'direct_edges':len(edges),'root_id':prefix+dict(roots)[family]}
    rows=list(records.values())
    # Count explicitly encoded disjointness only, never infer negatives from absent edges.
    selected_nodes=set().union(*members.values())
    tree=ET.parse(RAW/'bcio_moa.owl'); disjoint=[]
    for node in tree.findall('owl:Class',NS):
        a=node.get(ABOUT)
        if a not in selected_nodes:continue
        for e in node.findall('owl:disjointWith',NS):
            b=e.get(RESOURCE)
            if b in selected_nodes:disjoint.append([a,b])
    summary={'version':version,'roots':counts,'distinct_direct_edges':len(rows),
        'distinct_endpoint_nodes':len({r[k] for r in rows for k in ('child_id','parent_id')}),
        'edges_whose_child_definition_names_parent':sum(r['definition_names_parent'] for r in rows),
        'explicit_named_disjointWith_pairs_in_selected_nodes':disjoint,
        'negative_sampling':'Not performed; missing edges are unknown, not negatives.',
        'scope':'Defined Published native BCIO classes in seven topical subtrees; root choice is an analyst scope choice, not randomized sampling.'}
    save(ROOT/'公开构念层级候选池.json',rows)
    save(ROOT/'family_inventory_summary.json',summary)
    text=['# 公开构念层级候选池（不是模型实验结果）','',
       '来源为固定BCIO版本的正式父子边。不是原96题的答案；所有记录都是层级正例，尚无可用的完整分类负例集。','',
       '| 编号 | 构念家族 | 较具体概念 | 较一般概念 | 子定义含父名称 |','|---|---|---|---|---|']
    for r in rows:
        text.append('| '+' | '.join([r['id'],'/'.join(r['families']),r['child_label'],r['parent_label'],'是' if r['definition_names_parent'] else '否'])+' |')
    (ROOT/'公开构念层级候选池.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def cogat_pair_relations():
    rows=json.loads((ROOT/'both_endpoint_leads.json').read_text('utf-8'))
    ids=set()
    for row in rows:
        a=[c for c in row['sides']['a']['expanded_leads'] if c['source']=='Cognitive_Atlas']
        b=[c for c in row['sides']['b']['expanded_leads'] if c['source']=='Cognitive_Atlas']
        if a and b:ids.update(c['id'] for c in a+b)
    def get(nid):
        try:
            p,_=fetch('https://www.cognitiveatlas.org/api/v-alpha/concept?format=json&id='+nid,'cogat_'+nid+'.json')
            return nid,json.loads(p.read_text('utf-8'))
        except Exception as exc:return nid,{'error':str(exc)[:200]}
    with ThreadPoolExecutor(max_workers=3) as pool: details=dict(pool.map(get,sorted(ids)))
    links=[]
    for row in rows:
        a={c['id'] for c in row['sides']['a']['expanded_leads'] if c['source']=='Cognitive_Atlas'}
        b={c['id'] for c in row['sides']['b']['expanded_leads'] if c['source']=='Cognitive_Atlas'}
        for aid in sorted(a):
            d=details.get(aid,{})
            for rel in d.get('relationships',[]):
                if rel['id'] in b:
                    links.append({'item_id':row['id'],'from_id':aid,'to_id':rel['id'],
                        'relationship':rel.get('relationship'),'direction':rel.get('direction'),
                        'interpretation':'resource_assertion_only; original contexts require separate review'})
    save(ROOT/'cogat_pair_relationships.json',{'requested_ids':sorted(ids),'failures':{k:v['error'] for k,v in details.items() if 'error' in v},'links':links})
    print(json.dumps({'requested_count':len(ids),'links':links},ensure_ascii=False,indent=2))

if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1]=='fetch':get_public()
    elif sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='expand':expand()
    elif sys.argv[1]=='detail':detail()
    elif sys.argv[1]=='families':family_inventory()
    elif sys.argv[1]=='cogat-links':cogat_pair_relations()
    else:raise SystemExit('Use fetch or audit')
