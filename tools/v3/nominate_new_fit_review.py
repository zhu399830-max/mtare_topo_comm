"""Fixed metadata-only nomination after explicit user population-change approval."""
from _bootstrap import PROJECT_ROOT as ROOT
import collections,hashlib,json
POOL='results/gate3_semantics/gate3_20260910_gse_local_pair_pilot_inputs_v1_seed20260906'
OUT='configs/v3/gate3/gse_new_fit_review_nomination_v1.json'
SEED=20260906
def sha(b):return hashlib.sha256(b).hexdigest()
def rank(value):return sha((str(SEED)+'|'+value).encode())

def main():
    manifest=POOL+'/artifacts/manifest.json';raw=(ROOT/manifest).read_bytes()
    pins={p:h for h,p in (l.split('  ',1) for l in (ROOT/POOL/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    if sha(raw)!=pins[manifest]:raise ValueError('manifest drift')
    rows=json.loads(raw)
    old16path='configs/v3/gate3/gse_common_partial_structure_task_v1.json'
    old3path='configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json'
    old16=json.loads((ROOT/old16path).read_text());old3=json.loads((ROOT/old3path).read_text())
    excluded={e['source']['parent_id'] for e in old16['entries']}|{e['source']['parent'] for e in old3['entries']}
    grouped=collections.defaultdict(list)
    for r in rows:
        s=r['source'];parent=s['parent']
        if s['split']!='fit' or parent in excluded:continue
        if parent.rsplit('_',1)[-1] not in {f'C{i:02}' for i in range(1,7)}:raise ValueError('non-training cohort')
        if len(s['frame_rows'])!=5 or s['frame_rows']!=list(range(s['frame_rows'][0],s['frame_rows'][0]+5)):raise ValueError('nonconsecutive history')
        grouped[parent].append(r)
    if len(grouped)<12:raise ValueError(f'Need12 independent parents, available{len(grouped)}; no duplicate fill')
    selected=[]
    for parent in sorted(grouped,key=lambda p:(rank('parent|'+p),p))[:12]:
        r=min(grouped[parent],key=lambda r:(rank('observation|'+r['source']['task']+'|'+str(r['source']['source_global_sequence_index'])),r['student_path']))
        s=r['source'];p=POOL+'/'+r['student_path']
        if pins[p]!=r['student_sha256']:raise ValueError('saved student binding drift')
        selected.append(dict(case=len(selected),task=s['task'],parent=parent,split='fit',frame_rows=s['frame_rows'],traversal_index=s['traversal_index'],sequence_row=s['sequence_row'],source_global_sequence_index=s['source_global_sequence_index'],student_path=p,student_sha256=pins[p],parent_rank=rank('parent|'+parent)))
    scope=dict(entries=selected,observations=12,independent_parents=12,frames=60,raw_ray_slots=691200,
        effective_valid_returns='NOT_READ_YET',selection_seed=SEED,excluded_parents=sorted(excluded),eligible_remaining_parents=len(grouped),
        original_fit_observations=sum(r['source']['split']=='fit' for r in rows),
        source_files_sha256={manifest:sha(raw),old16path:sha((ROOT/old16path).read_bytes()),old3path:sha((ROOT/old3path).read_bytes())},
        selection_rule='Stable SHA256 parent rank then observation rank; one existing five-frame observation per parent; no reference quality, witness counts or model scores used',
        spacing='Five consecutive original frame rows. Metre/time spacing not encoded in this manifest and not asserted; independent units are parents, not adjacent frames.',
        bias='Existing cached pool was previously nominated with construction-based criteria; new selection is model-score-independent, not an unbiased world sample or strict unseen test.',
        prior_reference_exposure='Pool and parent identities known; no new point payload or reference payload read in nomination',
        no_training=True,no_C08_C10_reads=True,unknown_is_background=False)
    record=dict(schema='gse_new_fit_review_nomination_v1',status='EXACT_METADATA_NOMINATION_NOT_LABEL_QUALIFICATION',scope=scope,
        authorization=dict(status='APPROVED_SCOPE_CHANGE',confirmation_reference='User explicitly replied 包含啊 to keeping old three cases unresolved and selecting a fixed new batch from original training worlds without model-score selection, five-frame or metric changes.',scope_sha256=sha(json.dumps(scope,sort_keys=True).encode())),
        payload_reads=0,new_labels=0,training_steps=0)
    with (ROOT/OUT).open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(output=OUT,remaining_parents=len(grouped),selected=[(e['parent'],e['source_global_sequence_index']) for e in selected],observations=12,frames=60,raw_ray_slots=691200,payload_reads=0),ensure_ascii=False))

if __name__=='__main__':main()
