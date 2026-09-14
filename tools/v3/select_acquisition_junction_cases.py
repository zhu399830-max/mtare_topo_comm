"""Nominate three independent fit parents from sealed completed metadata only."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
R='results/gate3_semantics/gate3_20260910_gse_local_pair_support_pilot_v1_seed20260906'

def main():
    root=ROOT/R;log=root/'logs/observations.jsonl'
    seal={p:h for h,p in (x.split('  ',1) for x in (root/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    key=str(log.relative_to(ROOT))
    if key not in seal:key='logs/observations.jsonl'
    raw=log.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=seal[key]:raise ValueError('log drift')
    old=json.loads((ROOT/'configs/v3/gate3/gse_common_partial_structure_task_v1.json').read_text())
    oldkeys={(e['source']['task'],tuple(e['source']['frame_rows'])) for e in old['entries']}
    candidates=[]
    for row in map(json.loads,raw.decode().splitlines()):
        if row['state']!='COMPLETED' or row.get('split')!='fit' or not row['junction_positions_m']:continue
        source=row['source'];identity=(source['task'],tuple(source['frame_rows']))
        if identity in oldkeys:continue
        rank=hashlib.sha256(('20260906:'+json.dumps(identity,separators=(',',':'))).encode()).hexdigest()
        candidates.append((rank,row))
    selected=[];parents=set()
    for rank,row in sorted(candidates):
        s=row['source']
        if s['parent_id'] in parents:continue
        parents.add(s['parent_id'])
        selected.append(dict(source=s,index=row['index'],rank=rank,reference_path=R+'/artifacts/'+row['evidence_file'],
            reference_sha256=row['storage']['compressed_sha256'],supported_junction_count=len(row['junction_positions_m']),
            openings=row['openings'],full_training_gate_eligible=row['full_training_gate_eligible']))
        if len(selected)==3:break
    if len(selected)!=3:raise ValueError('fewer than3 independent eligible fit parents')
    result=dict(schema='gse_acquisition_junction_nomination_v1',status='METADATA_NOMINATION_NOT_LABEL_OR_EXPORT_APPROVAL',
        eligible_observations=len(candidates),selected=selected,parents=3,observations=3,selected_frames=15,
        rule='Existing fit-only observed junction metadata, exclude original16 task/frame identities, stable hash20260906,one observation per parent',
        limitation='Teacher-stratified nomination, not blind selection or independent method performance; no label qualification',
        source_log_sha256=hashlib.sha256(raw).hexdigest(),payload_reads=0,new_labels=0,training_steps=0)
    p=ROOT/'configs/v3/gate3/gse_acquisition_junction_nomination_v1.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
