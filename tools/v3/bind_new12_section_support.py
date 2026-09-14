"""Cached reference witnesses to actual observed patches; never new labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,gzip,hashlib,json,time,traceback,resource
from ai_junction_pilot import sha,write
from export_new12_affinity_targets import FEATURE,PYTHON
NAME='gse_new12_section_support_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
REFCARD='configs/v3/gate3/data_cards/gse_new_fit_reference_review_v1.json'


def freeze():
    ref=json.loads((ROOT/REFCARD).read_text());s=dict(entries=ref['scope']['entries'],references=ref['scope']['reference_reveal'],
        parents=12,frames=60,feature_run=FEATURE,spacing='same frozen original five frame rows; no new sampling',
        split='C01-C06 fit only; no validation claim',training_steps=0,new_labels=0,rerender=False,
        method='original crossing and surface ray IDs separately joined to existing point_patch_index; preserve outsideROI and overlaps',
        limits=dict(wall_seconds=120,host_bytes=4*1024**3,output_bytes=64*1024**2))
    a=dict(status='APPROVED',approved_by='user-implicit-structure-plan-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['audit'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='same12 cached partial section references to observed surface support only',
        confirmation_reference='User PLEASE IMPLEMENT implicit structure plan and active goal; same12 minimal cached correspondence specified in PLAN')
    write(ROOT/CARD,dict(schema_version='gse_new12_section_support_card_v1',scope=s,approval=a))
    pins={CARD:sha(ROOT/CARD),REFCARD:sha(ROOT/REFCARD)}
    for r in s['references']:
        if r['path'] is not None:pins[r['path']]=r['sha256']
    seal=ROOT/FEATURE/'artifacts/evidence_sha256.txt'
    old={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    for i in range(12):
        p=f'{FEATURE}/artifacts/window_{i:02d}.npz';pins[p]=old[p]
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
    sources={p:sha(ROOT/p) for p in ['tools/v3/bind_new12_section_support.py','tools/v3/export_new12_affinity_targets.py','tools/v3/ai_junction_pilot.py','src/mtare_topo/governance_new12_sections.py']}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/bind_new12_section_support.py','--execute'],
        question='Which observed patches support each saved partial section reference, and where is support missing or shared?',method=s['method'],
        baseline='Original per-reference witnesses, not a trained comparison',fallback='Missing references and out-of-domain returns remain explicit; no new teacher/labels',
        acceptance_criteria=['12 original identities preserved','Six missing references not filled','Crossing rays not treated as surface ownership','No labels or training'],
        expected_evidence=['all reference-to-patch indices/counts with geometry and overlap, missing cases, log, hashes'],
        estimated_cost=dict(compute='CPU cached6 reference payloads and12 feature files',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.063,wall_time_hours=.034),input_sha256=pins,source_sha256=sources))


def execute():
    import numpy as np
    from mtare_topo.governance_new12_sections import validate_card
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        write(run/'config/runtime_environment.json',dict(python=__import__('sys').version,numpy=np.__version__))
        write(run/'artifacts/source_snapshot.json',{p:(ROOT/p).read_text() for p in spec['source_sha256']})
        for e,r in zip(s['entries'],s['references'],strict=True):
            i=e['case'];row=dict(case=i,task=e['task'],frames=e['frame_rows'],references=[])
            if r['path'] is None:row['status']='MISSING_ORIGINAL_REFERENCE'
            else:
                raw=(ROOT/r['path']).read_bytes();full=json.loads(gzip.decompress(raw) if r['path'].endswith('.gz') else raw)
                if full['raw_interfaces']['source']['task']!=e['task'] or full['raw_interfaces']['source']['frame_rows']!=e['frame_rows']:raise ValueError('identity mismatch')
                target=full['produced_targets'];record=target['record'];prov=target['teacher_provenance'];owners=[]
                if record['source_frame_indices']!=e['frame_rows'] or len(record['openings'])!=len(prov['openings']):raise ValueError('reference alignment')
                with np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f:
                    mapping=f['patch_point_patch_index'];m=len(f['patch_centers_m'])
                    if mapping.shape!=(57600,) or not np.array_equal(np.flatnonzero(mapping>=0),f['surface_return_indices']):raise ValueError('ROI mapping mismatch')
                    for j,(opening,p) in enumerate(zip(record['openings'],prov['openings'],strict=True)):
                        item=dict(index=j,original_geometry=opening,source_teacher_only=p['primitive_id_teacher_only'],reference_arc_m=p['reference_arc_m'],training_qualified=False)
                        for kind in ('crossing','surface'):
                            ids=p[kind+'_ray_indices']
                            if any(type(v) is not int or not 0<=v<57600 for v in ids):raise ValueError('invalid ray indices')
                            ids=np.array(sorted(set(ids)),dtype=np.int64);local=ids[mapping[ids]>=0]
                            patch_ids,counts=np.unique(mapping[local],return_counts=True)
                            item[kind]=dict(original_ray_indices=ids.tolist(),local_surface_ray_indices=local.tolist(),outside_patch_domain=int(len(ids)-len(local)),
                                patch_ids=patch_ids.tolist(),patch_return_counts=counts.tolist(),per_frame=np.bincount(local//11520,minlength=5).tolist(),
                                patch_centers_m=f['patch_centers_m'][patch_ids].tolist(),patch_normals=f['patch_normals'][patch_ids].tolist(),normal_valid=f['patch_normal_valid'][patch_ids].tolist())
                        owners.append(set(item['surface']['patch_ids']));row['references'].append(item)
                    row['shared_surface_patches']=[dict(first=a,second=b,patch_ids=sorted(owners[a]&owners[b])) for a in range(len(owners)) for b in range(a+1,len(owners))]
                row['status']='BOUND_NOT_LABEL_QUALIFIED'
            rows.append(row);write(run/f'artifacts/case_{i:02d}.json',row)
            print(json.dumps(dict(case=i,status=row['status'],references=len(row['references']))),flush=True)
            if time.monotonic()-start>120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise RuntimeError('resource cap')
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise RuntimeError('disk cap')
    except BaseException:error=traceback.format_exc()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),references=sum(len(r['references']) for r in rows),
        missing_cases=[r['case'] for r in rows if r['status']=='MISSING_ORIGINAL_REFERENCE'],new_labels=0,training_steps=0,elapsed_s=time.monotonic()-start)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
