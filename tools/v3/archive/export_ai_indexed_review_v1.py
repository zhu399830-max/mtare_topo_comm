"""Prepare exact AI annotation inputs; no labels, teacher reads or training."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, platform, resource, shutil, sys, time, traceback

NAME='gse_ai_indexed_review_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v,mode='x'):
    with p.open(mode) as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)

def freeze():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
    card['scope']['activity']='Indexed observation input preparation for AI region annotation; zero annotations in this execution'
    card['scope']['output_contract']='gse_surface_review_bundle_v2; all valid five-frame returns with original ray IDs; no teacher fields'
    card['approval']['scope_sha256']=hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest()
    card['approval']['scope']='Same approved three-case AI annotation: indexed input preparation only, no labels or training'
    write(ROOT/CARD,card)
    files=['tools/v3/export_ai_indexed_review.py','tools/v3/_bootstrap.py','tools/v3/review/gse_surface_review.js','tools/v3/review/gse_surface_review.html']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'src/mtare_topo').rglob('*.py')]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='ai_annotation',
        data_card=CARD,user_authorization=card['approval'],
        command=['env','CUDA_VISIBLE_DEVICES=','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',PYTHON,'tools/v3/export_ai_indexed_review.py','--execute'],
        question='Can the three original observations be presented with exact raw ray identities for AI region annotation?',
        method='Existing causal registration, all valid returns, indexed v2 review bundle; input preparation only',
        baseline='Original saved student arrays and source frame/ray identities, not a model comparison',
        fallback='Fail and seal on drift, count mismatch or resource excess; never truncate observations',
        acceptance_criteria=['3 observations/15 frames; valid counts 51437/57147/54310','All retained raw identities unique and JSON roundtrip exact','No teacher payload, labels or training'],
        expected_evidence=['Indexed bundles, manifest, tool snapshot, environment, raw log, metrics, RUN_STATE, SHA seal'],
        estimated_cost=dict(compute='CPU only',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.05),
        input_sha256={CARD:sha(ROOT/CARD),**{e['student_path']:e['student_sha256'] for e in card['scope']['entries']}},
        source_sha256={p:sha(ROOT/p) for p in files},python_sha256=sha(__import__('pathlib').Path(PYTHON).resolve()))
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    if json.loads((run/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    if sha(__import__('pathlib').Path(sys.executable).resolve())!=spec['python_sha256']:raise ValueError('python drift')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    try:
        import numpy as np
        import torch
        from mtare_topo.data.gse_membership_fit_reader import load_student_window
        from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points,MAXIMUM_RANGE_M
        card=json.loads((ROOT/CARD).read_text())
        write(run/'config/environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,executable=sys.executable))
        for n in ('gse_surface_review.html','gse_surface_review.js'):shutil.copyfile(ROOT/'tools/v3/review'/n,run/n)
        shutil.copyfile(ROOT/'tools/v3/export_ai_indexed_review.py',run/'artifacts/runner_source.py')
        with (run/'logs/export.jsonl').open('x') as log:
            for i,e in enumerate(card['scope']['entries']):
                s=load_student_window(ROOT,dict(e,layout='saved_single_window',decoded_observations=1))
                rv=torch.stack((torch.tensor(s.ranges_m.copy())/MAXIMUM_RANGE_M,torch.tensor(s.valid_mask.copy(),dtype=torch.float32)),dim=1)[None]
                xyz,valid=register_causal_lidar_points(rv,torch.tensor(s.relative_translation_current_sensor_m.copy())[None],torch.tensor(s.relative_yaw_current_sensor_deg.copy())[None])
                xyz=xyz[0].reshape(5,11520,3).numpy();valid=valid[0].reshape(5,11520).numpy()
                slots,rays=np.nonzero(valid)
                if len(slots)!=card['scope']['valid_returns'][i]:raise ValueError('valid count drift')
                if not np.isfinite(xyz[valid]).all():raise ValueError('nonfinite registered point')
                b=dict(schema='gse_surface_review_bundle_v2',observation_id=f'case_{i}',coordinate_frame='current_sensor_m',source_frame_indices=e['frame_rows'],points_xyz_m=xyz[valid].tolist(),point_history_slots=slots.tolist(),point_ray_indices=rays.tolist())
                path=run/f'artifacts/case_{i}.json';write(path,b)
                if json.loads(path.read_text())!=b:raise ValueError('JSON roundtrip mismatch')
                row=dict(case=i,task=e['task'],bundle=str(path.relative_to(run)),sha256=sha(path),points=len(slots),source_student_sha256=e['student_sha256'])
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise MemoryError('RAM cap')
                if time.monotonic()-start>180 or sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>100*1024**2:raise RuntimeError('time/output cap')
    except Exception:error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',completed=len(rows),observations=rows,new_labels=0,training_steps=0,training_qualified=False,meaning='Annotation inputs exported, not annotations or method validation',elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,error=error)
    write(run/'metrics/summary.json',summary);write(run/'artifacts/review_manifest.json',dict(observations=rows,prior_reference_exposure=True,blind=False))
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
