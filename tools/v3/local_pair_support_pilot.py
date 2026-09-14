"""One bounded V8 observation-support diagnostic on the sealed159 inputs.

No candidate nomination from reference pairs, training, retries or teacher changes.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import zipfile
from check_local_pair_window_coverage import sha, write
from export_local_pair_pilot import RUN as INPUT
from multiview_historical_teacher_v1 import environment, old_environment, PYTHON
from request_scoped_teacher_client_v1 import RequestScopedTeacherClient
from v8_multiview_manifest_v1 import compile_manifest
from v8_multiview_population_v1 import summarize_target

SLUG='gse_local_pair_support_pilot_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
WALL=10800
OUTPUT=30*1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    seal=INPUT+'/artifacts/evidence_sha256.txt'
    pins={p:h for h,p in (l.split('  ',1) for l in (ROOT/seal).read_text().splitlines())}
    manifest=INPUT+'/artifacts/manifest.json'
    if sha(ROOT/manifest)!=pins[manifest]:raise ValueError('input manifest drift')
    rows=json.loads((ROOT/manifest).read_text())
    files={seal:sha(ROOT/seal),manifest:pins[manifest]}
    for r in rows:
        for key in ('student_path','source_evidence_path'):
            p=INPUT+'/'+r[key];files[p]=pins[p]
        for kind in ('constructions','codebooks'):
            p=INPUT+'/artifacts/source_evidence/'+r['source']['task']+'_'+kind+'.json';files[p]=pins[p]
    archived=compile_manifest();files.update(archived['source_sha256'])
    files[archived['archive_path']]=archived['archive_sha256']
    counts={split:dict(observations=sum(r['source']['split']==split for r in rows),
        parents=len({r['source']['parent'] for r in rows if r['source']['split']==split}),
        unique_variant_frames=len({(r['source']['task'],f) for r in rows if r['source']['split']==split for f in r['source']['frame_rows']}))
        for split in ('fit','calibration','development')}
    scope=dict(input_manifest=manifest,observations=[r['source'] for r in rows],counts=counts,
        files=files,archive=dict(path=archived['archive_path'],sha256=archived['archive_sha256']),
        geometry_settings=archived['geometry_settings'],qualify_cap_precision=True,
        method='Unmodified archived V8, request-scoped transport; all original local references, not nominated-pair-only targets',
        restrictions=['unknown_not_background','no_training','no_test','no_root_reachability_claim',
            'automatic_partial_labels_not_human_review','one_development_parent_not_generalization'],
        resources=dict(parent_address_space_bytes=1024**3,worker_address_space_bytes=3*1024**3,wall_seconds=WALL,output_bytes=OUTPUT))
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_gates=[3],authorized_operations=['data_export'],scope_sha256=digest(scope),
        scope='Exact159 frozen windows; unchanged V8 support diagnostic only, no training',
        confirmation_reference='User active goal and standing autonomous permission; PLAN explicitly allows bounded support diagnostic on fixed159 inputs.')
    write(ROOT/CARD,dict(schema_version='gse_local_pair_support_card_v1',scope=scope,approval=approval))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    sources[CARD]=sha(ROOT/CARD)
    command=['env','MALLOC_ARENA_MAX=2','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',PYTHON,'tools/v3/local_pair_support_pilot.py','--execute']
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Do fixed co-located junction/terminal windows provide observed positive and negative local memberships under unchanged V8?',
        method=scope['method'],baseline='Archived V8 algorithm; no model comparison or changes to source visibility',
        fallback='Stop and seal on failure/cap; retain unknowns, no automatic retry or training',
        estimated_cost=dict(compute='CPU only159 windows; zero GPU/model/optimizer',host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3),
        acceptance_criteria=['All159 fixed windows, no target-dependent replacement','Source/geometry/unknown semantics unchanged',
            'Full positive/negative/unknown provenance and independent-parent counts; export is not label qualification'],
        expected_evidence=['Full per-window evidence, targets and memberships, counts, environment, logs, seal'],
        input_sha256=files,source_sha256=sources,environment=environment(),teacher_environment=old_environment())
    write(ROOT/SPEC,spec);print(json.dumps(dict(spec=SPEC,counts=counts,executed=False)))


def load_bundle(row):
    import numpy as np
    source=dict(row['source'])
    # IDs below are legacy teacher transport metadata, never model features.
    source['parent_id']=source['parent']
    source['source_sequence_id']=source['source_global_sequence_index']
    with np.load(ROOT/INPUT/row['student_path'],allow_pickle=False) as a:
        student={k:a[k].copy() for k in a.files}
    student.update(frame_rows=np.asarray(source['frame_rows'],dtype=np.int32),
        source_sequence_ids=np.asarray(source['source_sequence_id'],dtype=np.int64))
    with np.load(ROOT/INPUT/row['source_evidence_path'],allow_pickle=False) as a:
        sensor={k:a[k].copy() for k in a.files}
    docs={k:json.loads((ROOT/INPUT/'artifacts/source_evidence'/(source['task']+'_'+k+'.json')).read_text()) for k in ('constructions','codebooks')}
    return dict(source=source,student=student,sensor_teacher_only=sensor,
        construction_teacher_only=docs['constructions'],codebook_teacher_only=docs['codebooks'])


def execute():
    from mtare_topo.data.gse_lossless_evidence_v1 import encode_evidence
    from mtare_topo.governance_branch_place_replay import validate_local_pair_support_card
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text())
    if (json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED'
        or json.loads((run/'config/run_spec.json').read_text())!=spec or not validate_local_pair_support_card(card).passed):raise ValueError('fresh bound run required')
    start=time.monotonic();completed=[];error=None
    def check(*args):
        if time.monotonic()-start>=WALL:raise TimeoutError('total wall cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>OUTPUT:raise OSError('output cap')
    def expire(*args):raise TimeoutError('total wall cap')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    signal.signal(signal.SIGALRM,expire);signal.alarm(WALL)
    try:
        if os.environ.get('MALLOC_ARENA_MAX')!='2':raise ValueError('allocator contract')
        if environment()!=spec['environment'] or old_environment()!=spec['teacher_environment']:raise ValueError('environment drift')
        for p,h in dict(spec['source_sha256'],**spec['input_sha256']).items():
            if sha(ROOT/p)!=h:raise ValueError('input/source drift '+p)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        rows=json.loads((ROOT/card['scope']['input_manifest']).read_text())
        with (run/'logs/teacher.log').open('xb') as log, (run/'logs/observations.jsonl').open('x') as progress:
            with RequestScopedTeacherClient(log,timeout_s=WALL,memory_bytes=3*1024**3) as client:
                resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
                for i,row in enumerate(rows):
                    check();bundle=load_bundle(row)
                    progress.write(json.dumps(dict(state='STARTED',index=i,source=bundle['source']))+'\n');progress.flush()
                    response=client.request(bundle)
                    if response['raw_interfaces']['source']!=bundle['source']:raise ValueError('source response drift')
                    item=summarize_target(bundle['source'],response['produced_targets'])
                    packed,storage=encode_evidence(response);check()
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())+len(packed)>OUTPUT:raise OSError('output cap')
                    name=f'observation_{i:05d}.json.gz'
                    with (run/'artifacts'/name).open('xb') as f:f.write(packed)
                    item.update(evidence_file=name,storage=storage,split=row['source']['split'])
                    completed.append(item)
                    progress.write(json.dumps(dict(state='COMPLETED',index=i,**item))+'\n');progress.flush()
                    print(json.dumps(dict(completed=i+1,total=len(rows),anchors=item['anchors'],openings=item['openings'],elapsed_s=time.monotonic()-start)),flush=True)
                    del bundle,response,packed
        for p,h in spec['input_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during execution')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    counts={}
    for split in ('fit','calibration','development'):
        rows=[r for r in completed if r['split']==split]
        counts[split]=dict(observations=len(rows),parents=len({r['source']['parent'] for r in rows}),
            **{k:sum(r[k] for r in rows) for k in ('anchors','openings','positive_memberships','negative_memberships','unknown_memberships')},
            negative_parents=len({r['source']['parent'] for r in rows if r['negative_memberships']}),
            zero_anchor_windows=sum(not r['anchors'] for r in rows),
            full_training_eligible=sum(r['full_training_gate_eligible'] for r in rows))
    write(run/'artifacts/target_manifest.json',completed)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(completed),counts=counts,
        elapsed_s=time.monotonic()-start,training_steps=0,scientific_gate_pass=False,
        parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
