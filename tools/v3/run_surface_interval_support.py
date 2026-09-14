"""Reuse sealed16 residual arc hypotheses; no ray computation or point labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON

SLUG='gse_surface_interval_support_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
OLD='results/gate3_semantics/gate3_20260911_gse_membership_surface_residual_v1_seed0'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    old=json.loads((ROOT/OLD/'config/data_card.json').read_text())['scope']['binding']
    seal={p:h for h,p in (line.split('  ',1) for line in (ROOT/OLD/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    files={OLD+'/artifacts/evidence_sha256.txt':sha(ROOT/OLD/'artifacts/evidence_sha256.txt')}
    entries=[];chunks={}
    for e in old['entries']:
        i=e['observation'];b=e['original_binding'];plan=e['arrays']['sensor_xyz_m'];f=b['frame_rows'][-1]
        prefix=plan['prefix'];key=str(f//plan['header']['chunks'][0])+'.0'
        p=OLD+f'/artifacts/window_{i:02d}.npz';files[p]=seal[p]
        for path in (e['documents']['constructions'],prefix+'/.zarray',prefix+'/'+key):files[path]=old['input_sha256'][path]
        chunks[prefix+'/'+key]=plan['header']['chunks'][0]*3*8
        entries.append(dict(observation=i,task=b['source']['task'],frame=f,header=plan['header'],
            coordinate_chunk=prefix+'/'+key,residual=p,construction=e['documents']['constructions']))
    s=dict(entries=entries,parents=8,unique_frames=80,labels_generated=0,residual_threshold=None,
        input_sha256=files,decoded_unique_coordinate_bytes=sum(chunks.values()),
        limits=dict(wall_seconds=600,host_bytes=2*1024**3,output_bytes=512*1024**2))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['audit'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Same16 saved arc hypotheses and original coordinates/constructions; no labels',
        confirmation_reference='Active goal and current PLAN explicitly allow saved arc to local interval support diagnosis')
    write(ROOT/CARD,dict(schema_version='gse_surface_interval_card_v1',scope=s,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/run_surface_interval_support.py','--execute'],
        question='Which saved surface hypotheses lie strictly within one local source-axis interval?',
        method='Original10m analytic axis intervals, all source hypotheses retained, strict arc containment',
        baseline='Source identity alone does not separate local reentry; no model comparison',fallback='Seal failures; no label changes or budget extension',
        estimated_cost=dict(compute='CPU saved16 arrays and coordinates; zero scans/GPU',host_ram_gb=2,gpu_vram_gb=0,disk_gb=.5,wall_time_hours=1/6),
        acceptance_criteria=['All16 represented without dropped sources','Boundary,missing coverage and source ambiguity distinct','Zero labels and no new surface computations'],
        expected_evidence=['per-row conditional states,interval inventory,summary,logs,source snapshot,seal'],input_sha256=files,source_sha256=sources))
    print(json.dumps(dict(observations=16,decoded_unique_coordinate_bytes=s['decoded_unique_coordinate_bytes'])))


def execute():
    import numpy as np
    import numcodecs
    from mtare_topo.governance_surface_residual import validate_interval_card
    from mtare_topo.teacher.gse_local_axis_intervals import local_axis_intervals
    from mtare_topo.teacher.gse_surface_interval_support import join_surface_interval_support
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_interval_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    resource.setrlimit(resource.RLIMIT_AS,(s['limits']['host_bytes'],s['limits']['host_bytes']))
    def expire(*args):raise TimeoutError('600 second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        write(run/'config/runtime_environment.json',dict(python=sys.version,numpy=np.__version__))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        with (run/'logs/windows.jsonl').open('x') as log:
            for e in s['entries']:
                i=e['observation'];h=e['header']
                if h['filters'] or h['dtype']!='<f8' or h['order']!='C':raise ValueError('coordinate layout drift')
                decoded=numcodecs.get_codec(h['compressor']).decode((ROOT/e['coordinate_chunk']).read_bytes())
                coords=np.frombuffer(decoded,dtype='<f8').reshape(h['chunks']);center=coords[e['frame']%h['chunks'][0]]
                primitives=json.loads((ROOT/e['construction']).read_text())['realized_primitives']
                with np.load(ROOT/e['residual'],allow_pickle=False) as z:records=z['records']
                intervals={int(k):local_axis_intervals(primitives[int(k)]['centerline_xyz_m'],center_m=center)
                           for k in np.unique(records[:,1])}
                result=join_surface_interval_support(records,intervals)
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
                write(run/f'artifacts/intervals_{i:02d}.json',intervals)
                row=dict(observation=i,task=e['task'],source_pairs=len(records),returns=len(result['ray_indices']),
                    state_counts=np.bincount(result['row_state'],minlength=4).tolist(),
                    single_source_interior_returns=len(result['single_source_interior_ray_indices']),
                    multisource_returns=int((result['source_hypotheses_per_ray']>1).sum()),
                    source_interval_counts={str(k):len(v['intervals_m']) for k,v in intervals.items()},labels_generated=0)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,
        state_names=['crossing_unresolved','no_axis_overlap','boundary_or_multiple_intervals','strict_single_interval_hypothesis'],
        labels_generated=0,point_mask_qualified=False,training_steps=0,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
