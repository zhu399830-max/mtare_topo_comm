"""Once-only fixed16 surface residual measurement, not label generation."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON

SLUG='gse_membership_surface_residual_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
BIND='configs/v3/gate3/gse_membership_surface_scope_v1.json'
ORIGINAL='configs/v3/gate3/gse_original_surface_source_binding_v1.json'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    from freeze_membership_surface_scope import compile_scope
    b=json.loads((ROOT/BIND).read_text())
    if b!=compile_scope():raise ValueError('source scope drift')
    original=json.loads((ROOT/ORIGINAL).read_text())
    s=dict(binding=b,original=original,roi_radius_m=10,labels_generated=0,training_steps=0,
        acceptance_distance_m=None,mesh_coordinates='original_float32_scene',
        limits=dict(wall_seconds=10800,host_bytes=4*1024**3,output_bytes=2*1024**3,max_candidates=200000))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['audit'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Same16 source surface residuals only; no point labels or training',
        confirmation_reference='Active goal and current PLAN authorize original surface residual diagnosis with exact bound16 inputs; standing user authorization')
    write(ROOT/CARD,dict(schema_version='gse_surface_residual_card_v1',scope=s,approval=a))
    files={**b['input_sha256'],BIND:sha(ROOT/BIND),ORIGINAL:sha(ROOT/ORIGINAL),
        original['archive_path']:original['archive_sha256']}
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='audit',data_card=CARD,
        user_authorization=a,question='How far are recorded local returns from each historical active-source surface?',
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/run_membership_surface_residual.py','--execute'],
        method='Original float32 scene meshes/rays; exhaustive-safe nearest triangle distance for every active source, no assignment threshold',
        baseline='Original reported world point versus actual float32 scene-ray point; same return and source',
        fallback='Seal partial failure, no widened limits, new labels or retries',
        estimated_cost=dict(compute='CPU same16 windows,80 frames; zeroGPU',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=3),
        acceptance_criteria=['Complete16 with each valid10m-ROI return and all coded sources retained','Report distances and ambiguities without labeling or acceptance threshold','Source and absolute-relative alignment exact'],
        expected_evidence=['per-source per-return residual NPZ,window summaries,raw logs,source snapshot,environment,seal'],input_sha256=files,source_sha256=sources))


def execute():
    import numpy as np
    import zarr
    from mtare_topo.governance_surface_residual import validate_card
    from mtare_topo.data.gse_surface_input_export_v1 import _ExactStore
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
    from mtare_topo.teacher.gse_original_surface_loader import load_original_surface,load_original_ray_points
    from mtare_topo.teacher.gse_return_surface_match import SourceSurfaceMatcher
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;rows=[]
    resource.setrlimit(resource.RLIMIT_AS,(s['limits']['host_bytes'],s['limits']['host_bytes']))
    def expire(*args):raise TimeoutError('10800 second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(s['limits']['wall_seconds'])
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        write(run/'config/runtime_environment.json',dict(python=sys.version,numpy=np.__version__,zarr=zarr.__version__))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        with (run/'logs/windows.jsonl').open('x') as log:
            for e in s['binding']['entries']:
                i=e['observation'];b=e['original_binding'];student=load_student_window(ROOT,b);sensor={}
                for field,plan in e['arrays'].items():
                    store=_ExactStore(ROOT,plan['prefix'],s['binding']['input_sha256'],{})
                    store.allowed={'.zarray',*plan['access']['chunk_keys']}
                    sensor[field]=np.asarray(zarr.Array(store=store,read_only=True).oindex[b['frame_rows']])
                construction=json.loads((ROOT/e['documents']['constructions']).read_text())
                codebook=json.loads((ROOT/e['documents']['codebooks']).read_text())
                verify_alignment(sensor,vars(student),construction,codebook,b['source'])
                points=load_original_ray_points(ROOT,s['original'],student.ranges_m,sensor['sensor_xyz_m'],sensor['yaw_deg'])
                world=points['world_points_xyz_m'].reshape(-1,3);scene=points['scene_points_xyz_m'].reshape(-1,3)
                valid=student.valid_mask.reshape(-1).astype(bool)
                roi=valid & (np.linalg.norm(world-sensor['sensor_xyz_m'][-1],axis=1)<=s['roi_radius_m'])
                indices=np.flatnonzero(roi);codes=sensor['primitive_membership_code'].reshape(-1)
                source_sets=codebook['source_sets'];primitives=construction['realized_primitives']
                selected_sources=sorted({k for r in indices for k in source_sets[int(codes[r])]})
                output=[]
                for k in selected_sources:
                    p=primitives[k]
                    primitive={key:p[key] for key in ('primitive_id','centerline_xyz_m','endpoint_half_axes_m','endpoint_shape_exponent')}
                    mesh=load_original_surface(ROOT,s['original'],primitive)
                    matcher=SourceSurfaceMatcher(mesh['scene_vertices_xyz_m'],mesh['triangle_vertex_indices'],max_candidates_per_return=s['limits']['max_candidates'])
                    source_indices=[r for r in indices if k in source_sets[int(codes[r])]]
                    for r in source_indices:
                        a=matcher.nearest_residual(scene[r]);v=matcher.nearest_residual(world[r]);faces=a['minimum_triangle_indices']
                        arcs=mesh['triangle_arc_bounds_m'][faces]
                        output.append((int(r),k,a['minimum_distance_m'],v['minimum_distance_m'],len(faces),float(arcs.min()),float(arcs.max())))
                    print(json.dumps(dict(observation=i,source=k,returns=len(source_indices),elapsed_s=time.monotonic()-start)),flush=True)
                    del matcher,mesh
                values=np.asarray(output,dtype=np.float64).reshape(-1,7)
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',records=values,roi_return_indices=indices,
                    point_quantization_delta_m=np.linalg.norm(world[indices]-scene[indices],axis=1))
                row=dict(observation=i,task=b['source']['task'],roi_returns=len(indices),source_pairs=len(values),
                    ambiguous_source_returns=sum(len(source_sets[int(codes[r])])>1 for r in indices),
                    scene_residual_quantiles_m=np.quantile(values[:,2],[0,.5,.9,.99,1]).tolist() if len(values) else [],
                    labels_generated=0,elapsed_s=time.monotonic()-start)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output budget exceeded')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,windows=rows,completed=len(rows),
        labels_generated=0,training_steps=0,point_label_qualified=False,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        record_columns=['ray_index','source_index','scene_residual_m','world_residual_m','exact_min_face_count','min_arc_m','max_arc_m'])
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
