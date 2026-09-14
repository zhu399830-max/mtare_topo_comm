"""Once-only original surface matching for the fixed new12, never old16."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from ai_junction_pilot import sha,write
NAME='gse_new12_surface_residual_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
BIND='configs/v3/gate3/gse_new12_surface_source_scope_v1.json'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'


def freeze():
    from freeze_new12_surface_scope import compile_scope
    b=json.loads((ROOT/BIND).read_text())
    if b!=compile_scope():raise ValueError('metadata scope drift')
    s=dict(binding=b,roi_radius_m=10,labels_generated=0,training_steps=0,acceptance_distance_m=None,
        mesh_coordinates='original_float32_scene',limits=dict(wall_seconds=10800,host_bytes=4*1024**3,output_bytes=2*1024**3,max_candidates=200000))
    a=dict(status='APPROVED',approved_by='user-standing-new-route-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['audit'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Fixed new12 original return-source evidence extraction only, no labels/training',
        confirmation_reference='User: 那就继续啊按照新的路线推进; 设立一个目标一直干下去; active goal explicitly includes fixed new12 source binding')
    write(ROOT/CARD,dict(schema_version='gse_new12_surface_residual_card_v1',scope=s,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/run_new12_surface_residual.py','--execute'],
        question='What original surface evidence supports all candidate sources of the new12 observed returns?',
        method='Original float32 mesh and ray reconstruction; exhaustive-safe candidate surface residuals with tied arc bounds',
        baseline='Same return reported-world versus actual float32 scene coordinates; not a learned performance comparison',
        fallback='Seal failures/unknown evidence, no widened thresholds or source selection',
        estimated_cost=dict(compute='CPU12 observations/60 frames; zeroGPU, no optimizer',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=3),
        acceptance_criteria=['All12 exact frame identities and valid counts','Every local return and all coded sources preserved','No assignment threshold or label qualification'],
        expected_evidence=['All return/source residual and arc records, raw logs,source snapshot,environment,seal'],
        input_sha256={**b['input_sha256'],BIND:sha(ROOT/BIND),CARD:sha(ROOT/CARD)},source_sha256=sources))


def execute():
    import numpy as np,zarr
    from mtare_topo.governance_surface_residual import validate_new12_card
    from mtare_topo.data.gse_surface_input_export_v1 import _ExactStore
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
    from mtare_topo.teacher.gse_original_surface_loader import load_original_surface,load_original_ray_points
    from mtare_topo.teacher.gse_return_surface_match import SourceSurfaceMatcher
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope'];b=s['binding']
    if not validate_new12_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;rows=[]
    resource.setrlimit(resource.RLIMIT_AS,(s['limits']['host_bytes'],)*2)
    def expire(*_):raise TimeoutError('10800s cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(s['limits']['wall_seconds'])
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        write(run/'config/runtime_environment.json',dict(python=sys.version,numpy=np.__version__,zarr=zarr.__version__))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        with (run/'logs/windows.jsonl').open('x') as log:
            for e in b['entries']:
                i=e['observation'];identity=e['identity'];sensor={}
                student=load_student_window(ROOT,{**identity,'layout':'saved_single_window','decoded_observations':1})
                for field,plan in e['arrays'].items():
                    store=_ExactStore(ROOT,plan['prefix'],b['input_sha256'],{})
                    store.allowed={'.zarray',*plan['access']['chunk_keys']}
                    sensor[field]=np.asarray(zarr.Array(store=store,read_only=True).oindex[identity['frame_rows']])
                construction=json.loads((ROOT/e['documents']['constructions']).read_text())
                codebook=json.loads((ROOT/e['documents']['codebooks']).read_text())
                verify_alignment(sensor,vars(student),construction,codebook,identity)
                valid=student.valid_mask.reshape(-1).astype(bool)
                if int(valid.sum())!=b['valid_returns_per_observation'][i]:raise ValueError('valid count drift')
                points=load_original_ray_points(ROOT,b['original_surface_binding'],student.ranges_m,sensor['sensor_xyz_m'],sensor['yaw_deg'])
                world=points['world_points_xyz_m'].reshape(-1,3);scene=points['scene_points_xyz_m'].reshape(-1,3)
                indices=np.flatnonzero(valid&(np.linalg.norm(world-sensor['sensor_xyz_m'][-1],axis=1)<=10))
                codes=sensor['primitive_membership_code'].reshape(-1);sets=codebook['source_sets'];primitives=construction['realized_primitives']
                output=[]
                for k in sorted({k for r in indices for k in sets[int(codes[r])]}):
                    p=primitives[k];primitive={key:p[key] for key in ('primitive_id','centerline_xyz_m','endpoint_half_axes_m','endpoint_shape_exponent')}
                    mesh=load_original_surface(ROOT,b['original_surface_binding'],primitive)
                    matcher=SourceSurfaceMatcher(mesh['scene_vertices_xyz_m'],mesh['triangle_vertex_indices'],max_candidates_per_return=s['limits']['max_candidates'])
                    source_indices=[r for r in indices if k in sets[int(codes[r])]]
                    for r in source_indices:
                        a=matcher.nearest_residual(scene[r]);v=matcher.nearest_residual(world[r])
                        arcs=mesh['triangle_arc_bounds_m'][a['minimum_triangle_indices']]
                        output.append((int(r),k,a['minimum_distance_m'],v['minimum_distance_m'],len(a['minimum_triangle_indices']),float(arcs.min()),float(arcs.max())))
                    print(json.dumps(dict(observation=i,source=k,returns=len(source_indices),elapsed_s=time.monotonic()-start)),flush=True)
                    del matcher,mesh
                values=np.asarray(output,dtype=np.float64).reshape(-1,7)
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',records=values,roi_return_indices=indices,
                    point_quantization_delta_m=np.linalg.norm(world[indices]-scene[indices],axis=1))
                row=dict(observation=i,task=identity['task'],roi_returns=len(indices),source_pairs=len(values),
                    ambiguous_source_returns=sum(len(sets[int(codes[r])])>1 for r in indices),
                    scene_residual_quantiles_m=np.quantile(values[:,2],[0,.5,.9,.99,1]).tolist() if len(values) else [],elapsed_s=time.monotonic()-start)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,windows=rows,completed=len(rows),
        new_labels=0,training_steps=0,point_label_qualified=False,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        record_columns=['ray_index','source_index','scene_residual_m','world_residual_m','exact_min_face_count','min_arc_m','max_arc_m'])
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
