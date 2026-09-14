"""One bounded development-reference export, reusing the training algorithms.

Parallelism is across complete independent observations only. Each observation
keeps all returns/sources; neither labels nor model scores choose its workload.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, os, resource, signal, subprocess, sys, time, traceback, zipfile
from ai_junction_pilot import sha, write
from export_conditional_development_features import INPUT, RUN as FEATURE, MANIFEST, PYTHON

NAME='gse_conditional_development_geometry_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
BINDING='configs/v3/gate3/gse_original_surface_source_binding_v1.json'


def freeze():
    manifest=json.loads((ROOT/MANIFEST).read_text())
    inputs=json.loads((ROOT/INPUT/'artifacts/manifest.json').read_text())
    lookup={(r['source']['task'],r['source']['source_global_sequence_index']):r for r in inputs}
    original=json.loads((ROOT/BINDING).read_text())
    # The same exact archive binding used by the fitted conditional references.
    old=json.loads((ROOT/'configs/v3/gate3/gse_new12_surface_source_scope_v1.json').read_text())['original_surface_binding']
    assert original==old
    pins={MANIFEST:sha(ROOT/MANIFEST),BINDING:sha(ROOT/BINDING),original['archive_path']:original['archive_sha256']}
    sealed={}
    for base in (INPUT,FEATURE):
        seal=ROOT/base/'artifacts/evidence_sha256.txt'
        pins[str(seal.relative_to(ROOT))]=sha(seal)
        sealed.update({p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())})
    entries=[]
    for e in manifest['entries']:
        i=e['case'];r=lookup[(e['task'],e['source_sequence_id'])]
        paths=dict(student=INPUT+'/'+r['student_path'],evidence=INPUT+'/'+r['source_evidence_path'],
                   construction=INPUT+'/artifacts/source_evidence/'+e['task']+'_constructions.json',
                   codebook=INPUT+'/artifacts/source_evidence/'+e['task']+'_codebooks.json',
                   feature=FEATURE+f'/artifacts/window_{i:02d}.npz')
        for path in paths.values():
            assert sha(ROOT/path)==sealed[path];pins[path]=sealed[path]
        entries.append(dict(identity=e,paths=paths))
    s=dict(entries=entries,parents=5,observations=240,frames=1200,patches=202340,roi_returns=8498948,
        original_surface_binding=original,target_schema='construction_conditioned_geometry_targets_v1',
        observability_certified=False,connectivity_certified=False,training_steps=0,
        selection='all frozen240; same singleton arc components and middle original slab; no contour qualification or score filtering',
        split_audit=manifest['split_audit'],
        limits=dict(workers=4,worker_address_bytes=4*1024**3,total_rss_bytes=32*1024**3,
                    wall_seconds=21600,case_seconds=900,output_bytes=4*1024**3,max_candidates=200000,reference_capacity=256))
    a=dict(status='APPROVED',approved_by='user-standing-conditional-development-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['data_export'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Same conditional-reference definition for fixed240development observations; no training, graph or test access',
        confirmation_reference='User confirmed construction-conditioned supervision; standing goal and PLAN authorize fixed-parent development evaluation')
    card=dict(schema_version='gse_conditional_development_geometry_card_v1',scope=s,approval=a)
    from mtare_topo.governance_conditional_development import validate_geometry_card
    assert validate_geometry_card(card).passed
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_development_geometry.py','--execute'],
        question='What same-definition conditional geometric targets and unknown population exist on the fixed240head-held-out development observations?',
        method='Original float32 source mesh nearest residual with all source memberships; unchanged singleton interval nomination/ring reference/conditional target compiler; four independent CPUworkers',
        baseline='Frozen training reference algorithm, axis1/height0 reference later; no model ranking here',
        fallback='Seal partial failure and stop workers; no resampling, dropped candidates, tolerance changes, retry or training',
        acceptance_criteria=['240exact identities and202340patches/8498948ROIreturns','All source and ambiguous records preserved','Original reference and unknown semantics unchanged','Zero model/optimizer/contour qualification/hidden connectivity'],
        expected_evidence=['Per-case all source residuals, nominated references, targets/masks, counts, worker logs, exact sources and environment, RUN_STATE and SHA seal'],
        estimated_cost=dict(compute='CPU4workers; same matcher as12fit, no GPU; estimated from810s/12 serial fit observations',host_ram_gb=32,gpu_vram_gb=0,disk_gb=4,wall_time_hours=6),
        input_sha256=pins,source_sha256=sources))
    print(json.dumps(dict(spec=SPEC,cases=240,workers=4,training_steps=0)))


def worker(case, *, spec_path=SPEC, card_path=CARD, run_path=RUN):
    import numpy as np
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
    from mtare_topo.teacher.gse_original_surface_loader import load_original_surface,load_original_ray_points
    from mtare_topo.teacher.gse_return_surface_match import SourceSurfaceMatcher
    from mtare_topo.teacher.gse_interior_section_candidates import nominate_interior_sections,section_frame_from_original_rings
    from mtare_topo.teacher.gse_conditional_geometry_targets import conditional_geometry_targets
    from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform as transform
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches,build_patch_neighbors
    spec=json.loads((ROOT/spec_path).read_text());s=json.loads((ROOT/card_path).read_text())['scope']
    e=s['entries'][case];paths=e['paths'];identity=e['identity'];limit=s['limits'];out=ROOT/run_path/f'artifacts/case_{case:03d}'
    assert identity['case']==case and json.loads((ROOT/run_path/'RUN_STATE.json').read_text())['state']=='RUNNING'
    out.mkdir();start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(limit['worker_address_bytes'],)*2)
    for p in paths.values():
        if sha(ROOT/p)!=spec['input_sha256'][p]:raise ValueError('input drift '+p)
    student_binding=e.get('student_binding',dict(student_path=paths['student'],student_sha256=spec['input_sha256'][paths['student']],layout='saved_single_window',decoded_observations=1))
    if student_binding['student_path']!=paths['student'] or student_binding['student_sha256']!=spec['input_sha256'][paths['student']]:raise ValueError('student binding drift')
    student=load_student_window(ROOT,student_binding)
    with np.load(ROOT/paths['evidence'],allow_pickle=False) as z:sensor={k:z[k] for k in z.files}
    construction=json.loads((ROOT/paths['construction']).read_text());codebook=json.loads((ROOT/paths['codebook']).read_text())
    verify_alignment(sensor,vars(student),construction,codebook,identity)
    points=load_original_ray_points(ROOT,s['original_surface_binding'],student.ranges_m,sensor['sensor_xyz_m'],sensor['yaw_deg'])
    world=points['world_points_xyz_m'].reshape(-1,3);scene=points['scene_points_xyz_m'].reshape(-1,3)
    origin=sensor['sensor_xyz_m'][-1];yaw=float(sensor['yaw_deg'][-1])
    roi_policy=s.get('roi_policy','original_world_distance_v1')
    if roi_policy=='original_world_distance_v1':
        roi=np.flatnonzero(student.valid_mask.reshape(-1).astype(bool)&(np.linalg.norm(world-origin,axis=1)<=10))
    elif roi_policy=='validated_frozen_student_roi_v1':
        from mtare_topo.data.gse_frozen_observation_roi import validated_frozen_roi
        with np.load(ROOT/paths['feature'],allow_pickle=False) as f:
            roi=validated_frozen_roi(f['registered_returns_xyz_m'],student.valid_mask.reshape(-1),f['surface_return_indices'])
    else:raise ValueError('unknown ROI authority policy')
    codes=sensor['primitive_membership_code'].reshape(-1);sets=codebook['source_sets'];primitives=construction['realized_primitives']
    records=[]
    for k in sorted({k for r in roi for k in sets[int(codes[r])]}):
        p=primitives[k];primitive={key:p[key] for key in ('primitive_id','centerline_xyz_m','endpoint_half_axes_m','endpoint_shape_exponent')}
        mesh=load_original_surface(ROOT,s['original_surface_binding'],primitive)
        matcher=SourceSurfaceMatcher(mesh['scene_vertices_xyz_m'],mesh['triangle_vertex_indices'],max_candidates_per_return=limit['max_candidates'])
        ids=[r for r in roi if k in sets[int(codes[r])]]
        for r in ids:
            a=matcher.nearest_residual(scene[r]);v=matcher.nearest_residual(world[r])
            arcs=mesh['triangle_arc_bounds_m'][a['minimum_triangle_indices']]
            records.append((int(r),k,a['minimum_distance_m'],v['minimum_distance_m'],len(a['minimum_triangle_indices']),float(arcs.min()),float(arcs.max())))
        print(json.dumps(dict(case=case,source=k,returns=len(ids),elapsed_s=time.monotonic()-start)),flush=True)
        del matcher,mesh
    records=np.asarray(records,dtype=np.float64).reshape(-1,7)
    np.savez_compressed(out/'source_records.npz',records=records,roi_return_indices=roi)
    nomination=nominate_interior_sections(records);candidates=nomination['candidates']
    if len(candidates)>limit['reference_capacity']:raise RuntimeError('reference capacity, no truncation')
    loaded=None;mesh=None
    for c in candidates:
        if c['reference_arc_m'] is None:continue
        k=c['source_index']
        if loaded!=k:
            p=primitives[k];mesh=load_original_surface(ROOT,s['original_surface_binding'],{key:p[key] for key in ('primitive_id','centerline_xyz_m','endpoint_half_axes_m','endpoint_shape_exponent')});loaded=k
        try:
            frame=section_frame_from_original_rings(mesh['scene_vertices_xyz_m'],mesh['vertex_arc_m'],c['selected_slab_m'])
        except ValueError as exc:
            if not str(exc).startswith('section slab must be consecutive'):raise
            c['geometry_error']=str(exc);continue
        center=transform(frame['center_m'],origin,yaw);normal=transform(frame['normal'],np.zeros(3),yaw);normal/=np.linalg.norm(normal)
        c.update(center_m=center.tolist(),normal=normal.tolist(),geometry_status='REFERENCE_FINITE' if np.linalg.norm(center)<=10 else 'CENTER_OUTSIDE_INPUT')
    write(out/'references.json',dict(identity=identity,**nomination,observability_certified=False,semantic_channel_identity=False))
    with np.load(ROOT/paths['feature'],allow_pickle=False) as f:
        if not np.array_equal(roi,f['surface_return_indices']):raise ValueError('ROI differs from frozen feature input')
        patches=SurfacePatches(**{k:f['patch_'+k] for k in SurfacePatches.__dataclass_fields__ if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
        neighbors=build_patch_neighbors(patches).neighbor_index
        targets=conditional_geometry_targets(candidates,patches.point_patch_index,roi,neighbors)
    targets['original_roi_indices']=roi;np.savez_compressed(out/'targets.npz',**targets)
    known=targets['axis_known'];cross=known&~targets['same_reference_component'];components=targets['patch_reference_component']
    pairs={tuple(sorted((int(components[p]),int(components[neighbors[p,k]])))) for p,k in zip(*np.nonzero(cross))}
    row=dict(case=case,parent=identity['parent_id'],task=identity['task'],patches=len(neighbors),roi_returns=len(roi),source_pairs=len(records),
        candidates=len(candidates),known_directed=int(known.sum()),cross_component_directed=int(cross.sum()),
        distinct_cross_component_pairs=len(pairs),unknown_directed=int(((neighbors>=0)&~known).sum()),
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'summary.json',row);print(json.dumps(row),flush=True)


def execute(*, run_path=RUN, spec_path=SPEC, card_path=CARD, validator=None, worker_script=None):
    import psutil
    from mtare_topo.governance_conditional_development import validate_geometry_card
    validate_geometry_card=validator or validate_geometry_card
    run=ROOT/run_path;spec=json.loads((ROOT/spec_path).read_text());card=json.loads((ROOT/card_path).read_text());s=card['scope'];limits=s['limits']
    assert validate_geometry_card(card).passed
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('bound drift '+p)
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;rows=[];active={};next_case=0;peak=0
    try:
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        write(run/'config/runtime_environment.json',dict(python=sys.version,psutil=psutil.__version__,workers=limits['workers'],cpu_only=True))
        with (run/'logs/progress.jsonl').open('x') as progress:
            if s.get('reuse_cases'):
                from mtare_topo.data.gse_reference_reuse import copy_reference_prefix
                rows=copy_reference_prefix(ROOT,run,s['reuse_cases'],spec['input_sha256'])
                next_case=len(rows)
                for row in rows:progress.write(json.dumps(row)+'\n')
                progress.flush()
            while next_case<len(s['entries']) or active:
                while next_case<len(s['entries']) and len(active)<limits['workers']:
                    log=(run/f'logs/case_{next_case:03d}.log').open('x')
                    process=subprocess.Popen([PYTHON,worker_script or __file__,'--worker',str(next_case)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[next_case]=(process,log,time.monotonic());next_case+=1
                for i,(p,log,t) in list(active.items()):
                    if time.monotonic()-t>limits['case_seconds']:raise TimeoutError(f'case{i}time cap')
                    status=p.poll()
                    if status is None:continue
                    log.close();del active[i]
                    if status:raise RuntimeError(f'case{i}exit{status}; see saved worker log; no retry')
                    row=json.loads((run/f'artifacts/case_{i:03d}/summary.json').read_text());rows.append(row)
                    progress.write(json.dumps(row)+'\n');progress.flush();print(json.dumps(row),flush=True)
                processes=[psutil.Process(),*psutil.Process().children(recursive=True)];rss=0
                for p in processes:
                    try:rss+=p.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
                if rss>limits['total_rss_bytes']:raise MemoryError('aggregate parent/worker/child RSS cap')
                if time.monotonic()-start>limits['wall_seconds']:raise TimeoutError('total wall cap')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>limits['output_bytes']:raise RuntimeError('output cap')
                if active:time.sleep(1)
        if sorted(r['case'] for r in rows)!=list(range(len(s['entries']))):raise ValueError('incomplete or duplicate case population')
        if sum(r['patches'] for r in rows)!=s['patches'] or sum(r['roi_returns'] for r in rows)!=s['roi_returns']:raise ValueError('population mismatch')
    except BaseException:error=traceback.format_exc()
    finally:
        for p,log,_ in active.values():
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p,log,_ in active.values():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=sorted(rows,key=lambda r:r['case']),
        training_steps=0,observability_certified=False,elapsed_s=time.monotonic()-start,peak_process_tree_rss_bytes=peak)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as stream:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':stream.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='windows'}));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');g.add_argument('--worker',type=int);a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:worker(a.worker)
