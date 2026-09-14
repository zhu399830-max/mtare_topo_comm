"""Bounded original interior sections versus cached observed surface support."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,time,traceback
from ai_junction_pilot import sha,write
from export_new12_affinity_targets import FEATURE,SOURCE,PYTHON
NAME='gse_new12_interior_support_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'


def freeze():
    binding=json.loads((ROOT/'configs/v3/gate3/gse_new12_surface_source_scope_v1.json').read_text())
    alignment=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new_fit_construction_alignment_v1.json').read_text())
    old=json.loads((ROOT/'configs/v3/gate3/gse_new_fit_construction_alignment_v1.json').read_text())['input_sha256']
    entries=[];pins={}
    for e,p in zip(binding['entries'],alignment['scope']['construction_alignment'],strict=True):
        i=e['observation'];assert i==p['case']
        entries.append(dict(observation=i,identity=e['identity'],construction=e['documents']['constructions'],poses=p['sensor']))
        pins[p['sensor']]=old[p['sensor']];pins[e['documents']['constructions']]=binding['input_sha256'][e['documents']['constructions']]
    original=binding['original_surface_binding'];pins[original['archive_path']]=original['archive_sha256']
    for base in (FEATURE,SOURCE):
        seal=ROOT/base/'artifacts/evidence_sha256.txt';saved={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
        for i in range(12):
            p=f'{base}/artifacts/window_{i:02d}.npz';pins[p]=saved[p]
    s=dict(entries=entries,original_surface_binding=original,parents=12,frames=60,roi_returns=446184,
        spacing=binding['spacing'],selection_bias=binding['selection_bias'],feature_run=FEATURE,residual_run=SOURCE,
        selection='middle unique observed arc slab per exact connected interval component; singleton only',
        evidence='source-specific .25m contour cells from actual returns inside10m; finite crossings are cached-ray geometry proxies',
        labels_generated=0,training_steps=0,range_error_bound_m=0.,range_semantics='directions/ranges reconstructed from cached endpoints, not an additional sensor accuracy claim',
        limits=dict(wall_seconds=900,host_bytes=4*1024**3,output_bytes=128*1024**2,candidates_per_observation=256))
    a=dict(status='APPROVED',approved_by='user-implicit-structure-plan-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['audit'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='same12 minimal interior observation-support supplement, no new labels',
        confirmation_reference='User approved implicit structure plan: supplement only missing causal evidence; active goal and PLAN interior-reference contract')
    write(ROOT/CARD,dict(schema_version='gse_new12_interior_support_card_v1',scope=s,approval=a));pins[CARD]=sha(ROOT/CARD)
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/check_new12_interior_support.py','--execute'],
        question='Do observed interior source sections have in-domain contour and finite-ray support?',method=s['selection'],
        baseline='Previous boundary references remain historical; no learned performance comparison',fallback='Preserve each unresolved geometry or missing support, no shifting sections, labels or training',
        acceptance_criteria=['12 fixed observations, source hashes unchanged','No source rematching or new scans','Source-specific contour support and crossings separate','No automatic structural labels'],
        expected_evidence=['candidate section geometry, loops, supporting original return IDs, unsupported cells, errors, environment, source snapshot, seal'],
        estimated_cost=dict(compute='CPU12 cached observations and original source meshes',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.125,wall_time_hours=.25),input_sha256=pins,source_sha256=sources))


def execute():
    import numpy as np
    from mtare_topo.governance_new12_sections import validate_interior_card
    from mtare_topo.teacher.gse_interior_section_candidates import nominate_interior_sections,section_frame_from_original_rings
    from mtare_topo.teacher.gse_original_surface_loader import load_original_surface
    from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform as transform
    from mtare_topo.teacher.gse_mesh_sections_v1 import section_ray_witnesses
    from mtare_topo.teacher.gse_local_contour_support_v1 import contour_surface_support
    from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_interior_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    start=time.monotonic();rows=[];error=None;write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*_):raise TimeoutError('900s total cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(900)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
        resource.setrlimit(resource.RLIMIT_AS,(s['limits']['host_bytes'],)*2)
        import zipfile
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        write(run/'config/runtime_environment.json',dict(python=__import__('sys').version,numpy=np.__version__,ray_semantics=s['range_semantics']))
        for e in s['entries']:
            i=e['observation'];construction=json.loads((ROOT/e['construction']).read_text())
            with np.load(ROOT/e['poses'],allow_pickle=False) as p:origin=p['sensor_xyz_m'][-1];yaw=float(p['yaw_deg'][-1])
            with np.load(ROOT/SOURCE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as a:records=a['records'];roi=a['roi_return_indices']
            nomination=nominate_interior_sections(records)
            if len(nomination['candidates'])>256:raise RuntimeError('candidate resource capacity exceeded; no truncation')
            with np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f:
                if not np.array_equal(roi,f['surface_return_indices']):raise ValueError('ROI drift')
                xyz=f['registered_returns_xyz_m'].astype(float);ids=f['patch_ray_input_index'];origins=f['patch_ray_origins_m'];endpoints=f['patch_ray_endpoints_m']
                delta=endpoints-origins;dist=np.linalg.norm(delta,axis=1)
                rays=CausalRaySegments(origins,delta/dist[:,None],dist,np.ones(len(ids),dtype=bool),f['patch_ray_frame_index'],4,0.)
                rays.validate()
            result=[];mesh=None;loaded=None
            for candidate in nomination['candidates']:
                c=dict(candidate,loops=[],geometry_status='UNRESOLVED',training_qualified=False)
                if candidate['reference_arc_m'] is not None:
                    k=candidate['source_index']
                    if loaded!=k:
                        p=construction['realized_primitives'][k]
                        mesh=load_original_surface(ROOT,s['original_surface_binding'],{key:p[key] for key in ('primitive_id','centerline_xyz_m','endpoint_half_axes_m','endpoint_shape_exponent')});loaded=k
                    try:
                        frame=section_frame_from_original_rings(mesh['scene_vertices_xyz_m'],mesh['vertex_arc_m'],candidate['selected_slab_m'])
                        center=transform(frame['center_m'],origin,yaw);normal=transform(frame['normal'],np.zeros(3),yaw);normal/=np.linalg.norm(normal)
                        c.update(center_m=center.tolist(),normal=normal.tolist())
                        if np.linalg.norm(center)>10:c['geometry_status']='CENTER_OUTSIDE_INPUT'
                        else:
                            vertices=transform(mesh['scene_vertices_xyz_m'].astype(float),origin,yaw)
                            section,witnesses=section_ray_witnesses(vertices,mesh['triangle_vertex_indices'],center_m=center,normal=normal,rays=rays)
                            known=np.zeros(57600,dtype=bool);known[candidate['source_return_indices']]=True
                            for loop,witness in zip(section.loops_m,witnesses):
                                inside=bool(np.all(np.linalg.norm(loop,axis=1)<=10))
                                evidence=contour_surface_support(loop,xyz,known) if inside else None
                                c['loops'].append(dict(vertices_m=loop.tolist(),inside_input=inside,crossing_original_ray_indices=ids[list(witness)].tolist(),contour=evidence,
                                    joint_geometric_support=bool(inside and witness and evidence['contour_fully_surface_supported'])))
                            c['geometry_status']='SECTION_EVALUATED'
                    except ValueError as exc:
                        message=str(exc)
                        expected=('section slab must be consecutive','section intersects source','open or nonmanifold section','duplicate section segment','section revisits','degenerate section cycle')
                        if not message.startswith(expected):raise
                        c['geometry_error']=message
                result.append(c)
                print(json.dumps(dict(observation=i,source=c['source_index'],component=c['component_index'],status=c['geometry_status'],loops=len(c['loops']))),flush=True)
            write(run/f'artifacts/case_{i:02d}.json',dict(identity=e['identity'],candidates=result,ambiguous_return_indices=nomination['ambiguous_return_indices'],labels_generated=0))
            rows.append(dict(case=i,candidates=len(result),evaluated=sum(c['geometry_status']=='SECTION_EVALUATED' for c in result),joint_supported=sum(l['joint_geometric_support'] for c in result for l in c['loops'])))
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise RuntimeError('output cap')
            del mesh;mesh=None
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,new_labels=0,training_steps=0,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
