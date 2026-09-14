"""Saved construction/observation alignment; no geometry casting or target producer."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json, shutil
from ai_junction_pilot import sha, write
from review_new_fit_references import pinned

NAME='gse_new_fit_construction_alignment_v1'
CARD='configs/v3/gate3/data_cards/'+NAME+'.json'
SPEC='configs/v3/gate3/'+NAME+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+NAME+'_seed0'
POOL='results/gate3_semantics/gate3_20260910_gse_local_pair_pilot_inputs_v1_seed20260906'
PREV='results/gate3_semantics/gate3_20260911_gse_new_fit_reference_review_v1_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'

def freeze():
    c=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new_fit_reference_review_v1.json').read_text())
    pins={p:h for h,p in (x.split('  ',1) for x in (ROOT/POOL/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    manifest=POOL+'/artifacts/manifest.json';rows=pinned(manifest,pins[manifest]);inputs={manifest:pins[manifest]}
    bindings=[]
    for e in c['scope']['entries']:
        r=next(r for r in rows if POOL+'/'+r['student_path']==e['student_path'])
        doc=POOL+'/artifacts/source_evidence/'+e['task']+'_constructions.json'
        sensor=POOL+'/'+r['source_evidence_path']
        inputs[doc]=pins[doc];inputs[sensor]=pins[sensor]
        bindings.append(dict(case=e['case'],construction=doc,sensor=sensor,original_pairs=r['source']['reference_pairs']))
    c['scope']['construction_alignment']=bindings
    c['scope']['review_protocol']='Same12 original physical node coordinates; cases9/11 saved interface axes versus five-frame observations; no raycast or new label'
    c['annotation']['input_bundle_contract']='Exact saved construction, sensor poses, indexed observations and frozen reference reveal'
    c['annotation']['output_schema']='Physical coordinates, axis layout, source witness overlap, comparison figures; not label qualification'
    c['approval']['scope']='Same12 saved-source post-reveal alignment only; no teacher generation/training'
    c['approval']['scope_sha256']=hashlib.sha256(json.dumps(c['scope'],sort_keys=True).encode()).hexdigest()
    write(ROOT/CARD,c);inputs[CARD]=sha(ROOT/CARD)
    inputs.update(c['scope']['indexed_bundles'])
    comp=PREV+'/artifacts/comparison.json';inputs[comp]=sha(ROOT/comp)
    for b in c['scope']['reference_reveal']:
        if b['path']: inputs[b['path']]=b['sha256']
    sources=['tools/v3/align_new_fit_construction.py','tools/v3/ai_junction_pilot.py','tools/v3/review_new_fit_references.py',
      'src/mtare_topo/teacher/gse_construction_paths_v3.py','src/mtare_topo/teacher/gse_construction_paths_v2.py',
      'src/mtare_topo/teacher/gse_construction_paths_v1r.py','src/mtare_topo/teacher/gse_directed_interface_binding_v1.py',
      'src/mtare_topo/data/primitive_relation_materialization.py','src/mtare_topo/data/primitive_relation_dataset.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='ai_annotation',data_card=CARD,user_authorization=c['approval'],
      command=['env','MPLCONFIGDIR=/tmp/gse-mpl','OPENBLAS_NUM_THREADS=1',PYTHON,'tools/v3/align_new_fit_construction.py','--execute'],
      question='Do saved physical junctions and branch axes explain the observed geometry and the missing references?',
      method=c['scope']['review_protocol'],baseline='Frozen first-pass and original partial targets; no performance comparison',
      fallback='Preserve all12, mark unavailable evidence and ambiguous structural correspondence; no teacher rerun',
      acceptance_criteria=['Exact12 source hashes and identities','Physical-reference positions distinct from observed labels','No new supervision or hidden-to-visible claims'],
      expected_evidence=['All12 physical coordinates, two case axis overlays, source support intersections, interpretation and seal'],
      estimated_cost=dict(compute='CPU saved-source geometry transforms and two overlays',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.1),input_sha256=inputs,source_sha256={p:sha(ROOT/p) for p in sources}))

def execute():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
    from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
    from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform as transform
    run=ROOT/RUN;s=json.loads((ROOT/SPEC).read_text());c=json.loads((ROOT/CARD).read_text())
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    for p,h in {**s['input_sha256'],**s['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    shutil.copyfile(ROOT/'tools/v3/align_new_fit_construction.py',run/'artifacts/runner_source.py')
    out=[]
    for e,b in zip(c['scope']['entries'],c['scope']['construction_alignment']):
        doc=pinned(b['construction'],s['input_sha256'][b['construction']]);groups=construction_incident_paths(doc)
        with np.load(ROOT/b['sensor'],allow_pickle=False) as a:origin=a['sensor_xyz_m'][-1].copy();yaw=float(a['yaw_deg'][-1])
        local=[]
        for group in groups:
            xyz=transform(np.asarray(group['anchor_world_m']),origin,yaw)
            if np.linalg.norm(xyz)<=10:
                local.append(dict(node=group['node_id_teacher_only'],position_m=xyz.tolist(),degree=len(group['paths']),distance_m=float(np.linalg.norm(xyz))))
        result=dict(case=e['case'],task=e['task'],physical_local_nodes=local,original_pairs=b['original_pairs'],source_geometry_available=True,new_labels=0)
        if e['case'] in (9,11):
            rb=c['scope']['reference_reveal'][e['case']];full=pinned(rb['path'],rb['sha256'])
            target=full['produced_targets'];raw=full['raw_interfaces']
            axes={a['interface_id_teacher_only']:a for a in bind_axes(groups,raw['interfaces_teacher_only'])}
            anchor=target['teacher_provenance']['anchors'][0];node=anchor['node_id_teacher_only']
            group=next(g for g in groups if g['node_id_teacher_only']==node)
            bundle_path=next(p for p in c['scope']['indexed_bundles'] if p.endswith('/case_'+str(e['case'])+'.json'))
            bundle=pinned(bundle_path,s['input_sha256'][bundle_path]);xyz=np.asarray(bundle['points_xyz_m']);slots=np.asarray(bundle['point_history_slots'])
            selected=[axes[k] for k in anchor['interface_ids']]
            directions=np.asarray([a['inward_direction'] for a in selected]);angles=np.degrees(np.arccos(np.clip(directions@directions.T,-1,1)))
            sets=[set(x) for x in anchor['witness_ray_indices']]
            result['junction_details']=dict(node=node,axes=selected,pairwise_angles_deg=angles.tolist(),
                witness_pair_overlap=[[len(a&b) for b in sets] for a in sets],
                witness_role_counts={k:[len(set(x)) for x in anchor[k]] for k in ('entering_witness_ray_indices','interior_witness_ray_indices','surface_entry_witness_ray_indices','surface_departure_witness_ray_indices')},
                actual_source_paths=[dict(endpoint_key=p['endpoint_key'],axis_start_index=p['axis_start_index'],points_current_m=transform(np.asarray(p['points_world_m']),origin,yaw).tolist()) for p in group['paths']])
            fig,axs=plt.subplots(5,2,figsize=(12,20),constrained_layout=True)
            for frame in range(5):
                cloud=xyz[slots==frame][::8]
                for ax,dims in zip(axs[frame],((0,1),(0,2))):
                    ax.scatter(cloud[:,dims[0]],cloud[:,dims[1]],s=.6,c='0.65')
                    for j,p in enumerate(group['paths']):
                        points=transform(np.asarray(p['points_world_m']),origin,yaw)
                        ax.plot(points[:,dims[0]],points[:,dims[1]],label='source axis '+str(j),linewidth=2)
                    for n in local:
                        point=np.asarray(n['position_m']);ax.scatter(point[dims[0]],point[dims[1]],marker='o',facecolors='none',edgecolors='purple',s=70)
                        ax.annotate(n['node']+' degree'+str(n['degree']),point[list(dims)],fontsize=7)
                    ax.scatter(0,0,marker='+',c='black');ax.add_patch(plt.Circle((0,0),10,fill=False,linestyle=':',color='black'))
                    ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',title='history '+str(frame)+' / '+('XY' if dims[1]==1 else 'XZ'))
            axs[0,0].legend(fontsize=7);fig.suptitle('Case '+str(e['case'])+': observations + CONSTRUCTION axes (not predicted openings)')
            fig.savefig(run/'previews'/('case_'+str(e['case'])+'_axes.png'),dpi=120);plt.close(fig)
        out.append(result)
        print(json.dumps(dict(case=e['case'],local_nodes=local),ensure_ascii=False),flush=True)
    write(run/'artifacts/alignment.json',dict(cases=out,teacher_calls=0,training_steps=0,new_labels=0))
    write(run/'metrics/summary.json',dict(status='GATE_MIXED',source_cases=12,interface_cases=[9,11],new_labels=0,training_steps=0,teacher_calls=0))
    write(run/'RUN_STATE.json',dict(state='AWAITING_IN_SESSION_INTERPRETATION'),'w')

def seal():
    run=ROOT/RUN
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='AWAITING_IN_SESSION_INTERPRETATION':raise ValueError('not ready')
    if not (run/'logs/interpretation.md').stat().st_size:raise ValueError('missing interpretation')
    write(run/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for m in ('freeze','execute','seal'):g.add_argument('--'+m,action='store_true')
    a=p.parse_args();globals()[next(m for m in ('freeze','execute','seal') if getattr(a,m))]()
