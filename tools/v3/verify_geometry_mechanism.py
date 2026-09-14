"""Bounded saved-input intervention: direction filter versus spatial structure.

Not a detector benchmark or a new model: transform ONLY retained predictions,
keep actual scans and poses unchanged, replay the same causal backend.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

SLUG = 'gse_geometry_mechanism_intervention_v1'
PARENT = 'results/gate6_single_robot/gate6_20260910_gse_learned_constrained_execution_v1_seed11'
CARD = 'configs/v3/gate6/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate6/'+SLUG+'.json'
RUN = 'results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11'
IMAGE = 'sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj, mode='x'):
    with path.open(mode) as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def freeze():
    if (ROOT/RUN).exists():raise FileExistsError('cannot refreeze an existing run')
    trace = PARENT+'/artifacts/geometry/live_geometry.jsonl'
    bag = PARENT+'/artifacts/sensors.bag'
    seal = {p: h for h, p in (s.split('  ', 1) for s in
            (ROOT/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    inputs = {p: sha(ROOT/p) for p in [trace, bag]}
    assert all(inputs[p] == seal[p] for p in inputs)
    rows = [json.loads(l) for l in (ROOT/trace).read_text().splitlines()]
    records = [r['result']['geometry'] for r in rows if r.get('result')]
    keys = sorted({k for g in records for k in g['source_frame_keys']})
    assert (len(rows), len(records), len(keys)) == (293, 289, 293)
    approval = dict(status='APPROVED', approved_by='user-standing-development-authorization',
        approved_at='2026-09-10', authorized_gates=[6], authorized_operations=['audit'],
        confirmation_reference='User: 那你就验证我现在真懵了; preceding continued-work authorization',
        scope='Saved development-input mechanism test, no training, labels, new simulation or protected worlds')
    scope = dict(worlds=['tunnel'], seed=11, episodes=1, raw_scans=293, effective_observations=289,
        independent_units='one historical development trajectory, not 289 independent scenes',
        temporal_spacing_s=[min(b['timestamp']-a['timestamp'] for a,b in zip(records,records[1:])),
                            max(b['timestamp']-a['timestamp'] for a,b in zip(records,records[1:]))],
        spatial_spacing='original recorded poses, no resampling', split='development only', teacher='none',
        source_frame_keys=keys, source_sha256=inputs, training=False, protected_worlds_read=False,
        interventions=['original retained axes', 'reflect retained axes about sensor, unchanged scan',
                       'rotate retained axes 90 degrees about sensor Z, unchanged scan'],
        explanation='Reflection preserves 10m domain and unoriented segment directions but changes spatial alignment. Rotation is a direction-sensitive positive control.',
        output_scope='causal targets, metric graph, observation places; NOT structural detection accuracy')
    approval['scope_sha256'] = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    write(ROOT/CARD, dict(schema_version='gse_geometry_mechanism_audit_v1', card_id=SLUG, approval=approval, scope=scope), 'w')
    sources = list((ROOT/'src/mtare_topo').rglob('*.py'))+[ROOT/CARD, Path(__file__), ROOT/'tools/v3/_bootstrap.py']
    command = ['docker','run','--rm','--network','none','--memory','4g','--cpus','2',
        '--env','OPENBLAS_NUM_THREADS=1','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount','type=bind,src='+str(ROOT)+',dst=/workspace,readonly',
        '--mount','type=bind,src='+str(ROOT/RUN)+',dst=/output',
        '--entrypoint','/bin/bash',IMAGE,'-c',
        'source /opt/ros/noetic/setup.bash && export PYTHONPATH=/workspace/src:$PYTHONPATH && python3 /workspace/tools/v3/verify_geometry_mechanism.py --execute --output /output']
    write(ROOT/SPEC, dict(schema_version='v3_run_spec_v1', gate=6,date='20260910',slug=SLUG,seed=11,
        operation='audit',data_card=CARD,config_path=CARD,user_authorization=approval, command=command,
        question='Does the current interface use spatial primitive layout, or only an unoriented direction filter?',
        method='Two fixed interventions on saved retained axes; identical actual scans, poses and causal backend',
        baseline='Unmodified saved predictions; reproduce all 289 original candidate decisions first',
        estimated_cost=dict(compute='CPU 2 threads, 4GiB, no model calls or simulation',disk_gb=.1,wall_time_hours=.1),
        acceptance_criteria=['289 exact original candidate reconstructions',
            'Record every changed axis/target/place and graph; no semantic accuracy claim',
            'Spatial-use hypothesis unsupported if reflection leaves all executable targets and place histories unchanged'],
        expected_evidence=['source-bound card, per-observation comparisons, three graphs and place histories, summary, raw log, seal'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},input_sha256=inputs), 'w')
    print(SPEC)


def execute(out):
    import numpy as np
    import rosbag
    from sensor_msgs import point_cloud2
    from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
    from mtare_topo.integration.geometry_constrained_tasks import constrained_tasks
    from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
    from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
    from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
    from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG, _sector_mask
    spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text())
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    assert card['approval']['scope_sha256']==hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest()
    for p,h in dict(spec['source_sha256'],**spec['input_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('input/source drift: '+p)
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    start=time.monotonic();error=None;summary={}
    try:
        rows=[r['result'] for r in map(json.loads,(ROOT/PARENT/'artifacts/geometry/live_geometry.jsonl').read_text().splitlines()) if r.get('result')]
        wanted={r['geometry']['source_frame_keys'][-1]:r for r in rows}
        variants=['original','reflected','rotated_90']
        graphs={k:GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4) for k in variants}
        places={k:GeometryBranchPlaces(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3) for k in variants}
        comparisons=[];seen=set();changed_geometry=0;totals={k:0 for k in variants}
        with rosbag.Bag(str(ROOT/PARENT/'artifacts/sensors.bag')) as bag, (out/'artifacts/comparisons.jsonl').open('x') as log:
            for _,msg,_ in bag.read_messages(topics=['/velodyne_points']):
                key=msg.header.frame_id+':'+str(msg.header.stamp.to_nsec())
                if key not in wanted:continue
                if key in seen:raise ValueError('duplicate scan')
                seen.add(key);row=wanted[key];g=row['geometry'];pose=np.asarray(g['sensor_to_local_odometry'])
                ranges,valid,_=aee_organized_pointcloud2_to_range_image(msg,point_cloud2)
                sectors=[]
                for s in RangeExitBaseline().predict(ranges,valid,np.asarray(ELEVATION_DEG))['sectors']:
                    mask=valid.astype(bool)&(ranges>=4)&(np.abs(np.asarray(ELEVATION_DEG))<=5)[:,None]&_sector_mask(s['heading_robot_deg'],s['angular_width_deg'])[None,:]
                    refs=[key+'/ray:'+str(int(i)) for i in np.flatnonzero(mask)]
                    if refs:sectors.append(dict(heading_robot_deg=s['heading_robot_deg'],source_refs=refs))
                entry=dict(timestamp=g['timestamp'],source_frame_keys=g['source_frame_keys'],observed_candidates=len(sectors),variants={})
                saved_proposals={}
                for name in variants:
                    record=deepcopy(g)
                    for p in record['primitives']:
                        axis=np.asarray(p['axis_controls_world_m']);local=(axis-pose[:3,3])@pose[:3,:3]
                        if name=='reflected':
                            local=-local
                            changed_geometry+=int(not np.allclose(axis,local@pose[:3,:3].T+pose[:3,3],atol=1e-9))
                        elif name=='rotated_90':local=local@np.array([[0,1,0],[-1,0,0],[0,0,1]])
                        if name!='original':p['axis_controls_world_m']=(local@pose[:3,:3].T+pose[:3,3]).tolist()
                    proposals,audit=constrained_tasks(record,sectors,lookahead_m=4,direction_tolerance_deg=15)
                    if name=='original' and (audit!=g['task_constraint_audit'] or proposals!=row['decision']['proposals']):
                        raise ValueError('original candidate reconstruction mismatch')
                    decision=graphs[name].update(record,task_proposals=proposals)
                    state=places[name].observe(record,proposals,navigation_node_id=decision['node'])
                    saved_proposals[name]=proposals;totals[name]+=len(proposals)
                    entry['variants'][name]=dict(accepted=len(proposals),audit=audit,place_update=state,
                        selected_target=None if decision['selected'] is None else decision['selected']['axis_target_xyz_m'],
                        targets=[p['axis_target_xyz_m'] for p in proposals])
                entry['reflection_candidates_identical']=saved_proposals['original']==saved_proposals['reflected']
                entry['rotation_candidates_identical']=saved_proposals['original']==saved_proposals['rotated_90']
                comparisons.append(entry);log.write(json.dumps(entry)+'\n')
        assert seen==set(wanted) and len(comparisons)==289
        snapshots={};place_snapshots={}
        for name in variants:
            snapshot=graphs[name].snapshot()
            for d in snapshot['decisions']:d.pop('raw_axis_proposals_not_executable',None)
            snapshots[name]=snapshot;place_snapshots[name]=places[name].snapshot()
            write(out/('artifacts/'+name+'_graph.json'),snapshot)
            write(out/('artifacts/'+name+'_places.json'),place_snapshots[name])
        summary=dict(status='GATE_MIXED',observations=289,observed_candidates=sum(x['observed_candidates'] for x in comparisons),
            accepted_candidates=totals,reflected_primitive_records_changed=changed_geometry,
            reflection_identical_candidate_windows=sum(x['reflection_candidates_identical'] for x in comparisons),
            rotation_changed_candidate_windows=sum(not x['rotation_candidates_identical'] for x in comparisons),
            reflection_executable_graph_identical=snapshots['original']==snapshots['reflected'],
            reflection_place_history_identical=place_snapshots['original']==place_snapshots['reflected'],
            empty_structure_outputs=sum(not r['geometry']['structures'] for r in rows),
            place_counts={k:dict(total=len(v['places']),supported=sum(p['state']=='supported_observation' for p in v['places'])) for k,v in place_snapshots.items()},
            hypothesis='spatial primitive arrangement influences current hybrid executable graph',
            hypothesis_supported=not (snapshots['original']==snapshots['reflected'] and place_snapshots['original']==place_snapshots['reflected']),
            training_steps=0,model_calls=0,ground_truth_accuracy_measured=False,
            limitation='Consumer-level necessary-condition test, not fair model training comparison, detector scores or counterfactual closed-loop performance. Original v13 tracking policy fixed across variants.')
    except Exception:error=traceback.format_exc()
    summary.update(error=error,elapsed_s=time.monotonic()-start)
    if error:summary['status']='GATE_FAIL'
    write(out/'metrics/summary.json',summary);write(out/'logs/raw.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(Path(RUN)/p.relative_to(out))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute and a.output:raise SystemExit(execute(a.output))
    else:p.error('choose --freeze or --execute --output')
