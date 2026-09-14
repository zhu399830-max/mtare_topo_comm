"""Replay sealed axes in one common ball. No fitting, inference or control."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from copy import deepcopy
import hashlib
import json
import platform
import time
import numpy as np
from mtare_topo.integration.common_geometry_domain import common_domain_record
from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG, _sector_mask

SLUG='gse_common_geometry_domain_v1'
SPEC='configs/v3/gate6/'+SLUG+'.json'
CARD='configs/v3/gate6/data_cards/'+SLUG+'.json'
RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed0'
PARENT='results/gate6_single_robot/gate6_20260910_gse_learned_geometry_pair_v1_seed0'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def write(p,x,mode='x'):
    with p.open(mode) as f:json.dump(x,f,indent=2,allow_nan=False)

def freeze():
    card=json.loads((ROOT/'configs/v3/gate6/data_cards/gse_learned_geometry_pair_v1.json').read_text())
    inputs=[PARENT+'/artifacts/inputs.npz',PARENT+'/artifacts/pair/predictions.jsonl']
    inputs += [PARENT+'/artifacts/pair/'+n+'_graph.json' for n in ['learned','nonlearning']]
    seal=dict(line.strip().split('  ',1)[::-1] for line in (ROOT/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines())
    for p in inputs:
        if sha(ROOT/p)!=seal[p]:raise ValueError('parent seal mismatch '+p)
    card['scope']['source_sha256']={p:seal[p] for p in inputs}
    card['scope']['operation_detail']='Saved axes backend replay only; zero model calls, fitting, training or control'
    card['scope']['common_domain_radius_m']=10
    card['scope']['domain_policy']='Keep all intersecting pieces; never bridge disconnected pieces or certify clipping boundary as opening'
    card['approval']['scope_sha256']=hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest()
    card['approval']['scope']=card['scope']['operation_detail']
    write(ROOT/CARD,card)
    original=json.loads((ROOT/'configs/v3/gate6/gse_learned_geometry_pair_v1.json').read_text())
    original.update(slug=SLUG,data_card=CARD,config_path=CARD,
        command=['python3','tools/v3/replay_common_geometry_domain.py','--execute'],
        question='How do saved predicted and fitted geometry affect the identical backend in the same 10m output domain?',
        method='Frozen saved learned axes and prediction provenance; same clipping and graph policy as baseline',
        baseline='Frozen saved fitted axes and original ray provenance; no refitting',
        estimated_cost=dict(compute='CPU one thread, no GPU, no simulator; bounded 290 windows',disk_gb=2,wall_time_hours=.1),
        acceptance_criteria=['290 decisions per method, exact saved axes/provenance binding',
            'All targets within shared 10m ball; no clipping-derived openings or inferred edges',
            'No training, inference, protected data or semantic accuracy claim'],
        input_sha256=card['scope']['source_sha256'])
    sources=list((ROOT/'src/mtare_topo').rglob('*.py'))+[ROOT/CARD,ROOT/'tools/v3/replay_common_geometry_domain.py',ROOT/'tools/v3/_bootstrap.py']
    original['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sources}
    original['user_authorization']['scope']=card['scope']['operation_detail']
    write(ROOT/SPEC,original)
    print(SPEC)

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    for p,h in dict(spec['source_sha256'],**spec['input_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');started=time.monotonic();error=None
    try:
        data=np.load(ROOT/PARENT/'artifacts/inputs.npz',allow_pickle=False)
        pairs=[json.loads(s) for s in (ROOT/PARENT/'artifacts/pair/predictions.jsonl').read_text().splitlines()]
        history={n:json.loads((ROOT/PARENT/('artifacts/pair/'+n+'_graph.json')).read_text())['graph']['decisions'] for n in ['learned','nonlearning']}
        assert len(pairs)==290 and len(data['poses'])==294
        graphs={n:GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4) for n in history}
        places={n:GeometryBranchPlaces(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3) for n in history}
        stats={n:dict(observations=0,primitives=0,proposals=0,current_supported=0,rejected=0) for n in history}
        with (run/'artifacts/records.jsonl').open('x') as output:
            for j,pair in enumerate(pairs):
                if time.monotonic()-started>360:raise TimeoutError('six minute budget exceeded')
                i=j+4;mask=np.zeros(720,bool)
                for sector in RangeExitBaseline().predict(data['range_valid'][i,0]*50,data['range_valid'][i,1],np.asarray(ELEVATION_DEG))['sectors']:
                    mask|=_sector_mask(sector['heading_robot_deg'],sector['angular_width_deg'])
                row={}
                for n in history:
                    old=history[n][j];axes=pair['methods'][n]['axes']
                    assert old['order']==pair['timestamp']==float(data['stamps'][i])
                    assert old['source_frame_keys']==list(data['source_keys'][i-4:i+1])
                    # Learned indices are model slots, not dense filtered-axis indices.
                    ids=sorted({p['primitive_index'] for p in old['proposals']})
                    if n=='learned' and len(ids)!=len(axes):raise ValueError('missing saved learned provenance')
                    primitives=[]
                    for k,axis in enumerate(axes):
                        if axis is None:continue
                        index=ids[k] if n=='learned' else k
                        proposals=[p for p in old['proposals'] if p['primitive_index']==index]
                        if not proposals:raise ValueError('missing saved axis provenance')
                        p=dict(index=index,axis_controls_world_m=axis,current_sector_columns=np.flatnonzero(mask).tolist(),source_rays=[])
                        if n=='learned':
                            p.update(fit_status='model_prediction_not_surface_fit',prediction_provenance=deepcopy(proposals[0]['prediction_provenance']))
                        else:
                            for ref in proposals[0]['source_refs']:
                                frame,ray=ref.rsplit('/ray:',1)
                                if frame not in old['source_frame_keys']:raise ValueError('ray outside window')
                                p['source_rays'].append(dict(frame_key=frame,ray_index=int(ray)))
                        primitives.append(p)
                    record=dict(schema_version='geometry_structure_trace_v1',timestamp=pair['timestamp'],coordinate_frame='map',
                        sensor_to_local_odometry=data['poses'][i].tolist(),source_frame_keys=old['source_frame_keys'],structures=[],primitives=primitives)
                    record=common_domain_record(record)
                    d=graphs[n].update(record);event=places[n].observe(record,d['proposals'],navigation_node_id=d['node'])
                    for p in d['proposals']:
                        assert np.linalg.norm(np.array(p['axis_target_xyz_m'])-data['poses'][i,:3,3])<=10+1e-8
                        assert not p['physical_opening'] and not p['creates_edge']
                    s=stats[n];s['observations']+=1;s['primitives']+=len(record['primitives']);s['rejected']+=len(record['domain_rejections'])
                    s['proposals']+=len(d['proposals']);s['current_supported']+=sum(p['current_direction_supported'] is True for p in d['proposals'])
                    row[n]=dict(axes=[p['axis_controls_world_m'] for p in record['primitives']],
                        parent_indices=[p['parent_primitive_index'] for p in record['primitives']],rejections=record['domain_rejections'],place_event=event)
                output.write(json.dumps(dict(timestamp=pair['timestamp'],methods=row),allow_nan=False)+'\n')
        for n in history:
            g=graphs[n].snapshot();pl=places[n].snapshot()
            write(run/('artifacts/'+n+'_graph.json'),dict(graph=g,places=pl))
            stats[n].update(nodes=len(g['nodes']),edges=len(g['edges']),places=len(pl['places']))
        write(run/'artifacts/environment.json',dict(python=platform.python_version(),numpy=np.__version__))
        write(run/'metrics/comparison.json',dict(methods=stats,domain_radius_m=10,model_calls=0,training_steps=0,
            semantic_accuracy_measured=False,advantage_proven=False,notes='Counts are not accuracy. Metric nodes/edges depend on shared recorded poses.'))
    except Exception as exc:error=repr(exc)
    finally:
        write(run/'logs/execution.json',dict(command=spec['command'],completed_observations=stats if 'stats' in locals() else {},error=error))
        write(run/'metrics/summary.json',dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-started))
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
        with (run/'artifacts/evidence_sha256.txt').open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print((run/'metrics/summary.json').read_text());return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('action required')
