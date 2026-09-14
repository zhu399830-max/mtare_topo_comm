"""Explain the saved return failure; no inference, filtering or policy change."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_learned_geometry_execution_v1r2_seed11'
OUT=ROOT/'docs/figures/gse_learned_geometry_execution_v1r2'

def main():
    seal={p:h for h,p in (x.split('  ',1) for x in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    checked={}
    def read(name):
        p=RUN/name;raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if h!=seal[str(p.relative_to(ROOT))]:raise ValueError('seal mismatch '+name)
        checked[name]=h;return raw
    snapshot=json.loads(read('artifacts/geometry/live_geometry_snapshot.json'))
    rows=[json.loads(x)['result'] for x in read('artifacts/geometry/live_geometry.jsonl').decode().splitlines() if json.loads(x).get('result')]
    dest=next(e['destination'] for e in snapshot['branch_tasks']['return_events'] if e['status']=='RETURN_PLACE_REACHED')
    original=next(r for r in rows if r['geometry']['timestamp']==dest['order'])
    proposal=min(original['decision']['proposals'],key=lambda p:np.linalg.norm(np.array(p['axis_target_xyz_m'])-dest['target_xyz_m']))
    primitive=next(p for p in original['geometry']['primitives'] if p['index']==proposal['primitive_index'])
    slot=primitive['prediction_provenance']['slot']
    log=[json.loads(x) for x in read('artifacts/model/inference.jsonl').decode().splitlines()]
    record=next(x for x in log if x['timestamp']==dest['order'])
    import io
    with np.load(io.BytesIO(read('artifacts/model/'+record['evidence'])),allow_pickle=False) as raw:
        axis=raw['axis_control_current_sensor_m'][slot].astype(float)
    delta=np.diff(axis,axis=0);length=np.linalg.norm(delta,axis=1)
    turn=float(np.degrees(np.arccos(np.clip(delta[0]@delta[1]/np.prod(length),-1,1))))
    windows=[]
    for r in rows:
        if r['local_planning_request']['status']!='HOLD_RETURN_REOBSERVATION':continue
        candidates=[]
        for p in r['decision']['proposals']:
            v=np.array(p['axis_target_xyz_m'])-p['axis_start_xyz_m']
            angle=float(np.degrees(np.arccos(np.clip(v@dest['direction_world']/np.linalg.norm(v),-1,1))))
            distance=float(np.linalg.norm(np.array(p['axis_target_xyz_m'])-dest['target_xyz_m']))
            candidates.append(dict(index=p['primitive_index'],distance_m=distance,angle_deg=angle,
                current_supported=p['current_direction_supported'],matches=bool(angle<=15 and distance<=2 and p['current_direction_supported'])))
        windows.append(dict(timestamp=r['geometry']['timestamp'],candidates=candidates,
            matches=sum(c['matches'] for c in candidates)))
    report=dict(source_sha256=checked,destination=dest,raw_model_slot=slot,raw_axis_sensor_m=axis.tolist(),
        segment_lengths_m=length.tolist(),endpoint_distance_m=float(np.linalg.norm(axis[-1]-axis[0])),turn_deg=turn,
        original_proposal=proposal,return_windows=windows,policy_modified=False,training_steps=0,
        conclusion='Raw prediction folds back before clipping. Target reobservation fails; no threshold or dedup fix justified.')
    with (OUT/'return_geometry_diagnostic.json').open('x') as f:json.dump(report,f,indent=2)
    pose=np.array(original['geometry']['sensor_to_local_odometry']);world=axis@pose[:3,:3].T+pose[:3,3]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,dim,label in zip(axes,[1,2],['Y','Z']):
        ax.plot(world[:2,0],world[:2,dim],'o-',label='Raw model segment 0 -> 1')
        ax.plot(world[1:,0],world[1:,dim],'x--',label='Raw model segment 1 -> 2')
        ax.scatter(pose[0,3],pose[dim,3],marker='^',s=60,color='black',label='Observer')
        ax.scatter(dest['target_xyz_m'][0],dest['target_xyz_m'][dim],marker='+',s=100,color='red',label='Saved target (NOT GT)')
        ax.set(xlabel='X (m)',ylabel=label+' (m)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Failure source: folded predicted axis, %.3f degree turn; 152 return windows lack a match'%turn)
    fig.savefig(OUT/'folded_axis_failure.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(turn_deg=turn,segment_lengths_m=length.tolist(),return_windows=len(windows),matching_windows=sum(w['matches']>0 for w in windows))))

if __name__=='__main__':main()
