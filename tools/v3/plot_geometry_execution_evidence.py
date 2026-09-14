"""Render a sealed development execution trace without modifying its run."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import os
import argparse
from pathlib import Path

RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_live_geometry_execution_v3_seed11'
OUT=ROOT/'docs/figures/gse_geometry_execution_v3'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    global RUN,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--branch',action='store_true');parser.add_argument('--return-run',action='store_true');parser.add_argument('--binding-run',action='store_true');parser.add_argument('--direct-run',action='store_true')
    parser.add_argument('--learned-run',choices=['v1r1','v1r2','constrained_v1','memory_v1']);args=parser.parse_args()
    if args.branch:
        RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_branch_geometry_execution_v1_seed11'
        OUT=ROOT/'docs/figures/gse_branch_geometry_execution_v1'
    if args.return_run:
        RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_branch_geometry_return_v1_seed11'
        OUT=ROOT/'docs/figures/gse_branch_geometry_return_v1'
    if args.binding_run:
        RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_branch_geometry_return_binding_v1_seed11'
        OUT=ROOT/'docs/figures/gse_branch_geometry_return_binding_v1'
    if args.direct_run:
        RUN=ROOT/'results/gate6_single_robot/gate6_20260910_gse_branch_geometry_direct_return_v1_seed11'
        OUT=ROOT/'docs/figures/gse_branch_geometry_direct_return_v1'
    if args.learned_run:
        slug='gse_learned_constrained_execution_v1' if args.learned_run=='constrained_v1' else 'gse_learned_geometry_execution_'+args.learned_run
        if args.learned_run=='memory_v1':slug='gse_learned_constrained_memory_v1'
        RUN=ROOT/('results/gate6_single_robot/gate6_20260910_'+slug+'_seed11')
        OUT=ROOT/('docs/figures/'+slug)
    trace=RUN/'artifacts/geometry/live_geometry.jsonl';summary=RUN/'metrics/summary.json'
    seal=dict((r.split('  ',1)[1],r.split('  ',1)[0]) for r in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())
    for p in (trace,summary):
        if sha(p)!=seal[str(p.relative_to(ROOT))]:raise ValueError('source seal mismatch')
    poses=[];targets=[];nodes={};structures=0;max_age=0.;statuses={};branch_targets=[];return_poses=[];errors=[]
    for line in trace.open():
        r=json.loads(line)
        if 'error' in r:
            if not args.learned_run:raise ValueError(r['error'])
            errors.append(r);continue
        max_age=max(max_age,r['pose_evidence']['past_pose_age_sec'])
        result=r.get('result')
        if result is None:continue
        g=result['geometry'];xyz=[row[3] for row in g['sensor_to_local_odometry'][:3]]
        poses.append(xyz);nodes.setdefault(result['decision']['node'],xyz)
        structures+=bool(g['structures'])
        request=result['local_planning_request'];statuses[request['status']]=statuses.get(request['status'],0)+1
        return_poses.append(xyz if request['status']=='FOLLOW_RECORDED_RETURN' else [float('nan')]*3)
        if request['status']=='NEW_LOCAL_PLANNING_REQUEST':
            targets.append(request['xyz_m'])
            if request.get('active',{}).get('branch_task'):
                branch_targets.append(dict(task=request['active']['branch_task'],xyz_m=request['xyz_m']))
    s=json.loads(summary.read_text())
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gse-execution-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p=np.asarray(poses);n=np.asarray(list(nodes.values()));t=np.asarray(targets)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,dim,label in zip(axes,(1,2),('Y','Z')):
        ax.plot(p[:,0],p[:,dim],'k-',lw=1,label='Live sensor trajectory (estimated pose)')
        if args.return_run or args.binding_run or args.direct_run or args.learned_run:
            back=np.asarray(return_poses);ax.plot(back[:,0],back[:,dim],color='purple',lw=2,label='Recorded-route return execution')
        ax.scatter(n[:,0],n[:,dim],s=20,label='Metric anchors, NOT semantic junctions')
        if len(t):ax.scatter(t[:,0],t[:,dim],marker='+',s=25,label='Geometric local-goal requests')
        if branch_targets:
            b=np.asarray([v['xyz_m'] for v in branch_targets]);ax.scatter(b[:,0],b[:,dim],facecolors='none',edgecolors='red',s=70,label='Selected geometry-place branch tasks')
        ax.set(xlabel='X (m)',ylabel=label+' (m)');ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.suptitle('Actual simulator execution: %.2f m, %d XY arrivals | %s | not full exploration validation'%(
        s['trial']['travel_m'],s['trial']['xy_arrivals'],s['status']))
    fig.tight_layout();fig.savefig(OUT/'trajectory.png',dpi=170);plt.close(fig)
    data=dict(source_run=str(RUN.relative_to(ROOT)),source_sha256={str(x.relative_to(ROOT)):sha(x) for x in (trace,summary)},
        measured_travel_m=s['trial']['travel_m'],xy_arrivals=s['trial']['xy_arrivals'],
        metric_nodes=len(nodes),geometry_observations=len(poses),structural_proposal_observations=structures,
        max_past_pose_age_sec=max_age,request_status_counts=statuses,
        branch_task_requests=branch_targets,branch_task_arrivals=s['trial'].get('branch_task_arrivals',0),
        return_places_reached=s['trial'].get('return_places_reached',0),return_directions_revalidated=s['trial'].get('return_directions_revalidated',0),
        complete_exploration_verified=False,semantic_graph_verified=False,errors=errors)
    with (OUT/'provenance.json').open('x') as f:json.dump(data,f,indent=2)
    print(json.dumps(data))

if __name__=='__main__':main()
