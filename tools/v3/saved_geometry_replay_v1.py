"""One immutable given-pose replay of already sealed geometric observations."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import sys
import time
import traceback
from mtare_topo.governance_geometry_replay import SCHEMA,SLUG,scope,validate_card
from mtare_topo.governance_continuous_geometry import digest

CARD='configs/v3/gate4/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate4/'+SLUG+'.json'
RUN='results/gate4_topology/gate4_20260910_'+SLUG+'_seed0'

def write(p,j,mode='x'):
    with p.open(mode) as f:json.dump(j,f,indent=2,sort_keys=True,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    s=scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_operations=['topology_replay'],authorized_gates=[4],scope_sha256=digest(s),
        scope='Saved30 C04 records, three separate given-pose graphs; no model, teacher or protected data.',
        confirmation_reference='User explicitly changed scope to run complete geometry-to-topology-to-exploration system, first short-route replay; standing autonomous execution authority.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,scope=s,approval=a)
    assert validate_card(card).passed
    (ROOT/CARD).parent.mkdir(parents=True,exist_ok=True);write(ROOT/CARD,card)
    sources=[CARD,'tools/v3/saved_geometry_replay_v1.py','src/mtare_topo/governance_geometry_replay.py',
        'src/mtare_topo/integration/geometry_navigation_replay.py','src/mtare_topo/planning/primitive_corridor.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=4,date='20260910',slug=SLUG,seed=0,
        operation='topology_replay',data_card=CARD,config_path=CARD,user_authorization=a,
        command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',sys.executable,'tools/v3/saved_geometry_replay_v1.py','--execute'],
        question='Does saved observed geometry feed causal target proposals and separate recorded-place graphs?',
        method='Frozen finite-axis outputs;4m metric anchors and4m curve lookahead, no loop merge or inferred edge.',
        baseline='Integration demonstration, not learned semantic graph superiority.',
        estimated_cost=dict(compute='CPU only',disk_gb=.5,wall_time_hours=1/6),
        acceptance_criteria=['Exact30 ordered records,3 separate graphs,all target decisions retained; hashes unchanged.',
            'No cross-variant edges or future observations; output explicitly not closed loop.'],
        expected_evidence=['graphs,decisions,XY/XZ plots,summary,logs,seal'],source_sha256={p:sha(ROOT/p) for p in sources}))
    print(SPEC)

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());start=time.monotonic();error=None;stats=[]
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    if json.loads((run/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        card=json.loads((ROOT/CARD).read_text())
        if not validate_card(card).passed:raise ValueError('card drift')
        for p,h in spec['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('source drift')
        for task,files in card['scope']['groups'].items():
            runtime=GeometryNavigationReplay(anchor_spacing_m=4.,lookahead_m=4.)
            poses=[]
            for item in files:
                if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('input drift')
                r=json.loads((ROOT/item['path']).read_text());runtime.update(r)
                poses.append(np.asarray(r['sensor_to_local_odometry'])[:3,3])
            s=runtime.snapshot();write(run/'artifacts'/f'{task}_graph.json',s)
            fig,axes=plt.subplots(1,2,figsize=(12,4));poses=np.asarray(poses)
            for ax,dim,label in zip(axes,(1,2),('Y','Z')):
                ax.plot(poses[:,0],poses[:,dim],'k.-',label='Recorded sensor path')
                nodes=np.asarray([n['xyz_m'] for n in s['nodes']]);ax.scatter(nodes[:,0],nodes[:,dim],c='blue',label='Metric anchors')
                for p,d in zip(poses,s['decisions']):
                    if d['selected']:
                        q=d['selected']['axis_target_xyz_m'];ax.plot([p[0],q[0]],[p[dim],q[dim]],color='orange',alpha=.5)
                ax.set(xlabel='X (m)',ylabel=label+' (m)');ax.legend();ax.grid()
            fig.suptitle(task+' | development given-pose replay, NOT closed loop')
            fig.tight_layout();fig.savefig(run/'previews'/f'{task}.png',dpi=150);plt.close(fig)
            stats.append(dict(task=task,observations=len(s['decisions']),nodes=len(s['nodes']),edges=len(s['edges']),
                geometry_targets=sum(d['selected'] is not None for d in s['decisions'])))
    except BaseException:error=traceback.format_exc()
    finally:
        result=dict(status='FAILED' if error else 'DEVELOPMENT_REPLAY_COMPLETE',error=error,tasks=stats,
            elapsed_s=time.monotonic()-start,closed_loop=False,semantic_nodes_verified=False)
        write(run/'metrics/summary.json',result);write(run/'logs/raw.json',result)
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
        seal=run/'artifacts/evidence_sha256.txt'
        with seal.open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p!=seal:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
