"""One bounded real C07 traversal fragment; engineering integration, not autonomy."""
from _bootstrap import PROJECT_ROOT as ROOT
from pathlib import Path
import argparse, hashlib, json, resource, signal, sys, time, traceback, zipfile

CODE = Path(__file__).resolve().parents[2]
NAME = 'gse_short_observation_chain_v1'
CARD = f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC = f'configs/v3/gate3/{NAME}.json'
RUN = f'results/gate3_semantics/gate3_20260912_{NAME}_seed0'
OLD = 'configs/v3/gate3/data_cards/gse_conditional_development_inputs_v1.json'
MANIFEST = 'configs/v3/gate3/gse_conditional_development_manifest_v1.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()

def write(p,x,mode='x'):
    with p.open(mode) as f: json.dump(x,f,indent=2,allow_nan=False)

def freeze():
    old=json.loads((ROOT/OLD).read_text())['scope']
    entries=json.loads((ROOT/MANIFEST).read_text())['entries']
    # First existing identity-order ellipse fragment with >=8 decisions.
    e=next(e for e in entries if e['variant']=='ellipse' and e['available_decisions']>=8)
    first=e['frame_rows'][0]-e['decision_index_in_traversal']
    seq=e['sequence_row']-e['decision_index_in_traversal']
    frames=list(range(first,first+12)); sequences=list(range(seq,seq+8))
    paths=old['task_prefixes'][e['task']]
    plans={}; pins={OLD:sha(ROOT/OLD),MANIFEST:sha(ROOT/MANIFEST)}
    from mtare_topo.governance_surface_input import SEALS
    archive={}
    for seal in SEALS.values():
        if sha(ROOT/seal['path'])!=seal['sha256']:raise ValueError('archive seal drift')
        pins[seal['path']]=seal['sha256']
        with (ROOT/seal['path']).open() as f:
            for line in f:
                digest,path=line.strip().split(None,1)
                if path.startswith(paths['sensor']+'/') or path.startswith(paths['teacher']+'/'):archive[path]=digest
    for role,fields,rows in [('sensor',['range_m','valid_mask','sensor_xyz_m','yaw_deg'],frames),
                            ('teacher',['frame_row','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg'],sequences)]:
        for field in fields:
            prefix=paths[role]+'/'+field; original=old['array_plans'][prefix];h=original['header']
            keys=['.'.join(map(str,[k]+[0]*(len(h['shape'])-1))) for k in sorted({i//h['chunks'][0] for i in rows})]
            plans[field]=dict(prefix=prefix,header=h,selected_rows=rows,chunk_keys=keys)
            for p in [prefix+'/.zarray',*[prefix+'/'+k for k in keys]]:
                digest=archive[p]
                if sha(ROOT/p)!=digest: raise ValueError('sealed source drift '+p)
                pins[p]=digest
    scope=dict(identity=e,frame_rows=frames,sequence_rows=sequences,frames=12,observations=8,
        independent_units=1,parents=1,variants=1,split='C07 historical development',
        sampling='First existing identity-order ellipse traversal with at least8 decisions; first8 only, no score selection',
        spacing='Original 1m decision arc spacing; actual sensor displacement recorded; no timing evidence',
        timestamp_kind='ordered_frame_index_not_seconds',continuous_scope='Single generated directed traversal fragment, not continuous world or robot execution',
        methods=['range_sectors','observed_surface_fit'],array_plans=plans,
        labels_generated=0,training_steps=0,teacher_use='Sequence alignment only; no structural labels, construction IDs or geometry teacher in forward',
        limits=dict(host_bytes=4*1024**3,wall_seconds=900,output_bytes=512*1024**2),
        limitations=['Existing horizontal sector nomination; not general3D opening detection','Metric anchors are not detected junctions',
            'No structural precision/recall without qualified reference','No M-TARE execution, autonomy, loop closure or learned relation claim'])
    approval=dict(status='APPROVED',approved_by='user-standing-development-execution',approved_at='2026-09-12',
        authorized_operations=['data_export'],authorized_gates=[3],
        confirmation_reference='User 所以继续啊你怎么老卡住完整链路总是不做完 following explicit limited chain proposal; standing developer-data authorization',
        scope='One12-frame C07 sequential engineering replay; no training, teacher labels or formal graph benefit claim',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_short_chain import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(CODE)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (CODE/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260912',slug=NAME,seed=0,
        operation='data_export',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__)),'--execute'],
        question='Where does the actual scan-to-candidate-to-task chain lose information on one recorded fragment?',
        method='Reuse frozen nonlearning surface fit and original range sectors, same task backend and given-pose anchors',
        baseline='Original range sector frontend; no fair learned geometry advantage claim',
        fallback='Preserve failure; no candidate replacement, hidden reference or threshold tuning',
        estimated_cost=dict(compute='One CPU, zero training/GPU',host_ram_gb=4,disk_gb=.5,wall_time_hours=.25),
        acceptance_criteria=['12 exact frames and8 contiguous windows','All candidates and rejected fit reasons saved','No invented edges across fragments','Original decisions immutable and prefix replay equal','No autonomous or graph-quality success claim'],
        expected_evidence=['Input hashes, raw scans/poses, geometry traces, task states, metric graph, XY/XZ preview, failure reasons, logs and seal'],
        input_sha256=pins,execution_code_root=str(CODE),execution_source_sha256=code,
        environment=dict(python=sys.version,executable=sys.executable,executable_sha256=sha(Path(sys.executable).resolve())))
    write(ROOT/SPEC,spec);print(json.dumps(dict(spec=SPEC,scope=scope),ensure_ascii=False))

def read_rows(plan):
    import numpy as np, numcodecs
    h=plan['header'];codec=numcodecs.get_codec(h['compressor']);rows={}
    for key in plan['chunk_keys']:
        data=np.frombuffer(codec.decode((ROOT/(plan['prefix']+'/'+key)).read_bytes()),dtype=h['dtype']).reshape(h['chunks'])
        first=int(key.split('.')[0])*h['chunks'][0]
        for i in plan['selected_rows']:
            if first<=i<first+len(data):rows[i]=data[i-first].copy()
    return np.stack([rows[i] for i in plan['selected_rows']])

def range_proposals(ranges,valid,pose,key):
    import numpy as np
    from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
    from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG,_sector_mask
    result=[]
    for s in RangeExitBaseline().predict(ranges,valid,np.asarray(ELEVATION_DEG))['sectors']:
        mask=_sector_mask(s['heading_robot_deg'],s['angular_width_deg'])
        supported=valid.astype(bool)&(ranges>=4)&(np.abs(np.asarray(ELEVATION_DEG))<=5)[:,None]&mask[None,:]
        refs=[key+'/ray:'+str(int(i)) for i in np.flatnonzero(supported)]
        if not refs:continue
        a=np.deg2rad(s['heading_robot_deg']);d=pose[:3,:3]@np.array([np.cos(a),np.sin(a),0.])
        result.append(dict(axis_start_xyz_m=pose[:3,3].tolist(),axis_target_xyz_m=(pose[:3,3]+4*d).tolist(),
            source_refs=refs,current_direction_supported=True,geometry_source_kind='range_sector_not_primitive',
            creates_edge=False,physical_opening=False,traversability='unknown'))
    return result

def execute():
    import numpy as np
    from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline
    from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
    from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
    from mtare_topo.governance_short_chain import validate_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert validate_card(card).passed
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for p,h in spec['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
    for p,h in spec['execution_source_sha256'].items():
        if sha(CODE/p)!=h:raise ValueError('source drift '+p)
    start=time.monotonic();error=None;rows=[];checks={};write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*_):raise TimeoutError('900 second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(s['limits']['wall_seconds'])
    def check():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise MemoryError('RSS cap')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise RuntimeError('output cap')
    try:
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(CODE/p,p)
        data={f:read_rows(p) for f,p in s['array_plans'].items()};check()
        expected=np.array([s['frame_rows'][i:i+5] for i in range(8)])
        if not np.array_equal(data['frame_row'],expected):raise ValueError('source window continuity mismatch')
        poses=np.tile(np.eye(4),(12,1,1));poses[:,:3,3]=data['sensor_xyz_m']
        for i,a in enumerate(np.deg2rad(data['yaw_deg'])):poses[i,:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
        for j in range(8):
            i=j+4;t=(poses[j:i+1,:3,3]-poses[i,:3,3])@poses[i,:3,:3]
            if not np.allclose(t,data['relative_translation_current_sensor_m'][j],atol=2e-5,rtol=1e-6):raise ValueError('relative pose mismatch')
            yaw=(data['yaw_deg'][j:i+1]-data['yaw_deg'][i]+180)%360-180
            delta=(yaw-data['relative_yaw_current_sensor_deg'][j]+180)%360-180
            if np.max(np.abs(delta))>2e-5:raise ValueError('relative yaw mismatch')
        np.savez_compressed(out/'artifacts/inputs.npz',ranges_m=data['range_m'],valid_mask=data['valid_mask'],poses=poses,frame_rows=s['frame_rows'])
        pipe=LiveGeometryPipeline(composition_policy=dict(max_residual_m=.01,min_crossing_sine=.1,endpoint_tolerance_m=1e-8,maximum_candidates=32),anchor_spacing_m=4,lookahead_m=4,rigid_motion=True)
        trackers={m:GeometryBranchPlaces(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3) for m in s['methods']}
        for i,frame in enumerate(s['frame_rows']):
            key=s['identity']['task']+'/frame:'+str(frame)
            result=pipe.push(data['range_m'][i],data['valid_mask'][i],sensor_to_map=poses[i],stamp_sec=float(frame),source_key=key,coordinate_frame='given_pose_map')
            if result is None:continue
            r=result['geometry'];r['timestamp_kind']='ordered_frame_index_not_seconds';d=result['decision']
            alternatives={'range_sectors':range_proposals(data['range_m'][i],data['valid_mask'][i],poses[i],key),'observed_surface_fit':d['proposals']}
            events={m:trackers[m].observe(r,p,navigation_node_id=d['node']) for m,p in alternatives.items()}
            row=dict(frame_row=frame,geometry=r,decision=d,method_proposals=alternatives,task_events=events)
            rows.append(row);write(out/'artifacts'/f'window_{i-4:02d}.json',row)
            print(json.dumps(dict(window=i-4,counts={m:len(p) for m,p in alternatives.items()},fit_reasons=[p['fit_status'] for p in r['primitives']])),flush=True);check()
        # Replay the saved first half, not a new frontend or new experiment.
        prefix=GeometryNavigationReplay(anchor_spacing_m=4,lookahead_m=4)
        for row in rows[:4]:
            if prefix.update(row['geometry'])!=row['decision']:raise ValueError('prefix decision mismatch')
        graph=pipe.graph.snapshot()
        if graph['decisions']!=[row['decision'] for row in rows]:raise ValueError('historical decision mutation')
        checks=dict(source_windows_exact=True,relative_motion_matches=True,prefix_decisions_equal=True,historical_decisions_unchanged=True)
        write(out/'artifacts/graph.json',graph);write(out/'artifacts/task_states.json',{m:t.snapshot() for m,t in trackers.items()})
        write(out/'artifacts/trajectory.json',dict(kind='generated_sensor_pose_not_robot_execution',frame_rows=s['frame_rows'],xyz_m=poses[:,:3,3].tolist()))
        from collections import Counter
        counts={m:dict(proposals=sum(len(r['method_proposals'][m]) for r in rows),place_events=dict(Counter(r['task_events'][m]['status'] for r in rows)),stored_places=len(trackers[m].places)) for m in trackers}
        summary=dict(methods=counts,metric_nodes=len(graph['nodes']),recorded_fragment_edges=len(graph['edges']),
            fit_reasons=dict(Counter(p['fit_status'] for r in rows for p in r['geometry']['primitives'])),checks=checks)
        write(out/'metrics/chain.json',summary)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,2,figsize=(12,9))
        for col,m in enumerate(s['methods']):
            for line,axes in enumerate([(0,1),(0,2)]):
                ax=axs[line,col];a,b=axes;ax.plot(poses[:,a,3],poses[:,b,3],'k.-',label='recorded sensor route')
                for row in rows:
                    for p in row['method_proposals'][m]:
                        x=np.asarray([p['axis_start_xyz_m'],p['axis_target_xyz_m']]);ax.plot(x[:,a],x[:,b],alpha=.35,color='tab:orange')
                for n in graph['nodes']:ax.scatter(n['xyz_m'][a],n['xyz_m'][b],color='tab:blue')
                ax.set_title(m+' / '+('XY' if line==0 else 'XZ'));ax.set_aspect('equal');ax.grid(True);ax.set_xlabel('x (m)');ax.set_ylabel(('y' if line==0 else 'z')+' (m)')
        fig.suptitle('Real scan fragment: orange=candidate only; blue=metric anchor; NOT closed-loop exploration')
        fig.tight_layout();fig.savefig(out/'artifacts/chain_preview.png',dpi=160);plt.close(fig);check()
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,windows=len(rows),checks=checks,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        training_steps=0,closed_loop=False,structural_accuracy_measured=False,advantage_proven=False)
    write(out/'metrics/summary.json',result);write(out/'logs/raw_result.json',result)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
