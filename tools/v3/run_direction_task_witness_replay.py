"""One bounded cached-model task replay against partial ray-section witnesses."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write,range_proposals
from pathlib import Path
import argparse,hashlib,json,resource,sys,time,traceback,zipfile
NAME='gse_direction_task_witness_replay_v1'
SPEC=f'configs/v3/gate3/{NAME}.json'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
CACHE='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'
SOURCE='results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'
ENTRYPOINT=Path(__file__).resolve()
SUPPORT_POLICY='historical_peak_centered_sector_support_v1'
CONTROL_RUN=None

def freeze():
    original='configs/v3/gate3/data_cards/gse_direction_task_model_cache_v1.json'
    source_scope=json.loads((ROOT/original).read_text())['scope']
    manifest=json.loads((ROOT/SOURCE/'artifacts/manifest.json').read_text())
    scope=dict(source_scope=source_scope,manifest=manifest,observations=12,unique_frames=24,parents=3,support_policy=SUPPORT_POLICY,
        split='fit',training_steps=0,model_forwards=0,methods=['POSE_DIRECTION','A','B','C'],control_run=CONTROL_RUN,
        correspondence_policy='Mutual unique positive cosine maxima; exact ties/undefined retain provisional tasks',
        reference_policy='First-return ray strictly crosses inscribed disk at unique outward source-axis crossing of existing10m sphere; all in-domain degree>=3 constructions, not selected model candidates',
        unknown_policy='No crossing or multiple witnessed ports unknown; all predictions retained',
        reference_scope='Partial construction-conditioned ray-section identity; NOT complete channel/traversability labels',
        training_qualified=False,independent_structural_accuracy_qualified=False,
        limits=dict(host_bytes=4*1024**3,output_bytes=1024**3,wall_seconds=300))
    approval=dict(status='APPROVED',approved_by='user-scoped-pipeline-correction',approved_at='2026-09-13',
        authorized_operations=['data_export'],authorized_gates=[3],scope='Same three fit fragments; independent ray evidence and shared task replay, no training',
        confirmation_reference='User 那你改啊 following diagnosed supervision/application mismatch; standing development authority',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_task_witness import validate_card
    assert validate_card(card).passed
    pins={original:sha(ROOT/original)}
    for folder in (CACHE,SOURCE):
        seal=ROOT/folder/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        for line in seal.read_text().splitlines():
            h,p=line.split('  ',1);assert sha(ROOT/p)==h,p;pins[p]=h
    if CONTROL_RUN:
        p=CONTROL_RUN+'/metrics/summary.json';pins[p]=sha(ROOT/p)
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(ENTRYPOINT),'--execute'],
        question='Do same-backend task assignments differ, and which are covered by independent finite-ray section witnesses?',
        method=SUPPORT_POLICY+'; cached ABC directional context; same mutual matching and task ledger; teacher only after all decisions saved',
        baseline='Observed direction with shared recorded pose, same candidates and matching',fallback='Unknown remains unknown; insufficient reference ends unqualified, no training',
        estimated_cost=dict(compute='12windows CPU, zero forwards/optimizer',host_ram_gb=4,gpu_vram_gb=0,disk_gb=1,wall_time_hours=300/3600),
        acceptance_criteria=['All candidates and decisions saved before construction read','Zero cross-segment links and prefix drift','All competing witnesses/unknowns retained','No qualified accuracy or training claim from partial reference'],
        expected_evidence=['Allcandidate predictions, reference crossings, ledger histories, conditional scores, source snapshot, logs, figure and seal'],
        input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np,signal
    from mtare_topo.topology.structural_task_tokens import bind_directional_context
    from mtare_topo.topology.task_correspondence import cosine_matrix,mutual_unique_matches
    from mtare_topo.topology.direction_task_ledger import DirectionTaskLedger
    from mtare_topo.teacher.direction_task_ray_witness import construction_sections,crossed_sections,candidate_witnesses
    from mtare_topo.evaluation.direction_task_maintenance import score_task_history
    from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG
    from mtare_topo.governance_task_witness import validate_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert validate_card(card).passed
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;predictions=[];references=[];summary={}
    def timeout(*_):raise TimeoutError('wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(s['limits']['wall_seconds'])
    try:
        write(out/'config/runtime_environment.json',dict(python=sys.version,numpy=np.__version__,device='CPU'))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(ROOT/p,p)
        ledgers={m:DirectionTaskLedger() for m in s['methods']};previous={m:[] for m in ledgers};last_segment=None
        el=np.deg2rad(ELEVATION_DEG);az=np.deg2rad(np.arange(720)*.5)
        unit=np.stack(np.broadcast_arrays(np.cos(el)[:,None]*np.cos(az),np.cos(el)[:,None]*np.sin(az),np.sin(el)[:,None]+np.zeros((16,720))),axis=-1).reshape(-1,3)
        # All student decisions are completed before any construction document is opened.
        for i,e in enumerate(s['manifest']):
            meta=e['source'];segment=str(meta['fragment_id']);key=meta['task']+'/'+str(meta['frame_rows'][-1])
            with np.load(ROOT/SOURCE/e['student_path'],allow_pickle=False) as data:
                ranges=data['ranges_m'][-1];valid=data['valid_mask'][-1].astype(bool)
            with np.load(ROOT/SOURCE/e['source_evidence_path'],allow_pickle=False) as data:
                # Shared recorded pose diagnostic, not deployment localization.
                origin=data['sensor_xyz_m'][-1];yaw=float(data['yaw_deg'][-1])
            a=np.deg2rad(yaw);pose=np.eye(4);pose[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];pose[:3,3]=origin
            proposals=range_proposals(ranges,valid,pose,key)
            candidates=[dict(candidate=j,source_frame_keys=[key],target_xyz_m=p['axis_target_xyz_m']) for j,p in enumerate(proposals)]
            descriptors={'POSE_DIRECTION':[(np.asarray(p['axis_target_xyz_m'])-p['axis_start_xyz_m']).tolist() for p in proposals]}
            with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as common:centers=common['patch_centers_m']
            for m in 'ABC':
                with np.load(ROOT/CACHE/f'artifacts/{m}_{i:02d}.npz',allow_pickle=False) as data:tokens=data['local_tokens'][0]
                descriptors[m]=[p['descriptor'] for p in bind_directional_context(proposals,centers,tokens,pose)]
            decisions={}
            for m in ledgers:
                if segment!=last_segment:previous[m]=[]
                scores=cosine_matrix(descriptors[m],previous[m]);matches=mutual_unique_matches(scores)
                prefix=ledgers[m].snapshot()['history'];event=ledgers[m].update(segment=segment,order=i,candidates=candidates,previous_matches=matches)
                assert ledgers[m].history[:-1]==prefix
                decisions[m]=dict(scores=[[None if np.isnan(x) else float(x) for x in row] for row in scores],event=event)
                previous[m]=descriptors[m]
            last_segment=segment
            row=dict(index=i,segment=segment,source=e,pose=pose.tolist(),proposals=proposals,decisions=decisions)
            predictions.append(row);write(out/f'artifacts/prediction_{i:02d}.json',row)
        for m,g in ledgers.items():write(out/f'artifacts/{m}_ledger.json',g.snapshot())
        prediction_hashes={str(p.relative_to(out)):sha(p) for p in (out/'artifacts').glob('*') if p.suffix=='.json'}
        write(out/'artifacts/prediction_freeze.json',prediction_hashes)
        with (out/'logs/observations.jsonl').open('x') as log:
            for i,row in enumerate(predictions):
                e=row['source'];task=e['source']['task'];pose=np.asarray(row['pose']);origin=pose[:3,3]
                construction=json.loads((ROOT/SOURCE/f'artifacts/source_evidence/{task}_constructions.json').read_text())
                sections,unavailable=construction_sections(construction,origin)
                with np.load(ROOT/SOURCE/e['student_path'],allow_pickle=False) as data:
                    ranges=data['ranges_m'][-1].reshape(-1);valid=data['valid_mask'][-1].astype(bool).reshape(-1)
                endpoints=origin+(unit*ranges[:,None])@pose[:3,:3].T
                # Invalid rays have zero length, so cannot witness a crossing.
                endpoints[~valid]=origin
                hits=crossed_sections(np.broadcast_to(origin,endpoints.shape),endpoints,sections)
                witnesses=[candidate_witnesses([int(x.rsplit('/ray:',1)[1]) for x in p['source_refs']],hits,sections) for p in row['proposals']]
                supported=[section['port_reference'] for j,section in enumerate(sections) if hits[:,j].any()]
                ref=dict(segment=row['segment'],order=i,candidate_labels={j:w['conditional_port_reference'] for j,w in enumerate(witnesses)},
                    observable_tasks=supported,evidence_ref=f'partial finite-ray sections in witness_{i:02d}.json')
                references.append(ref)
                evidence=dict(reference=ref,sections=sections,unavailable_sections=unavailable,candidates=witnesses,
                    supporting_ray_indices={sec['port_reference']:np.flatnonzero(hits[:,j]).tolist() for j,sec in enumerate(sections)},
                    training_qualified=False,physical_traversability_qualified=False,hidden_connections_used=False)
                write(out/f'artifacts/witness_{i:02d}.json',evidence)
                counts=dict(index=i,sections=len(sections),witnessed_sections=len(supported),candidates=len(witnesses),
                    uniquely_referenced=sum(w['conditional_port_reference'] is not None for w in witnesses),unknown=sum(w['conditional_port_reference'] is None for w in witnesses))
                log.write(json.dumps(counts)+'\n');log.flush();print(json.dumps(counts),flush=True)
            summary['conditional_scores']={m:score_task_history(g.history,references)['totals'] for m,g in ledgers.items()}
            summary['tasks_created']={m:len(g.tasks) for m,g in ledgers.items()}
        for p,h in prediction_hashes.items():assert sha(out/p)==h,'prediction changed after teacher read'
        summary['candidate_count']=sum(len(r['proposals']) for r in predictions)
        if CONTROL_RUN:
            summary['control_summary']=json.loads((ROOT/CONTROL_RUN/'metrics/summary.json').read_text())
            summary['parent_candidate_count']=sum(len({p['parent_candidate'] for p in row['proposals']}) for row in predictions)
            assert summary['parent_candidate_count']==19,'parent population changed'
            summary['observed_decomposition_evidence']=[dict(observation=i,parents=[
                next(p['parent_decomposition_evidence'] for p in row['proposals'] if p['parent_candidate']==j)
                for j in sorted({p['parent_candidate'] for p in row['proposals']})]) for i,row in enumerate(predictions)]
        summary['covered_candidate_count']=sum(v is not None for r in references for v in r['candidate_labels'].values())
        summary['abc_matching_difference_observations']=sum(row['decisions']['B']['event']['assignments']!=row['decisions']['C']['event']['assignments'] for row in predictions)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        fig,axes=plt.subplots(3,4,figsize=(16,12))
        for i,row in enumerate(predictions):
            ax=axes.ravel()[i];pose=np.asarray(row['pose'])
            with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as data:
                points=data['registered_returns_xyz_m'][data['surface_return_indices']][::5]@pose[:3,:3].T
            ax.scatter(points[:,0],points[:,1],s=1,c='silver')
            ref=json.loads((out/f'artifacts/witness_{i:02d}.json').read_text())
            for j,p in enumerate(row['proposals']):
                delta=np.asarray(p['axis_target_xyz_m'])-pose[:3,3]
                known=ref['candidates'][j]['conditional_port_reference'] is not None
                ax.annotate(str(j),xy=delta[:2],xytext=(0,0),arrowprops=dict(arrowstyle='->',color='green' if known else 'darkorange'),fontsize=8)
            for sec in ref['sections']:
                delta=np.asarray(sec['center_world_m'])-pose[:3,3]
                ax.scatter(delta[0],delta[1],marker='s',s=20,facecolors='none',edgecolors='blue')
            n=sum(x['conditional_port_reference'] is not None for x in ref['candidates'])
            ax.set_title(f"片段{row['segment']} 观察{i}：{n}/{len(row['proposals'])}有唯一截面证据",fontproperties=font,fontsize=9)
            ax.set_xlim(-11,11);ax.set_ylim(-11,11);ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle('全部12观察：灰色回波；绿色候选有条件性射线证据，橙色未知；蓝框为参考截面中心\n俯视图不表示可通行；证据不是完整通道标签，不能作独立方法胜负结论',fontproperties=font,fontsize=12)
        fig.tight_layout();fig.savefig(out/'previews/ray_section_witnesses.png',dpi=130);plt.close(fig)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=s['limits']['host_bytes']
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=s['limits']['output_bytes']
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary.update(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,observations_completed=len(references),elapsed_s=time.monotonic()-start,
        training_steps=0,model_forwards=0,training_qualified=False,independent_task_accuracy_qualified=False,
        evidence_scope='partial construction-conditioned ray section correspondence ONLY',method_advantage_proven=False)
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
