"""One explicit readout correction on saved model tokens; no model rerun."""
from _bootstrap import PROJECT_ROOT as ROOT
from pathlib import Path
import argparse,hashlib,json,resource,sys,time,traceback,zipfile
from run_short_observation_chain import sha,write
CODE=Path(__file__).resolve().parents[2]
NAME='gse_directional_context_readout_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260912_{NAME}_seed0'
SOURCE='results/gate3_semantics/gate3_20260912_gse_structural_token_chain_v1_seed0'
RAW='results/gate3_semantics/gate3_20260912_gse_short_observation_chain_v1_seed0'

def freeze():
    pins={}
    for base in (SOURCE,RAW):
        seal=ROOT/base/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        for line in seal.read_text().splitlines():
            h,p=line.split('  ',1)
            if '/artifacts/' in p and (p.endswith('.npz') or p.endswith('/graph.json') or '/window_' in p):
                if sha(ROOT/p)!=h:raise ValueError('source drift')
                pins[p]=h
    s=dict(source=SOURCE,raw=RAW,frames=12,windows=8,variants=list('ABC'),candidates_per_variant=32,
        training_steps=0,model_forwards=0,teacher_reads=0,
        readout='All observed patch centers in direction-positive hemisphere; positive cosine weighted local token, normalized. No new radius/angle threshold.',
        claims='Directional context and unverified retrieval only, never patch/channel membership or task identity',
        limits=dict(host_bytes=4*1024**3,wall_seconds=300,output_bytes=256*1024**2))
    a=dict(status='APPROVED',approved_by='user-standing-new-method-integration',approved_at='2026-09-12',authorized_gates=[3],authorized_operations=['data_export'],
        scope='One saved-token readout correction after demonstrated missing direct-ray support; same population, no training or labels',
        confirmation_reference='User requests actual new-method execution and standing evidence-based implementation; direct-ray problem reported before correction',
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version=NAME,scope=s,approval=a));pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(CODE)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (CODE/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260912',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__)),'--execute'],
        question='Do saved learned local tokens provide directional context when direct open-ray endpoints lie outside the patch domain?',
        method=s['readout'],baseline='Identical A/B/C tokens and candidate directions; old direct-ray result preserved',
        fallback='Preserve undefined contexts; no feature resampling, training, hidden truth or merge',
        estimated_cost=dict(compute='CPU saved-output processing only',host_ram_gb=4,disk_gb=.25,wall_time_hours=1/12),
        acceptance_criteria=['All32candidate contexts per variant retained','All preceding-window retrieval scores stored','No new predictions/weights/qualified correspondence claims'],
        expected_evidence=['Context supports/weights, descriptors, retrieval matrices, task-attributed graph, visual comparison, raw log, source snapshot and seal'],
        input_sha256=pins,execution_code_root=str(CODE),execution_source_sha256=code))
    print(SPEC)

def execute():
    import numpy as np
    from mtare_topo.topology.structural_task_tokens import bind_directional_context,retrieval_candidates
    from mtare_topo.governance_context_readout import validate_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert validate_card(card).passed and json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for p,h in spec['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('input drift')
    for p,h in spec['execution_source_sha256'].items():
        if sha(CODE/p)!=h:raise ValueError('code drift')
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;history={m:[] for m in 'ABC'}
    try:
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(CODE/p,p)
        poses=np.load(ROOT/RAW/'artifacts/inputs.npz',allow_pickle=False)['poses'];prev={m:[] for m in 'ABC'}
        with (out/'logs/windows.jsonl').open('x') as log:
            for j in range(8):
                r=json.loads((ROOT/RAW/f'artifacts/window_{j:02d}.json').read_text())
                c=np.load(ROOT/SOURCE/f'artifacts/common_{j:02d}.npz',allow_pickle=False)['patch_centers']
                for m in 'ABC':
                    t=np.load(ROOT/SOURCE/f'artifacts/{m}_tokens_{j:02d}.npz',allow_pickle=False)['local_tokens']
                    tasks=bind_directional_context(r['method_proposals']['range_sectors'],c,t,poses[j+4])
                    match=retrieval_candidates(prev[m],tasks);prev[m]=tasks
                    row=dict(window=j,order=r['frame_row'],metric_anchor=r['decision']['node'],tasks=tasks,retrieval=match)
                    history[m].append(row);write(out/f'artifacts/{m}_tasks_{j:02d}.json',row)
                    log.write(json.dumps(dict(window=j,variant=m,contexts=sum(t['descriptor'] is not None for t in tasks)))+'\n')
                log.flush()
                if time.monotonic()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise RuntimeError('resource cap')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise RuntimeError('output cap')
        graph=json.loads((ROOT/RAW/'artifacts/graph.json').read_text())
        for m in 'ABC':write(out/f'artifacts/{m}_task_graph.json',dict(metric_graph=graph,task_context_history=history[m],confirmed_merges=0,closed_loop=False))
        # Fixed last transition; raw similarity is not a probability or score of correctness.
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,m,title in zip(axs,'ABC',['A：观测特征','B：增加面片几何','C：增加面片关系']):
            mat=np.full((4,4),np.nan)
            for r in history[m][-1]['retrieval']:
                for v in r['scores']:mat[r['current'],v['previous']]=v['cosine']
            ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(title,fontproperties=font)
            for i in range(4):
                for k in range(4):ax.text(k,i,f'{mat[i,k]:.3f}',ha='center',va='center',fontsize=9)
            ax.set_xlabel('上一窗口候选编号',fontproperties=font);ax.set_ylabel('当前候选编号',fontproperties=font)
        fig.suptitle('新模型已进入方向检索：数值为余弦相似度，不是连接概率或正确率',fontproperties=font)
        fig.tight_layout();fig.savefig(out/'artifacts/retrieval.png',dpi=160);plt.close(fig)
    except BaseException:error=traceback.format_exc()
    counts={m:dict(candidates=sum(len(r['tasks']) for r in history[m]),with_context=sum(t['descriptor'] is not None for r in history[m] for t in r['tasks']),
        retrieval_scores=sum(len(t['scores']) for r in history[m] for t in r['retrieval']),confirmed_merges=0) for m in 'ABC'}
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,counts=counts,training_steps=0,model_forwards=0,
        elapsed_s=time.monotonic()-start,graph_advantage_proven=False,meaning='Explicit geometry-context readout, not learned task identity')
    write(out/'metrics/summary.json',result);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
