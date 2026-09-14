"""Read sealed position-only evidence; render all12 assigned references, no query rescore."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
from pathlib import Path
import numpy as np
from mtare_topo.governance_surface_selection import digest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(run):
    seal=run/'artifacts/evidence_sha256.txt';paths=set()
    for row in seal.read_text().splitlines():
        h,p=row.split('  ',1);f=ROOT/p
        if sha(f)!=h:raise ValueError('sealed drift '+p)
        paths.add(f)
    if paths!={f for f in run.rglob('*') if f.is_file() and f!=seal}:raise ValueError('seal membership')
    return dict(files=len(paths),seal_sha256=sha(seal),bytes=sum(f.stat().st_size for f in paths))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    stem=ROOT/'docs/figures/gse_graph/position_only_diagnostic_20260909'
    if any(Path(str(stem)+s).exists() for s in ('.json','.png','_cases.png')):raise FileExistsError('no overwrite')
    records={};runs={}
    for mode in ('fixed','dynamic'):
        run=ROOT/('results/gate3_semantics/gate3_20260909_gse_primitive_position_'+mode+'_v1r_seed0');runs[mode]=run
        checked=verify(run)
        def read(name):return json.loads((run/name).read_text())
        if read('RUN_STATE.json')['state']!='COMPLETED':raise ValueError('not completed')
        logs=[json.loads(s) for s in (run/'logs/updates.jsonl').read_text().splitlines()]
        assert len(logs)==1000 and all(r['existence_gradient']==r['branch_gradient']==0 for r in logs)
        changes={};last={}
        for row in logs:
            for o in row['observations']:
                key=digest(o['source']);a=o['assignment']
                changes[key]=changes.get(key,0)+int(key in last and last[key]!=a);last[key]=a
        evaluations=[read(f'metrics/evaluation_{s:04d}.json') for s in range(0,1001,100)]
        # Verify only the assigned reference coordinates; do not rerun full queries/scoring.
        for e in evaluations:
            for row in e['observations']:
                key=digest(row['source'])
                with np.load(run/'artifacts'/f'input_{key}.npz') as data:t=data['target_positions_m']
                with np.load(run/'artifacts'/f'prediction_{e["summary"]["step"]:04d}_{key}.npz') as data:p=data['position_m'][row['assignment']]
                assert np.allclose(np.linalg.norm(p-t,axis=-1),row['error_m'],atol=1e-6)
        records[mode]=dict(seal=checked,summary=read('metrics/summary.json'),evaluations=evaluations,
            initialization=read('metrics/input_initialization_check.json'),
            inactive_batches=[r['step'] for r in logs if not r['active_observations']],
            supervised_visits=sum(r['active_observations'] for r in logs),
            training_assignment_changes=changes,total_training_assignment_changes=sum(changes.values()))
    assert records['fixed']['initialization']['initial_sha256']==records['dynamic']['initialization']['initial_sha256']
    assert (runs['fixed']/'config/schedule.json').read_bytes()==(runs['dynamic']/'config/schedule.json').read_bytes()
    per_case=[]
    for i,row in enumerate(records['fixed']['evaluations'][0]['observations']):
        key=digest(row['source'])
        if not row['error_m']:continue
        entry=dict(source=row['source'],initial_error_m=row['error_m'][0])
        for mode in records:
            series=[e['observations'][i] for e in records[mode]['evaluations']]
            assert all(s['source']==row['source'] for s in series)
            entry[mode]=dict(errors_m=[s['error_m'][0] for s in series],assignments=[s['assignment'][0] for s in series],
                training_assignment_changes=records[mode]['training_assignment_changes'][key])
        per_case.append(entry)
    payload=dict(runs=records,cases=per_case,interpretation='position-only assigned references, not detection score or method win',
        old_all_query_reaudits=0,old_supervision_audits=0,tool_sha256=sha(Path(__file__)))
    with Path(str(stem)+'.json').open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for mode,color,label in [('fixed','#1671ba','固定对应，仅位置'),('dynamic','#d97706','动态对应，仅位置')]:
        h=records[mode]['summary']['history'];x=[r['step'] for r in h]
        axes[0].plot(x,[r['mean_m'] for r in h],'-o',color=color,label=label)
        axes[1].plot(x,[r['within1m'] for r in h],'-o',color=color,label=label)
    axes[0].set(ylabel='12个参考的平均位置误差（米）',xlabel='固定排程批次',yscale='log')
    axes[1].set(ylabel='一对一对应落入1米内的参考数',xlabel='固定排程批次',ylim=(-.5,12.8))
    axes[1].axhline(11,color='gray',ls='--',label='预登记继续线：至少11/12')
    for ax in axes:ax.grid(alpha=.2);ax.legend(prop=font)
    fig.suptitle('同16观察／同初态：位置回归隔离诊断，不是检测优胜成绩')
    for text in fig.findobj(Text):text.set_fontproperties(font)
    fig.tight_layout();fig.savefig(str(stem)+'.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,12))
    for ax,case in zip(axes.flat,per_case):
        key=digest(case['source'])
        with np.load(runs['fixed']/'artifacts'/f'input_{key}.npz') as z:xyz=z['xyz_m'];t=z['target_positions_m'][0]
        with np.load(runs['fixed']/'artifacts'/f'prediction_0000_{key}.npz') as z:q=z['position_m'][case['fixed']['assignments'][0]]
        # Identical identity-order visual decimation only; training keeps full inputs.
        xyz=xyz[::max(1,len(xyz)//2000)]
        ax.scatter(xyz[:,0],xyz[:,1],s=1,color='#b3bec7',alpha=.45)
        ax.scatter(*t[:2],s=95,marker='*',color='#111827',label='参考中心')
        ax.scatter(*q[:2],s=32,marker='s',facecolors='none',edgecolors='gray',label='初始指定查询')
        for mode,color,mark in [('fixed','#1671ba','o'),('dynamic','#d97706','x')]:
            with np.load(runs[mode]/'artifacts'/f'prediction_1000_{key}.npz') as z:p=z['position_m'][case[mode]['assignments'][-1]]
            ax.scatter(*p[:2],s=40,marker=mark,color=color,label='固定对应' if mode=='fixed' else '动态对应')
            ax.plot([t[0],p[0]],[t[1],p[1]],color=color,alpha=.7)
        s=case['source'];name=s['task'].split('__');variant={'ellipse':'椭圆','rounded_rectangle':'圆角矩形','c1_mixed':'混合'}.get(name[1],name[1])
        ax.set_title(f'{name[0][:3]} {variant} / {s["source_sequence_id"]}\n三维误差：固定{case["fixed"]["errors_m"][-1]:.3f}m；动态{case["dynamic"]["errors_m"][-1]:.3f}m')
        ax.set_aspect('equal');ax.set_xlim(-11,11);ax.set_ylim(-11,11);ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,prop=font)
    fig.suptitle('全部12个参考（固定身份顺序）：XY展示，误差按三维计算；仅画被监督对应，不代表全部预测')
    for text in fig.findobj(Text):text.set_fontproperties(font)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(str(stem)+'_cases.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(seals={k:v['seal'] for k,v in records.items()},cases=per_case,
        inactive_batches={k:v['inactive_batches'] for k,v in records.items()},
        assignment_changes={k:v['total_training_assignment_changes'] for k,v in records.items()}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
