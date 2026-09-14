"""Report new A versus reused B from sealed metrics, without rerunning old audits."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import hashlib
from pathlib import Path
import math
from mtare_topo.governance_surface_selection import digest

A=ROOT/'results/gate3_semantics/gate3_20260910_gse_geometry_match_presence_a_v1_seed0'
B=ROOT/'results/gate3_semantics/gate3_20260909_gse_grouping_center_supervision_v2_seed0'
S=ROOT/'results/gate3_semantics/gate3_20260909_gse_spatial_center_fit_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph/geometry_presence_a_20260910'


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(run):
    seal=run/'artifacts/evidence_sha256.txt';files=set()
    for line in seal.read_text().splitlines():
        h,p=line.split('  ',1);p=ROOT/p
        if sha(p)!=h:raise ValueError('sealed drift '+str(p))
        files.add(p)
    if files!={p for p in run.rglob('*') if p.is_file() and p!=seal}:raise ValueError('seal inventory changed')
    return dict(files=len(files),seal_sha256=sha(seal))


def main():
    if Path(str(OUT)+'.json').exists() or Path(str(OUT)+'.png').exists():raise FileExistsError('do not overwrite')
    seals={'A':verify(A),'B':verify(B)}
    # Reuse already sealed B localization reduction; do not recompute full-query audits.
    expected='c519999eb14afcbaecb808e8df57c5f77b921f7a60fef2098756605811a55765'
    seal=S/'artifacts/evidence_sha256.txt'
    if sha(seal)!=expected:raise ValueError('old localization seal changed')
    idx={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    path=S/'metrics/paired_fit_diagnostic.json'
    if sha(path)!=idx[str(path.relative_to(ROOT))]:raise ValueError('old B localization changed')
    b_funnel=load(path)['primitive'];by_step={r['step']:r for r in b_funnel}
    curves=[]
    for step in range(0,1001,100):
        a=load(A/'metrics'/f'evaluation_{step:04d}.json');b=load(B/'metrics'/f'evaluation_{step:04d}.json')
        loc=load(A/'metrics'/f'localization_{step:04d}.json')
        for x,y in zip(a['observations'],b['observations']):assert x['source']==y['source']
        curves.append(dict(step=step,A=a['summary'],B=b['summary'],A_localization=loc['summary'],B_localization=by_step[step]['summary']))
    case=[];last_a=load(A/'metrics/evaluation_1000.json');last_b=load(B/'metrics/evaluation_1000.json')
    aloc={digest(o['source']):o for o in load(A/'metrics/localization_1000.json')['observations']}
    bloc={digest(o['source']):o['funnel'] for o in by_step[1000]['observations']}
    radii={}
    for radius in ('0.5','1.0','2.0','4.0'):
        radii[radius]={}
        for name,data in [('A',last_a),('B',last_b)]:
            counts={k:sum(o['scores'][radius][k] for o in data['observations']) for k in ('tp','fp','fn','ignored')}
            tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
            counts.update(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0)
            radii[radius][name]=counts
    for a,b in zip(last_a['observations'],last_b['observations']):
        key=digest(a['source']);case.append(dict(source=a['source'],A={k:a['scores']['1.0'][k] for k in ('tp','fp','fn','ignored')},
            B={k:b['scores']['1.0'][k] for k in ('tp','fp','fn','ignored')},A_localization=aloc[key],B_localization=bloc[key]))
    logs=[json.loads(s) for s in (A/'logs/updates.jsonl').read_text().splitlines()]
    assert len(logs)==1000 and all(math.isfinite(v) for r in logs for v in r['gradient_norms'].values())
    assert not any('head.head.branch' in n for r in logs for n in r['gradient_norms'])
    summary=load(A/'metrics/summary.json');assert load(A/'RUN_STATE.json')['state']=='COMPLETED'
    expected_pass=summary['final']['precision']>=.9 and summary['final']['recall']>=.9 and summary['optimizer_steps']==1000
    assert (summary['status']=='FIT_PASS')==expected_pass
    protocol=dict(A_full_fit_passed=expected_pass,retain_A_for_fair_validation=expected_pass,
        selected_spec='configs/v3/gate3/gse_geometry_match_presence_a_v1.json' if expected_pass else None,
        further_unique_root_cause_search_required=False if expected_pass else None,
        new_fair_training_started=False,branch_training=False,graph_started=False,
        next='prepare fair validation with retained A protocol; no more root-cause detours' if expected_pass else 'full fit failed; report specific localization/detection gap, no unapproved expansion')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for name,color,label in [('A','#0072b2','A：匹配不使用存在分数'),('B','#d55e00','B：旧完整运行（复用）')]:
        x=[r['step'] for r in curves]
        axes[0].plot(x,[r[name+'_localization']['raw_one_to_one_tp'] for r in curves],'-o',color=color,label=label+' 定位覆盖')
        axes[0].plot(x,[r[name]['tp'] for r in curves],'--',color=color,label=label+' 实际检出')
        axes[1].plot(x,[r[name]['precision'] for r in curves],'-o',color=color,label=label+' 精确率')
        axes[1].plot(x,[r[name]['recall'] for r in curves],'--',color=color,label=label+' 召回率')
        axes[2].plot(x,[r[name]['fp'] for r in curves],'-o',color=color,label=label+' 已知误检')
        axes[2].plot(x,[r[name]['ignored'] for r in curves],'--',color=color,label=label+' 未知预测')
    axes[0].set(title='位置够近 ≠ 完整检测正确',ylim=(-.5,12.5))
    axes[1].set(title='完整中心检测：原1米／0.5阈值',ylim=(-.05,1.05));axes[1].axhline(.9,color='gray',ls=':')
    axes[2].set_title('已知误检与无法确认预测分别列出')
    for ax in axes:ax.set_xlabel('原固定批次');ax.grid(alpha=.2);ax.legend(prop=FontProperties(fname=font,size=7))
    fig.suptitle('同16观察、同step0、同修复存在性损失：只改变匹配成本；不是泛化成绩')
    for t in fig.findobj(Text):t.set_fontproperties(FontProperties(fname=font,size=t.get_fontsize()))
    fig.tight_layout();fig.savefig(str(OUT)+'.png',dpi=160);plt.close(fig)
    result=dict(seals=seals,summary=summary,equivalence=load(A/'metrics/b_equivalence.json'),curves=curves,
        cases=case,all_radii=radii,protocol_decision=protocol,B_new_updates=0,decoder_reruns=0,pure_position_reruns=0,
        old_loss_audits_rerun=0,figure_sha256=sha(Path(str(OUT)+'.png')),tool_sha256=sha(Path(__file__)))
    with Path(str(OUT)+'.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(seals=seals,summary=summary,protocol=protocol,all_radii=radii,
        cases=[dict(source=c['source'],A=c['A'],B=c['B'],A_refs=c['A_localization']['references']) for c in case]),ensure_ascii=False))


if __name__=='__main__':main()
