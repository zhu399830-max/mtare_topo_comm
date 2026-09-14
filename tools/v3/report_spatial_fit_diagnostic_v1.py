"""Summarize the completed SPATIAL fit versus reused PRIMITIVE predictions.

Descriptive reduction only: no models, optimizer, teacher or loss-mask audit.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

RUN=ROOT/'results/gate3_semantics/gate3_20260909_gse_spatial_center_fit_v1_seed0'
OUT=ROOT/'docs/figures/gse_graph'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def identity(source):return json.dumps(source,sort_keys=True)


def main():
    figure=OUT/'spatial_vs_primitive_fit_20260909.png'
    report=OUT/'spatial_vs_primitive_fit_20260909.json'
    if report.exists() or figure.exists():raise FileExistsError('report outputs already exist')
    if load(RUN/'RUN_STATE.json')['state']!='COMPLETED':raise ValueError('fit not complete')
    seal=RUN/'artifacts/evidence_sha256.txt';files=set()
    for line in seal.read_text().splitlines():
        h,p=line.split('  ',1);path=ROOT/p
        if sha(path)!=h:raise ValueError('new evidence drift')
        files.add(path.resolve())
    if files!={p.resolve() for p in RUN.rglob('*') if p.is_file() and p!=seal}:raise ValueError('new evidence inventory drift')
    pair=load(RUN/'metrics/paired_fit_diagnostic.json')
    summary=load(RUN/'metrics/summary.json')
    logs=[json.loads(s) for s in (RUN/'logs/updates.jsonl').read_text().splitlines()]
    assert len(logs)==1000 and logs[-1]['optimizer_steps']==summary['optimizer_steps']==1000
    assert all(math.isfinite(v) for r in logs for v in r['gradient_norms'].values())
    assert not any('head.head.branch' in key for r in logs for key in r['gradient_norms'])
    curves={}
    for method in ('primitive','spatial'):
        curves[method]=[]
        for s in pair[method]:
            row=dict(s['summary'],step=s['step'],actual=s['actual_summary'])
            row['mean_nearest_distance_m']=statistics.mean(t['nearest_distance_m'] for o in s['observations'] for t in o['funnel']['references'])
            curves[method].append(row)
    by_source={identity(o['source']):o for o in pair['spatial'][-1]['observations']}
    cases=[];population=[]
    for o in pair['primitive'][-1]['observations']:
        other=by_source[identity(o['source'])]
        population.append(dict(source=o['source'],primitive=o['funnel'],spatial=other['funnel'],
            primitive_actual={k:o['actual'][k] for k in ('tp','fp','fn','ignored')},
            spatial_actual={k:other['actual'][k] for k in ('tp','fp','fn','ignored')}))
        for t in o['funnel']['references']:
            if not t['matched_actual']:
                cases.append(dict(source=o['source'],primitive=t,spatial=other['funnel']['references'][t['reference_index']]))
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for method,label,color in [('primitive','几何分块','#0072b2'),('spatial','空间分组','#d55e00')]:
        rows=curves[method];x=[r['step'] for r in rows]
        axes[0].plot(x,[r['raw_one_to_one_tp'] for r in rows],'-o',color=color,label=label+'：定位上限')
        axes[0].plot(x,[r['actual_tp'] for r in rows],'--',color=color,label=label+'：实际正确数')
        axes[1].plot(x,[r['actual']['fp'] for r in rows],'-o',color=color,label=label+'：已知误检')
        axes[1].plot(x,[r['actual']['ignored'] for r in rows],'--',color=color,label=label+'：未知预测')
        axes[2].plot(x,[r['mean_nearest_distance_m'] for r in rows],'-o',color=color,label=label)
    axes[0].set_title('一对一定位上限与实际召回（最多12）');axes[0].set_ylim(-.5,12.5)
    axes[1].set_title('已知误检与未知预测分开统计')
    axes[2].set_title('全部12参考的最近预测平均误差');axes[2].set_ylabel('米')
    for ax in axes:ax.set_xlabel('固定更新次数');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('同16观察、同检测器及修复监督：只补空间分组，不作方法胜负结论',fontsize=14)
    for text in fig.findobj(Text):text.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=text.get_fontsize()))
    fig.tight_layout();fig.savefig(figure,dpi=170);plt.close(fig)
    result=dict(new_run=str(RUN.relative_to(ROOT)),sealed_files=len(files),seal_sha256=sha(seal),
        curves=curves,previous_four_misses=cases,all_sixteen=population,spatial_summary=summary,
        additional_primitive_updates=0,spatial_updates=1000,branch_gradients=0,
        old_mask_gradient_reaudits=0,method_winner_claim=False,
        script_sha256=sha(Path(__file__)),figure_sha256=sha(figure))
    with report.open('x') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(sealed_files=len(files),seal_sha256=sha(seal),figure=str(figure),report=str(report))))


if __name__=='__main__':main()
