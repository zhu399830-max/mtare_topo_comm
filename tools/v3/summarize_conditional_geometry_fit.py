"""Read sealed initial/final predictions; create a report, never alter the run."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json,statistics
from pathlib import Path
RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_new12_conditional_fit_v1_seed0'
TARGET=ROOT/'results/gate3_semantics/gate3_20260911_gse_new12_conditional_geometry_v1_seed0'
OUT=ROOT/'docs/figures/gse_conditional_geometry_fit_v1'


def main(variant_runs=None, output_dir=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folders=variant_runs or {v:RUN for v in 'ABC'}
    out=Path(output_dir) if output_dir is not None else OUT
    for base in (*set(folders.values()),TARGET):
        for line in (base/'artifacts/evidence_sha256.txt').read_text().splitlines():
            h,p=line.split('  ',1)
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('seal drift '+p)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.png').exists():raise ValueError('report already exists; no overwrite')
    stats={};pred={v:{'axis':[],'height':[]} for v in 'ABC'};truth={'axis':[],'height':[]};rows={}
    for v in 'ABC':
        rows[v]=json.loads((folders[v]/f'metrics/{v}_step2000.json').read_text())
        stats[v]={}
        for step in (0,2000):
            data=json.loads((folders[v]/f'metrics/{v}_step{step}.json').read_text())
            cross=[x['metrics']['cross'] for x in data if x['metrics']['cross']['count']]
            stats[v][str(step)]={k:statistics.mean(x[k] for x in cross) for k in ('axis_mae','height_mae_m','constant_axis_mae','constant_height_mae_m')}
    for i in range(12):
        with np.load(TARGET/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as t:
            mask=t['axis_known']&~t['same_reference_component']
            truth['axis'].extend(t['axis_abs_dot'][mask]);truth['height'].extend(t['height_difference_m'][mask])
            for v in 'ABC':
                with np.load(folders[v]/f'artifacts/{v}_step2000_{i:02d}.npz',allow_pickle=False) as p:
                    assert np.array_equal(p['neighbors'][0],t['neighbor_index'])
                    pred[v]['axis'].extend(p['axis'][0][mask]);pred[v]['height'].extend(p['height'][0][mask])
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors={'A':'#7f8c8d','B':'#e67e22','C':'#2471a3'}
    for v in 'ABC':
        ax[0,0].scatter(truth['axis'],pred[v]['axis'],s=10,alpha=.4,label=v,color=colors[v])
        ax[0,1].scatter(truth['height'],pred[v]['height'],s=10,alpha=.4,label=v,color=colors[v])
    axis_title='轴向关系：三组拟合对比' if variant_runs else '轴向关系：预测几乎全为1'
    for panel,key,title in ((ax[0,0],'axis',axis_title),(ax[0,1],'height','参考中心高差：三组拟合对比')):
        lo=min(truth[key]);hi=max(truth[key]);panel.plot([lo,hi],[lo,hi],'k--',lw=1,label='理想预测')
        panel.set(xlabel='构造条件参考（高差单位：米）',ylabel='模型预测',title=title);panel.legend()
    ids=[x['observation'] for x in rows['C'] if x['metrics']['cross']['count']]
    x=np.arange(len(ids))
    for j,v in enumerate('ABC'):
        values=[rows[v][i]['metrics']['cross']['height_mae_m'] for i in ids]
        ax[1,0].bar(x+(j-1)*.24,values,width=.24,label=v,color=colors[v])
    ax[1,0].set(xticks=x,xticklabels=ids,xlabel='固定观察编号（1、9无跨片段目标）',ylabel='高差平均绝对误差（米）',title='逐父地图结果：改善与退步均保留');ax[1,0].legend()
    labels=['常数基线','A：观测','B：加面片','C：加关系']
    heights=[stats['A']['2000']['constant_height_mae_m']]+[stats[v]['2000']['height_mae_m'] for v in 'ABC']
    ax[1,1].bar(labels,heights,color=['#bdc3c7']+list(colors.values()))
    ax[1,1].set(ylabel='跨片段高差误差（米）',title='父地图等权平均；同一对内重复项取平均')
    for i,v in enumerate(heights):ax[1,1].text(i,v,f'{v:.4f}',ha='center',va='bottom')
    fig.suptitle('构造条件几何拟合：三组各2000次更新\n仅12例训练集；409项跨片段预测不是409个独立地点',fontsize=15)
    fig.savefig(out/'results.png',dpi=160);plt.close(fig)
    evidence=dict(runs={v:str(p.relative_to(ROOT)) for v,p in folders.items()},stats=stats,scope='fit only;10 parents with cross pairs,23 distinct component pairs',
        scatter_scope='known cross-component evaluation subset; inference outputs were not filtered',
        c_axis_prediction_range=[float(min(pred['C']['axis'])),float(max(pred['C']['axis']))],
        c_height_prediction_range=[float(min(pred['C']['height'])),float(max(pred['C']['height']))])
    (out/'summary.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':main()
