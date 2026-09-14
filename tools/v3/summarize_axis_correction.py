"""Compare sealed C before/after the one objective correction, not methods."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json,statistics


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    base=ROOT/'results/gate3_semantics'
    old=base/'gate3_20260911_gse_new12_conditional_fit_v1_seed0'
    new=base/'gate3_20260911_gse_new12_axis_logit_correction_v1_seed0'
    target=base/'gate3_20260911_gse_new12_conditional_geometry_v1_seed0'
    out=ROOT/'docs/figures/gse_conditional_geometry_fit_v1'
    if (out/'axis_correction.png').exists():raise ValueError('existing derived figure, no overwrite')
    for folder in (old,new,target):
        for line in (folder/'artifacts/evidence_sha256.txt').read_text().splitlines():
            h,p=line.split('  ',1);assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    rows={k:json.loads((r/'metrics/C_step2000.json').read_text()) for k,r in [('old',old),('corrected',new)]}
    stats={}
    for k,data in rows.items():
        cross=[r['metrics']['cross'] for r in data if r['metrics']['cross']['count']]
        stats[k]={f:statistics.mean(r[f] for r in cross) for f in ('axis_mae','height_mae_m')}
    truth=[[],[]];pred={k:[[],[]] for k in rows}
    for i in range(12):
        with np.load(target/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as t:
            mask=t['axis_known']&~t['same_reference_component']
            for j,field in enumerate(('axis_abs_dot','height_difference_m')):truth[j].extend(t[field][mask])
            for key,folder in [('old',old),('corrected',new)]:
                with np.load(folder/f'artifacts/C_step2000_{i:02d}.npz',allow_pickle=False) as p:
                    assert np.array_equal(p['neighbors'][0],t['neighbor_index'])
                    for j,field in enumerate(('axis','height')):pred[key][j].extend(p[field][0][mask])
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for j,name in enumerate(('轴向相似度（1同轴、0垂直）','参考中心高差（米）')):
        axes[0,j].scatter(truth[j],pred['old'][j],s=12,alpha=.4,label='原C',color='#999999')
        axes[0,j].scatter(truth[j],pred['corrected'][j],s=12,alpha=.5,label='修正C',color='#2471a3')
        lo=min(truth[j]);hi=max(truth[j]);axes[0,j].plot([lo,hi],[lo,hi],'k--',lw=1)
        axes[0,j].set(xlabel='构造条件参考',ylabel='预测值',title=name);axes[0,j].legend()
        ids=[r['observation'] for r in rows['old'] if r['metrics']['cross']['count']];x=np.arange(len(ids))
        metric=('axis_mae','height_mae_m')[j]
        for offset,key,label,color in [(-.18,'old','原C','#999999'),(.18,'corrected','修正C','#2471a3')]:
            axes[1,j].bar(x+offset,[rows[key][i]['metrics']['cross'][metric] for i in ids],width=.36,label=label,color=color)
        axes[1,j].set(xticks=x,xticklabels=ids,xlabel='固定观察编号（每例不同父地图）',ylabel='平均绝对误差',title='逐例跨片段误差');axes[1,j].legend()
    fig.suptitle('同一个C模型：仅改变轴向损失，初态／样本顺序／2000更新预算不变\n仅训练集拟合验证；不能作为几何方法优于A/B的证据',fontsize=14)
    fig.savefig(out/'axis_correction.png',dpi=160);plt.close(fig)
    report=dict(scope='same12 fit only; corrected C versus original C, not geometry superiority',stats=stats,
        corrected_prediction_ranges={k:[float(min(pred['corrected'][j])),float(max(pred['corrected'][j]))] for j,k in enumerate(('axis','height'))},
        original_initial_tensors_and_schedule_verified_equal=True,initial_predictions_all12_verified_equal=True)
    with (out/'axis_correction_summary.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
