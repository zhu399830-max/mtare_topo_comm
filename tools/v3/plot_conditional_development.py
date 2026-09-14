"""Derived full-parent comparison; no cherry-picked predictions or inference."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statistics import mean
from ai_junction_pilot import sha,write

RUN='results/gate3_semantics/gate3_20260911_gse_conditional_development_evaluation_v1_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1'


def main():
    path=ROOT/RUN/'metrics/paired_parents.json'
    sealed={p:h for h,p in (l.split('  ',1) for l in (ROOT/RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    assert sha(path)==sealed[str(path.relative_to(ROOT))]
    data=json.loads(path.read_text());rows=data['per_parent'];parents=sorted({r['parent'] for r in rows})
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,5));colors=['#808080','#e6a23c','#167bad','#7b4ba5']
    for ax,metric,title in zip(axes,['axis_mae','height_mae_m'],['跨组件轴向关系误差（不是角度）','跨组件参考中心高差误差（米）']):
        for j,v in enumerate(['A','B','C','常数参考']):
            key=metric if j<3 else ('constant_axis_mae' if metric=='axis_mae' else 'constant_height_mae_m')
            vv=v if j<3 else 'A'
            values=[next(r for r in rows if r['parent']==p and r['variant']==vv)['metrics']['cross'][key] for p in parents]
            ax.bar([i+(j-1.5)*.2 for i in range(5)],values,width=.19,label=v,color=colors[j])
        ax.set_xticks(range(5),[p.split('_')[0] for p in parents]);ax.set_title(title);ax.set_ylabel('平均绝对误差，越低越好');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('冻结模型的五父地图开发评价：240例全部推理，231例有跨组件参考\nC07曾用于旧编码器选择；这是关系头留出开发评价，不是严格未见测试',fontsize=11)
    fig.tight_layout(rect=[0,0,.99,.87]);target=ROOT/OUT/'development_abc.png'
    if target.exists():raise FileExistsError('no figure overwrite')
    fig.savefig(target,dpi=170);plt.close(fig)
    aggregate={v:{k:mean(r['metrics']['cross'][k] for r in rows if r['variant']==v) for k in ['axis_mae','height_mae_m','constant_axis_mae','constant_height_mae_m']} for v in 'ABC'}
    write(ROOT/OUT/'development_abc_summary.json',dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),aggregate=aggregate,paired=data['paired_cross_component']))
    print(json.dumps(aggregate))


if __name__=='__main__':main()
