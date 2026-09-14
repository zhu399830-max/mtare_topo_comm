"""Read sealed direction predictions only; no inference, labels or tuning."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    root=Path(__file__).resolve().parents[2]
    run=root/'results/gate3_semantics/gate3_20260912_gse_observation_axis_comparison_v1_seed0'
    result=json.loads((run/'metrics/paired_parents.json').read_text())
    out=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison'
    out.mkdir(exist_ok=True)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    def title(ax,text):ax.set_title(text,fontproperties=font,fontsize=11)
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    parents=result['parents']; x=np.arange(len(parents)); w=.25
    for ax,category,label in zip(axs,['all','cross'],['全部已知关系','跨构造片段关系']):
        for offset,method,color in [(-w,'POINT','#df7f35'),(0,'NORMAL','#2079a4')]:
            vals=[result['results'][method][category]['per_parent'][p]['full_population_error'] for p in parents]
            ax.bar(x+offset,vals,w,label=method,color=color)
        const=[result['results']['POINT'][category]['per_parent'][p]['constant_one_mae'] for p in parents]
        ax.bar(x+w,const,w,label='constant parallel',color='#777777')
        ax.set_xticks(x,[p.split('_')[0] for p in parents]);ax.set_ylabel('absolute dot-product error');ax.legend(fontsize=8)
        title(ax,label+'：越低越好；无答案按最大误差计入')
    fig.suptitle('真实240观察：几何组合优于点坐标主轴，但两者均差于常数参考',fontproperties=font,fontsize=14)
    fig.tight_layout();fig.savefig(out/'parent_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(5,2,figsize=(11,17))
    for row,case in enumerate([0,48,96,144,192]):
        for col,method in enumerate(['POINT','NORMAL']):
            with np.load(run/f'artifacts/{method}_{case:03d}.npz') as f:
                centers=f['centers']; axes=f['axes']; valid=f['axis_valid']
            ax=axs[row,col];ax.scatter(centers[:,0],centers[:,1],s=2,c=centers[:,2],cmap='viridis')
            indices=np.arange(0,len(centers),20);indices=indices[valid[indices]]
            a=axes[indices];c=centers[indices]
            ax.quiver(c[:,0]-.4*a[:,0],c[:,1]-.4*a[:,1],.8*a[:,0],.8*a[:,1],
                angles='xy',scale_units='xy',scale=1,width=.003,headwidth=0,headlength=0,headaxislength=0,color='#b83137')
            ax.set_aspect('equal');ax.set_xlim(-10,10);ax.set_ylim(-10,10);ax.set_xlabel('X / m');ax.set_ylabel('Y / m')
            title(ax,f'case {case} / {method}；有效方向 {valid.sum()}/{len(valid)}')
    fig.suptitle('固定每父地图首例：面片中心按高度着色；红线为估计轴向，不是通道或边\n每20个面片固定显示一个方向；图用于查错，不代表完整结构预测',fontproperties=font,fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'fixed_cases_xy.png',dpi=150);plt.close(fig)
    print(str(out))


if __name__=='__main__':main()
