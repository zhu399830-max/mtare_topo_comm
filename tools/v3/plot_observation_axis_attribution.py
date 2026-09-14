"""Plot saved full-population attribution; no new inference or selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison'
    data=json.loads((folder/'error_attribution.json').read_text())['summary']
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    bins=data['NORMAL']['cross']['bins']['reference_angle'];x=np.arange(4)
    axes[0].bar(x,[100*b['mass'] for b in bins],color='#64788c')
    axes[0].set_xticks(x,['0–5°','5–15°','15–30°','30–90°'])
    axes[0].set_ylabel('weighted population / %')
    axes[0].set_title('参考大多近似平行\n小于15°约86%',fontproperties=font)
    for method,color in [('POINT','#df7f35'),('NORMAL','#2079a4')]:
        b=data[method]['cross']['bins']['reference_angle']
        axes[1].plot(x,[v['conditional_predicted_angle_deg'] for v in b],'o-',label=method,color=color)
    axes[1].set_xticks(x,['0–5°','5–15°','15–30°','30–90°'])
    axes[1].set_ylabel('mean predicted angle / degree');axes[1].legend()
    axes[1].set_title('按参考夹角分组的实际预测\n几何组合仍未区分大夹角关系',fontproperties=font)
    bins=data['NORMAL']['cross']['bins']['relative_eigengap']
    axes[2].bar(x-.18,[100*b['mass'] for b in bins],.36,label='population',color='#64788c')
    total=data['NORMAL']['cross']['full_error']
    axes[2].bar(x+.18,[100*b['error_contribution']/total for b in bins],.36,label='error contribution',color='#c25754')
    axes[2].set_xticks(x,['<0.01','0.01–0.1','0.1–0.5','≥0.5'])
    axes[2].set_ylabel('%');axes[2].legend(fontsize=8)
    axes[2].set_title('法向方向约束强度（谱间隙）\n弱约束约61%人口、约77%误差',fontproperties=font)
    fig.suptitle('全部240观察的只读错误分解：固定描述性分组，不用于删样本或改阈值',fontproperties=font,fontsize=14)
    fig.tight_layout();fig.savefig(folder/'error_attribution.png',dpi=170);plt.close(fig)
    print(str(folder/'error_attribution.png'))


if __name__=='__main__':main()
