"""Font-corrected rendering from saved diagnostic numbers; no experiment rerun."""
import hashlib
import json
from pathlib import Path
from _bootstrap import PROJECT_ROOT as ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager


def main():
    src=ROOT/'docs/figures/gse_graph/continuous_anchor_motion_20260909'
    report=src/'motion_review.json'
    lines=(src/'evidence_sha256.txt').read_text().splitlines()
    pins={p:h for h,p in (line.split('  ',1) for line in lines)}
    assert hashlib.sha256(report.read_bytes()).hexdigest()==pins[str(report.relative_to(ROOT))]
    data=json.loads(report.read_text())
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    out=src.parent/'continuous_anchor_motion_20260909_zh'
    out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    for ax,r in zip(axes.flat,data['records']):
        p=np.array(r['sensor_positions_first_frame_m']);x=np.array(r['compensated_positions_m']);mask=np.array(r['selected'])
        ax.plot(p[:,0],p[:,1],'k.-',label='机器人轨迹')
        for i in range(10):
            ax.scatter(x[i,mask[i],0],x[i,mask[i],1],c=np.full(mask[i].sum(),i),cmap='viridis',vmin=0,vmax=9,s=30)
        ax.set_title(r['task'].split('__')[-1]+' / '+r['method'].upper())
        ax.set_xlabel('固定坐标 X（米）');ax.set_ylabel('固定坐标 Y（米）');ax.axis('equal');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('机器人前进约9米，候选路口也随之移动\n黑线：机器人；紫→黄：早→晚的全部候选。未用真值，不代表已确认的路口对应。',fontsize=13)
    fig.savefig(out/'motion_xy.png',dpi=150);fig.savefig(out/'motion_xy.pdf');plt.close(fig)
    provenance=dict(source=str(report.relative_to(ROOT)),source_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        font=str(font),font_sha256=hashlib.sha256(font.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes='Chinese font rendering only; original analysis and figures retained')
    (out/'provenance.json').write_text(json.dumps(provenance)+'\n')
    print(out)


if __name__=='__main__':main()
