"""Render sealed axis hypotheses, without new inference or target generation."""
import _bootstrap
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_axis_pair_diagnostic_v1_seed0'
OUT=ROOT/'docs/figures/gse_axis_pair_diagnostic_20260908'
SEAL='6b2e4ae2ff89617f5ec2edf81a55e19c40f2b1bbfc5b6bc040dc5e49f75c06b2'
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
CASES=(('straight','直道'),('terminal','尽头'),('visible_blocker','可见阻挡'),('T','T形路口'),('Y','Y形路口'),('four_way','四叉路口'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seal=RUN/'artifacts/evidence_sha256.txt'
    if sha(seal)!=SEAL:raise ValueError('seal drift')
    pins={p:h for h,p in (s.split('  ',1) for s in seal.read_text().splitlines())}
    records=[];sources={}
    for kind,title in CASES:
        p=RUN/('artifacts/'+kind+'__circle__view2.json');key=str(p.relative_to(ROOT))
        if sha(p)!=pins[key]:raise ValueError('source drift')
        sources[key]=pins[key];records.append((title,json.loads(p.read_text())))
    if OUT.exists():raise FileExistsError('no figure overwrite')
    OUT.mkdir(parents=True)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False})
    for vertical,name in ((1,'XY'),(2,'XZ')):
        fig,axs=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,(title,r) in zip(axs.flat,records):
            for n,line in enumerate(r['axes_m']):
                x=np.asarray(line)
                ax.plot(x[:,0],x[:,vertical],'-o',markersize=3,color='steelblue',alpha=.7,label='拟合轴线（不是墙面）' if n==0 else None)
            candidates=[p for p in r['proposals']['pairs'] if p['position_m'] is not None]
            for n,p in enumerate(candidates):
                x=p['position_m'];ax.scatter(x[0],x[vertical],c='darkorange',marker='x',s=80,label='未验证候选' if n==0 else None)
                ax.annotate(str(n+1),(x[0],x[vertical]),xytext=(5,6+n*9),textcoords='offset points',fontsize=8)
            s=r['score']['scores']['1.0']
            ax.set_title(f"{title}：候选{len(candidates)} / 匹配{s['matched']} / 多余{s['false_positives']} / 漏失{s['missed']}")
            ax.set_xlabel('当前传感器 X（米）');ax.set_ylabel('当前传感器 '+name[1]+'（米）')
            ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle('同一封存实验：圆形截面 view2，六类固定案例\n仅显示已输出轴线与候选；重叠编号表示多个候选，未绘制真实地图或补造尽头',fontsize=14)
        fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)
    provenance=dict(source_run=str(RUN.relative_to(ROOT)),seal_sha256=SEAL,input_sha256=sources,
                    script_sha256=sha(Path(__file__)),font_sha256=sha(FONT),selection='All six fixed circle/view2 types, no score selection',
                    new_inference=0,new_labels=0,display='fitted axes and unverified proposals only')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(OUT.iterdir()) if p.is_file()))
    print(OUT)


if __name__=='__main__':main()
