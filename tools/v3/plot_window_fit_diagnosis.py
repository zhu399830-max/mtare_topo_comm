"""Publish deterministic figures from sealed predictions; no model or label edits."""
import _bootstrap
import hashlib,io,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_synthetic_window_fit_v1_seed0'
SEAL='61df3a37c0a70daf4fd8785763292ee9323f818b5f16f692406a45f5f108bf17'
OUT=ROOT/'docs/figures/gse_window_fit_diagnosis_20260908_fontfixed'
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
NAMES={'straight':'直道','terminal':'尽头','T':'T形路口','Y':'Y形路口','four_way':'四叉路口','visible_blocker':'可见阻挡'}


def main():
    seal=(RUN/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!=SEAL:raise ValueError('sealed results drift')
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    opened={}
    def read(relative):
        p=RUN/relative;b=p.read_bytes();key=str(p.relative_to(ROOT))
        if hashlib.sha256(b).hexdigest()!=pins[key]:raise ValueError('input drift')
        opened[key]=pins[key];return b
    scores=json.loads(read('metrics/scores.json'))
    result=torch.load(io.BytesIO(read('artifacts/paired_result.pt')),weights_only=True)
    examples=torch.load(io.BytesIO(read('artifacts/shared_examples.pt')),weights_only=True)
    if OUT.exists():raise FileExistsError('figure directory already exists; no overwrite')
    OUT.mkdir(parents=True)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False})
    totals={}
    for branch in ('A','B','C'):
        groups=defaultdict(lambda:{k:dict(tp=0,fp=0,fn=0) for k in ('anchor','opening')})
        for row in scores[branch]['final']['per_case']:
            group=groups[row['case_id'].split('__')[0]]
            for kind in group:
                for key in group[kind]:group[kind][key]+=row[kind][key]
        totals[branch]=dict(groups)
    for kind,title in NAMES.items():
        case_id=kind+'__circle__view2';index=next(i for i,e in enumerate(examples) if e['header']['observation_id']==case_id)
        ex=examples[index];target=ex['targets'];xyz=ex['compact']['points_xyz_m'][0].numpy();valid=ex['compact']['valid'][0].numpy()
        points=xyz[valid&(np.linalg.norm(xyz,axis=1)<=10)][::8]
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for col,branch in enumerate(('A','B','C')):
            pred=result['evaluations'][branch]['final'][index]['prediction']
            for row,(dims,plane) in enumerate((((0,1),'XY平面'),((0,2),'XZ平面'))):
                ax=axes[row,col];ax.scatter(points[:,dims[0]],points[:,dims[1]],s=1,c='.75',alpha=.35,label='观测点（显示抽稀）')
                for label,color,marker,prefix,truth in [('真实节点','#006400','o','anchor',True),('预测节点','#c00000','x','anchor',False),
                    ('真实开口','#0044bb','s','opening',True),('预测开口','#ee8800','+','opening',False)]:
                    source=target if truth else pred
                    mask=source[prefix+'_valid'][0] if truth else source[prefix+'_presence_logits'][0].sigmoid()>=.5
                    p=source[prefix+'_position_m'][0][mask].numpy()
                    ax.scatter(p[:,dims[0]],p[:,dims[1]],s=70,c=color,marker=marker,label=label,alpha=.85)
                ax.set_title(branch+{'A':' 原始观测','B':' 面片属性','C':' 面片关系'}[branch]+' · '+plane)
                ax.set_aspect('equal');ax.set_xlim(-11,11);ax.set_ylim(-11,11);ax.grid(alpha=.2)
                ax.set_xlabel('X / 米');ax.set_ylabel(('Y' if row==0 else 'Z')+' / 米')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
        fig.suptitle(title+'：同一观测的真实标签与最终预测\n固定圆形截面/中央视点；训练集拟合，不是未见测试；置信度≥0.5',fontsize=14)
        fig.savefig(OUT/(kind+'.png'),dpi=150);plt.close(fig)
    (OUT/'error_counts.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(source_run=str(RUN.relative_to(ROOT)),source_seal_sha256=SEAL,input_sha256=opened,
        case_selection='All six scored program types; circle/view2 fixed, no ranking or best-case selection.',
        visualization_only_stride=8,scoring_unchanged=True,new_optimizer_steps=0,
        font_path=str(FONT),font_sha256=hashlib.sha256(FONT.read_bytes()).hexdigest(),
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(OUT.iterdir()) if p.is_file()))
    print(json.dumps(dict(output=str(OUT),figures=6,error_counts=totals)),flush=True)


if __name__=='__main__':main()
