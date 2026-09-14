"""Same saved outputs: visualize all12 and identify the two split cores."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import write,sha
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

RUN=ROOT/'results/gate3_semantics/gate3_20260913_gse_local_conflict_replay_v1_seed0'
FIT=ROOT/'results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0'
REVIEW=ROOT/'results/gate3_semantics/gate3_20260913_gse_ai_branch_review_v1r1_seed0'
OUT=ROOT/'docs/figures/gse_conditional_geometry_fit_v1/local_conflict_replay'

def main():
    for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h
    summary=json.loads((RUN/'metrics/summary.json').read_text());OUT.mkdir(parents=True,exist_ok=True)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axs=plt.subplots(12,2,figsize=(14,18));splits=[]
    colors=matplotlib.colors.ListedColormap(['#dddddd','#377eb8','#ff7f00','#4daf4a','#984ea3'])
    for i,row in enumerate(summary['observations']):
        with np.load(FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids'];pairs=d['pairs'];prob=d['probabilities']
        with np.load(REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:labels=d['core_labels'];assert np.array_equal(ids,d['ray_ids'])
        h=json.loads((RUN/f'artifacts/groups_{i:02d}.json').read_text());byray=np.full(11520,-1,dtype=int)
        for j,g in enumerate(h['groups']):byray[g]=j
        mapped=byray[ids];truth=np.full(11520,0,dtype=int);truth[ids]=labels+1
        displayed=np.zeros(11520,dtype=int);displayed[ids]=mapped+1
        axs[i,0].imshow(truth.reshape(16,720),aspect='auto',origin='lower',cmap=colors,vmin=0,vmax=4)
        axs[i,1].imshow(displayed.reshape(16,720),aspect='auto',origin='lower',cmap=colors,vmin=0,vmax=4)
        for ax in axs[i]:ax.set_yticks([]);ax.set_xticks([0,180,360,540,719]);ax.set_ylabel(str(i),rotation=0)
        axs[i,1].text(725,8,'核心完整' if row['new']['groups_per_core']==[1,1] else '核心被切分',fontproperties=font,fontsize=9)
        if row['new']['groups_per_core']!=[1,1]:
            a,b=pairs[:,0],pairs[:,1]
            candidate=np.flatnonzero((labels[a]>=0)&(labels[a]==labels[b])&(mapped[a]!=mapped[b])&(prob>=.9));assert len(candidate)
            j=int(candidate[0]);u,v=map(int,pairs[j]);g1,g2=int(mapped[u]),int(mapped[v])
            negative=np.flatnonzero((prob<=.1)&(((mapped[a]==g1)&(mapped[b]==g2))|((mapped[a]==g2)&(mapped[b]==g1))));assert len(negative)
            k=int(negative[0]);n1,n2=map(int,pairs[k])
            splits.append(dict(observation=i,core_counts_per_group=[[int(((mapped==g)&(labels==c)).sum()) for c in (0,1)] for g in range(len(h['groups']))],
                same_core_positive_cut=dict(ray_ids=[int(ids[u]),int(ids[v])],p_same=float(prob[j]),reference_core=int(labels[u]),final_groups=[g1,g2]),
                repulsive_constraint_between_these_groups=dict(ray_ids=[int(ids[n1]),int(ids[n2])],p_same=float(prob[k]),reference_core_labels=[int(labels[n1]),int(labels[n2])]),
                same_core_positive_pairs_separated=int(len(candidate)),interpretation='A correct within-core attraction is incompatible with group-level repulsion involving unlabelled evidence; unknown is not automatically a proven negative error'))
    axs[0,0].set_title('部分AI核心参考：灰色为未标，不是背景',fontproperties=font)
    axs[0,1].set_title('新后端：全部射线预测分组；颜色不代表正确性',fontproperties=font)
    fig.suptitle('固定12例；旧后端全部0组。新后端10例核心完整、2例切分、0核心错并\n未标区域的分组没有真值资格，不能据颜色宣布正确',fontproperties=font)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'all12_grouping.png',dpi=130);plt.close(fig)
    write(OUT/'split_attribution.json',dict(source_run=str(RUN.relative_to(ROOT)),run_summary_sha256=sha(RUN/'metrics/summary.json'),code_sha256=sha(ROOT/'tools/v3/explain_local_conflict_replay.py'),cases=splits,training_steps_added=0,thresholds_changed=False))
    print(json.dumps(splits,ensure_ascii=False))

if __name__=='__main__':main()
