"""Read-only causal conflict attribution of sealed final predictions."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
from collections import deque
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

RUN=ROOT/'results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0'
REVIEW=ROOT/'results/gate3_semantics/gate3_20260913_gse_ai_branch_review_v1r1_seed0'
OUT=ROOT/'docs/figures/gse_conditional_geometry_fit_v1/branch_core_fit'

def main():
    assert json.loads((RUN/'RUN_STATE.json').read_text())['updates']==500
    for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for i in range(12):
        file=RUN/f'artifacts/prediction_0500_{i:02d}.npz'
        with np.load(file,allow_pickle=False) as d:ids=d['ray_ids'];pairs=d['pairs'];prob=d['probabilities']
        with np.load(REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:assert np.array_equal(ids,d['ray_ids']);labels=d['core_labels']
        n=len(ids);parent=list(range(n));adj=[[] for _ in range(n)]
        def root(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        for index,(a,b) in enumerate(pairs):
            if prob[index]>=.9:
                a=int(a);b=int(b);parent[root(a)]=root(b);adj[a].append((b,index));adj[b].append((a,index))
        negative=[j for j,(a,b) in enumerate(pairs) if prob[j]<=.1 and root(int(a))==root(int(b))]
        supervised=[j for j in negative if labels[pairs[j,0]]>=0 and labels[pairs[j,1]]>=0 and labels[pairs[j,0]]!=labels[pairs[j,1]]]
        assert supervised
        j=supervised[0];start,goal=map(int,pairs[j]);prev={start:None};queue=deque([start])
        while queue and goal not in prev:
            a=queue.popleft()
            for b,k in adj[a]:
                if b not in prev:prev[b]=(a,k);queue.append(b)
        assert goal in prev;path=[];node=goal
        while prev[node] is not None:
            a,k=prev[node];path.append(dict(from_ray=int(ids[a]),to_ray=int(ids[node]),p_same=float(prob[k]),reference='SAME_CORE' if labels[a]>=0 and labels[a]==labels[node] else 'UNKNOWN'));node=a
        path.reverse();assert any(e['reference']=='UNKNOWN' for e in path)
        known=(labels[pairs[:,0]]>=0)&(labels[pairs[:,1]]>=0)
        row=dict(observation=i,prediction_sha256=sha(file),positive_connected_components=len({root(a) for a in range(n)}),negative_edges_inside_positive_component=len(negative),supervised_negative_edges_inside_positive_component=len(supervised),
            unknown_confident_pairs=int(((~known)&((prob>=.9)|(prob<=.1))).sum()),unknown_pairs=int((~known).sum()),
            witness=dict(negative_pair=[int(ids[start]),int(ids[goal])],p_same=float(prob[j]),reference='DIFFERENT_AI_CORES',positive_path=path),
            interpretation='Positive path conflicts with a separately correct core-negative; unknown links are not ground-truth errors, but predictions are not globally consistent')
        rows.append(row)
    write(OUT/'conflicts.json',dict(source_run=str(RUN.relative_to(ROOT)),code_sha256=sha(ROOT/'tools/v3/explain_branch_core_fit.py'),observations=rows,training_steps_added=0,thresholds_changed=False))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    r=rows[0];path=r['witness']['positive_path'];nodes=[path[0]['from_ray']]+[e['to_ray'] for e in path];fig,ax=plt.subplots(figsize=(max(10,len(nodes)*2),4))
    for j,ray in enumerate(nodes):
        ax.scatter(j,0,s=500,c='#dddddd',edgecolors='black',zorder=3);ax.text(j,-.2,f'射线 {ray}',ha='center',fontproperties=font)
    for j,e in enumerate(path):
        ax.annotate('',xy=(j+1,0),xytext=(j,0),arrowprops=dict(arrowstyle='->',color='green',lw=2));ax.text(j+.5,.1,f"同组 {e['p_same']:.4f}\n参考：{'未知' if e['reference']=='UNKNOWN' else '同核心'}",ha='center',fontproperties=font)
    ax.annotate('',xy=(len(nodes)-1,0),xytext=(0,0),arrowprops=dict(arrowstyle='<->',color='red',lw=2,connectionstyle='arc3,rad=-.4'))
    ax.text((len(nodes)-1)/2,-.75,f"首尾属于两个已标核心：模型同组概率 {r['witness']['p_same']:.6f}\n同组路径与异组关系冲突；后端整块拒绝，不是训练没有发生",ha='center',fontproperties=font)
    ax.set(xlim=(-.6,len(nodes)-.4),ylim=(-1.1,.6));ax.axis('off');ax.set_title('固定观察0：保存预测中的真实冲突路径（未调阈值）',fontproperties=font)
    fig.tight_layout();fig.savefig(OUT/'conflict_case0.png',dpi=150);plt.close(fig)
    final=json.loads((RUN/'metrics/evaluation_0500.json').read_text());initial=json.loads((RUN/'metrics/evaluation_0000.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].bar(['step0','step500'],[initial['macro_f1'],final['macro_f1']],color=['gray','steelblue']);axs[0].set_ylim(0,1.05);axs[0].set_title('已标关系 macro-F1（拟合，不是测试）',fontproperties=font)
    axs[1].bar(['step0','step500'],[initial['complete_core_cases'],final['complete_core_cases']],color=['gray','steelblue']);axs[1].set_ylim(0,12);axs[1].set_title('全预测分组完整恢复案例数 / 12',fontproperties=font)
    fig.tight_layout();fig.savefig(OUT/'fit_vs_grouping.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(cases=12,unknown_confident=sum(r['unknown_confident_pairs'] for r in rows),unknown_total=sum(r['unknown_pairs'] for r in rows),witness=r['witness'])))

if __name__=='__main__':main()
