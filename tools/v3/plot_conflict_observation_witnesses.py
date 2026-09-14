"""Project saved causal conflict witnesses onto existing observed returns."""
from _bootstrap import PROJECT_ROOT as ROOT
from explain_local_conflict_replay import OUT, RUN, FIT
from run_short_observation_chain import write, sha
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

CACHE=ROOT/'results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'

def main():
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    cases=json.loads((OUT/'rejection_chronology.json').read_text())['cases'];measurements=[]
    for c in cases:
        i=c['observation'];cache=CACHE/f'artifacts/common_{i:02d}.npz'
        with np.load(cache,allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'][4*11520:5*11520].copy()
        with np.load(FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids']
        assert xyz.shape==(11520,3) and np.isfinite(xyz[ids]).all()
        fig,axs=plt.subplots(1,3,figsize=(18,6))
        edges=c['accepted_path_left']+c['accepted_path_right']
        for ax,(a,b) in zip(axs,[(0,1),(0,2),(1,2)]):
            ax.scatter(xyz[ids,a],xyz[ids,b],s=.7,c='#bbbbbb',alpha=.4)
            for field,col in [('accepted_path_left','#1976d2'),('accepted_path_right','#ef6c00')]:
                for e in c[field]:
                    q=xyz[e['ray_ids']];ax.plot(q[:,a],q[:,b],color=col,alpha=.7,linewidth=1)
            for pair,col,style in [(c['rejected_edge'],'#008000','-'),(c['witness']['ray_ids'],'#d50000','--')]:
                q=xyz[pair];ax.plot(q[:,a],q[:,b],color=col,linestyle=style,linewidth=3,marker='o',markersize=4)
                for r in pair:ax.annotate(str(r),(xyz[r,a],xyz[r,b]),fontsize=7)
            ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('XYZ'[a]+' (m)');ax.set_ylabel('XYZ'[b]+' (m)')
        fig.suptitle(f'观察{i}：灰=当前真实回波；蓝/橙=之前已接受的关系路径\n绿=被拒绝的已标核心关系；红虚线=当时的排斥见证（未标）\n线表示模型关系，不表示机器人可走路径；不据此新增标签',fontproperties=font)
        fig.tight_layout(rect=(0,0,1,.88));fig.savefig(OUT/f'conflict_geometry_{i:02d}.png',dpi=140);plt.close(fig)
        def describe(pair):
            q=xyz[pair];norm=np.linalg.norm(q,axis=1)
            return dict(ray_ids=pair,xyz_m=q.tolist(),range_m=norm.tolist(),separation_m=float(np.linalg.norm(q[0]-q[1])),angle_deg=float(np.degrees(np.arccos(np.clip(np.dot(q[0],q[1])/np.prod(norm),-1,1)))))
        longest=max(edges,key=lambda e:np.linalg.norm(xyz[e['ray_ids'][0]]-xyz[e['ray_ids'][1]]))
        measurements.append(dict(observation=i,cache_sha256=sha(cache),rejected=describe(c['rejected_edge']),repulsion=describe(c['witness']['ray_ids']),longest_accepted_witness_edge=dict(**describe(longest['ray_ids']),p_same=longest['p_same'],reference_labels=longest['reference_labels']),all_lines_are_relations_not_traversal=True))
    write(OUT/'conflict_geometry.json',dict(cases=measurements,labels_added=0,code_sha256=sha(ROOT/'tools/v3/plot_conflict_observation_witnesses.py')))
    print(json.dumps(measurements,ensure_ascii=False))

if __name__=='__main__':main()
