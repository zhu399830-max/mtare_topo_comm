"""Read-only support inventory; nearest returns are not branch identity labels."""
from _bootstrap import PROJECT_ROOT as ROOT
from explain_local_conflict_replay import OUT, RUN, FIT, REVIEW
from plot_conflict_observation_witnesses import CACHE
from run_short_observation_chain import write, sha
import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    entries=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_branch_core_fit_v1.json').read_text())['scope']['entries']
    rows=[];font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axs=plt.subplots(5,2,figsize=(12,18));row=0
    for i,e in enumerate(entries):
        with np.load(REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:labels=dict(zip(d['ray_ids'].tolist(),d['core_labels'].tolist()))
        groups=json.loads((RUN/f'artifacts/groups_{i:02d}.json').read_text())['groups']
        unknown=[(j,g) for j,g in enumerate(groups) if all(labels[r]<0 for r in g)]
        if not unknown:continue
        cache=CACHE/f'artifacts/common_{i:02d}.npz'
        with np.load(cache,allow_pickle=False) as d:points=d['registered_returns_xyz_m'].reshape(5,11520,3).copy()
        with np.load(ROOT/e['student_path'],allow_pickle=False) as d:valid=d['valid_mask'].reshape(5,11520).astype(bool)
        for j,g in unknown:
            xyz=points[4,g];history=[]
            for t in range(4):
                source=points[t,valid[t]&np.isfinite(points[t]).all(1)]
                assert len(source)
                distances=cKDTree(source).query(xyz,k=1)[0]
                history.append(dict(history_index=t,median_nearest_return_m=float(np.median(distances)),p90_nearest_return_m=float(np.quantile(distances,.9)),maximum_nearest_return_m=float(distances.max())))
            rows.append(dict(observation=i,group_index=j,rays=len(g),range_m_quantiles=np.quantile(np.linalg.norm(xyz,axis=1),[0,.5,1]).tolist(),bounds_m=[xyz.min(0).tolist(),xyz.max(0).tolist()],previous_observation_support=history,cache_sha256=sha(cache),student_sha256=sha(ROOT/e['student_path']),branch_identity_confirmed=False))
            for ax,(a,b) in zip(axs[row],[(0,1),(0,2)]):
                q=points[4,valid[4]];ax.scatter(q[:,a],q[:,b],s=.5,c='#bbbbbb',alpha=.5);ax.scatter(xyz[:,a],xyz[:,b],s=3,c='#a000c8')
                ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('XYZ'[a]+' (m)');ax.set_ylabel('XYZ'[b]+' (m)')
                ax.set_title(f'观察{i}，组{j}：{len(g)}条回波；紫色均未标',fontproperties=font)
            row+=1
    assert row==5
    fig.suptitle('5个未知独立组：有观测回波不等于独立支路\n仅显示已有输出；不删除未知，不添加标签',fontproperties=font)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'unknown_groups.png',dpi=120);plt.close(fig)
    write(OUT/'unknown_group_support.json',dict(cases=rows,code_sha256=sha(ROOT/'tools/v3/explain_unknown_branch_groups.py'),labels_added=0,model_forwards=0,interpretation='Nearest prior aligned return distances quantify surface persistence only; no correspondence threshold or branch identity is inferred. Overlapping history is not independent evidence.'))
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
