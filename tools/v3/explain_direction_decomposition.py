"""Read-only attribution of the single observed component experiment."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
from collections import deque
import json
import numpy as np

RUN='results/gate3_semantics/gate3_20260913_gse_direction_task_decomposition_v1_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1/direction_decomposition'
SOURCE='results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'

def bridge_path(allowed,starts,ends):
    queue=deque(sorted(starts));previous={v:None for v in starts}
    while queue:
        v=queue.popleft()
        if v in ends:
            result=[]
            while v is not None:result.append(v);v=previous[v]
            return result[::-1]
        r,c=divmod(v,720)
        for nr,nc in ((r-1,c),(r+1,c),(r,(c-1)%720),(r,(c+1)%720)):
            n=nr*720+nc
            if 0<=nr<16 and n in allowed and n not in previous:previous[n]=v;queue.append(n)
    return None

def main():
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True);run=ROOT/RUN;seals=0
    for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h,p;seals+=1
    rows=[];bridges=[];fig_inputs=[]
    for i in range(12):
        p=json.loads((run/f'artifacts/prediction_{i:02d}.json').read_text());w=json.loads((run/f'artifacts/witness_{i:02d}.json').read_text())
        source=p['source'];file=ROOT/SOURCE/source['student_path'];assert sha(file)==source['student_sha256']
        with np.load(file,allow_pickle=False) as data:ranges=data['ranges_m'][-1].copy()
        parents={c['parent_candidate']:c['parent_decomposition_evidence'] for c in p['proposals']}
        rows.append(dict(observation=i,parent_component_sizes={str(j):[len(v) for v in a['component_ray_indices']] for j,a in parents.items()},
                         residual_ray_count=sum(len(a['residual_ray_indices']) for a in parents.values())))
        for j,c in enumerate(w['candidates']):
            if c['reason']!='COMPETING_SECTIONS':continue
            rays={int(s.rsplit('/ray:',1)[1]) for s in p['proposals'][j]['source_refs']}
            ports=c['witnessed_ports'];a=set(w['supporting_ray_indices'][ports[0]['port_reference']])&rays;b=set(w['supporting_ray_indices'][ports[1]['port_reference']])&rays
            path=bridge_path(rays,a,b)
            bridges.append(dict(observation=i,candidate=j,first_reference=ports[0]['port_reference'],second_reference=ports[1]['port_reference'],
                image_adjacency_path=path,path_min_first_return_m=float(ranges.reshape(-1)[path].min()) if path else None,
                interpretation='Image-direction adjacency only, NOT robot path or proof no discriminative geometry exists'))
        if i in (0,5):fig_inputs.append((i,p,w,ranges))
    summary=json.loads((run/'metrics/summary.json').read_text())
    report=dict(source_run=RUN,seal_verified=seals,parent_components=rows,mixed_component_bridges=bridges,
        mixed_component_count=len(bridges),all_mixed_have_observed_direction_paths=all(b['image_adjacency_path'] is not None for b in bridges),
        original_candidates=19,new_candidates=summary['candidate_count'],conditional_scores=summary['conditional_scores'],
        conclusion='Fixed10m boundary adjacency neither separates all reference branches nor prevents ray-grid fragmentation; stop this component rule, no radius/pruning search',
        new_predictions=0,new_labels=0,training_steps=0)
    write(out/'attribution.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import ListedColormap
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(2,2,figsize=(14,7))
    for row,(i,p,w,ranges) in enumerate(fig_inputs):
        assignments=np.full((16,720),-1,int)
        for j,c in enumerate(p['proposals']):
            ids=[int(s.rsplit('/ray:',1)[1]) for s in c['source_refs']];assignments.reshape(-1)[ids]=j
        axes[row,0].imshow(ranges,origin='lower',aspect='auto',extent=[0,360,0,16],vmin=0,vmax=50,cmap='viridis')
        axes[row,0].set_title(f'观察{i}：原始回波距离，未使用参考切分',fontproperties=font)
        axes[row,1].imshow(np.ma.masked_less(assignments,0),origin='lower',aspect='auto',extent=[0,360,0,16],cmap='tab20',interpolation='nearest')
        axes[row,1].set_title(f'观察{i}：观察侧分组，共{len(p["proposals"])}个；空白射线保留在残余记录',fontproperties=font)
        for ax in axes[row]:ax.set_xlabel('方位角（度）',fontproperties=font);ax.set_ylabel('激光线束行',fontproperties=font)
    fig.suptitle('一次固定规则的实际结果：有的支路仍混在一起，有的区域被切出小碎片\n颜色仅为观察侧分组编号，不是语义标签；没有按参考删小块或调半径',fontproperties=font)
    fig.tight_layout();fig.savefig(out/'components.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(seal_verified=seals,mixed_components=len(bridges),bridge_paths=sum(b['image_adjacency_path'] is not None for b in bridges),parent_component_sizes=rows)))

if __name__=='__main__':main()
