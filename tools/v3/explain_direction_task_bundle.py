"""Explain sealed task-support ambiguity, without new labels or predictions."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
import json
import numpy as np

RUN='results/gate3_semantics/gate3_20260913_gse_direction_task_witness_replay_v2_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1/task_bundle_attribution'

def maximum_single_label_coverage(sets):
    owners={}
    def augment(candidate,seen):
        for port in sorted(sets[candidate]):
            if port in seen:continue
            seen.add(port)
            if port not in owners or augment(owners[port],seen):owners[port]=candidate;return True
        return False
    for candidate in range(len(sets)):augment(candidate,set())
    return len(owners)

def main():
    run=ROOT/RUN;out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True)
    sealed=0
    for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h;sealed+=1
    rows=[];mixed=[]
    for i in range(12):
        prediction=json.loads((run/f'artifacts/prediction_{i:02d}.json').read_text())
        witness=json.loads((run/f'artifacts/witness_{i:02d}.json').read_text());sets=[]
        for j,c in enumerate(witness['candidates']):
            rays={int(x.rsplit('/ray:',1)[1]) for x in prediction['proposals'][j]['source_refs']}
            groups={p['port_reference']:set(witness['supporting_ray_indices'][p['port_reference']])&rays for p in c['witnessed_ports']}
            sets.append(set(groups))
            if len(groups)>1:
                exclusive={p:len(ids-set().union(*(v for k,v in groups.items() if k!=p))) for p,ids in groups.items()}
                mixed.append(dict(observation=i,candidate=j,exclusive_ray_counts=exclusive,
                    overlapping_rays=len(set.intersection(*groups.values())),
                    source_refs_sha256=__import__('hashlib').sha256(json.dumps(prediction['proposals'][j]['source_refs']).encode()).hexdigest()))
        reference=len(witness['reference']['observable_tasks']);upper=maximum_single_label_coverage(sets)
        rows.append(dict(observation=i,reference_section_occurrences=reference,candidates=len(sets),
            single_label_recovery_upper_bound=upper,unrecoverable_under_current_single_label_interface=reference-upper))
    report=dict(source_run=RUN,seal_verified=sealed,observations=rows,mixed_candidates=mixed,
        mixed_candidates_count=len(mixed),total_single_label_unrecoverable=sum(r['unrecoverable_under_current_single_label_interface'] for r in rows),
        interpretation='Upper bound only under saved ray-section witness contract; not complete physical task recall, new model score, or training labels',
        new_labels=0,new_predictions=0)
    write(out/'attribution.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for ax,i in zip(axes,[0,8]):
        p=json.loads((run/f'artifacts/prediction_{i:02d}.json').read_text())
        w=json.loads((run/f'artifacts/witness_{i:02d}.json').read_text())
        candidate=0;rays={int(x.rsplit('/ray:',1)[1]) for x in p['proposals'][candidate]['source_refs']}
        columns=np.array(sorted({r%720 for r in rays}));ax.scatter(columns*.5,np.zeros(len(columns)),s=6,c='grey',label='一个旧候选的实际列范围')
        for j,port in enumerate(w['candidates'][candidate]['witnessed_ports']):
            ids=sorted(set(w['supporting_ray_indices'][port['port_reference']])&rays)
            ax.scatter([x%720*.5 for x in ids],[x//720 for x in ids],s=6,label=f'参考支路{j+1}的射线（{len(ids)}条）')
        ax.set_title(f'观察{i}，候选0：同一个候选中，两组不同射线分别支持两条参考支路',fontproperties=font)
        ax.set_xlabel('方位角（度）',fontproperties=font);ax.set_ylabel('激光线束行',fontproperties=font)
        ax.set_xlim(0,360);ax.legend(prop=font);ax.grid(alpha=.2)
    fig.suptitle('先压成一个方向，再给它一个任务编号，会丢失内部支路\n这是构造条件射线证据，不是完整通道真值或可通行证明',fontproperties=font)
    fig.tight_layout();fig.savefig(out/'mixed_candidate_rays.png',dpi=140);plt.close(fig)
    print(json.dumps(report))

if __name__=='__main__':main()
