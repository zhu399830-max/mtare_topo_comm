"""Ideal AI-core relations through the unchanged grouping backend, no model."""
from ai_branch_review import ROOT,RUN,write,sha
import numpy as np
from dataclasses import asdict
from mtare_topo.topology.branch_hypotheses import group_signed_relations

def main():
    out=ROOT/RUN;rows=[]
    write(out/'config/grouping_check.json',dict(code='tools/v3/check_branch_review_grouping.py',sha256=sha(ROOT/'tools/v3/check_branch_review_grouping.py'),backend_sha256=sha(ROOT/'src/mtare_topo/topology/branch_hypotheses.py'),meaning='Perfect scores on partial AI-core relations; not model results'))
    for i in range(12):
        with np.load(out/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:
            ids=d['ray_ids'].tolist();labels=d['core_labels'];pairs=d['pairs'];targets=d['conditional_ai_targets']
        edges=[(ids[a],ids[b],1. if t==1 else -1.) for (a,b),t in zip(pairs,targets) if t>=0]
        h=group_signed_relations(ids,edges)
        assert h==group_signed_relations(ids[::-1],edges[::-1])
        lookup=dict(zip(ids,labels.tolist()));coregroups=[0,0];wrong=0
        for group in h.groups:
            kinds={lookup[r] for r in group}
            assert -1 not in kinds
            wrong+=int(len(kinds)>1)
            for k in kinds:coregroups[k]+=1
        row=dict(observation=i,groups=len(h.groups),groups_per_ai_core=coregroups,wrong_core_merges=wrong,unresolved=len(h.unresolved_ray_ids),
            labeled_unresolved=sum(lookup[r]>=0 for r in h.unresolved_ray_ids),order_invariant=True,model_result=False,training_qualified=False)
        write(out/f'artifacts/ideal_groups_{i:02d}.json',asdict(h));rows.append(row)
    write(out/'metrics/ideal_grouping.json',dict(observations=rows,meaning='Partial AI core oracle, not full branch oracle or method score',training_steps=0))
    print(rows)

if __name__=='__main__':main()
