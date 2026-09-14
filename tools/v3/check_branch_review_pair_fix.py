"""One preregistered dyadic pair coverage correction; frozen AI cores."""
from ai_branch_review import ROOT,RUN,write,sha
import json,numpy as np,torch
from dataclasses import asdict
from mtare_topo.representation.branch_relation_learning import observed_ray_pairs
from mtare_topo.topology.branch_hypotheses import group_signed_relations

def main():
    out=ROOT/RUN;rows=[]
    write(out/'config/pair_correction.json',dict(policy='dyadic_v2',definition='azimuth powers2:1..512 plus360; elevation1,2,4,8; valid rays only',
        reason='Original local_v1 never compares >64deg and skips separated elevation runs; one evidence-located correction, no label-driven pairs',
        code={p:sha(ROOT/p) for p in ['tools/v3/check_branch_review_pair_fix.py','src/mtare_topo/representation/branch_relation_learning.py','src/mtare_topo/topology/branch_hypotheses.py']},
        review_sha256=sha(out/'artifacts/review.json'),training_steps=0))
    for i in range(12):
        with np.load(out/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids'].tolist();labels=d['core_labels']
        pairs=observed_ray_pairs(torch.tensor(ids),policy='dyadic_v2').numpy()
        a,b=labels[pairs[:,0]],labels[pairs[:,1]];known=(a>=0)&(b>=0);target=np.where(known,(a==b).astype(np.int8),-1)
        edges=[(ids[a],ids[b],1. if t else -1.) for (a,b),t in zip(pairs,target) if t>=0]
        h=group_signed_relations(ids,edges);assert h==group_signed_relations(ids[::-1],edges[::-1])
        lookup=dict(zip(ids,labels.tolist()));counts=[0,0];wrong=0
        for group in h.groups:
            kinds={lookup[r] for r in group};assert -1 not in kinds;wrong+=int(len(kinds)>1)
            for k in kinds:counts[k]+=1
        row=dict(observation=i,pairs=len(pairs),same_core_pairs=int((target==1).sum()),different_core_pairs=int((target==0).sum()),unknown_pairs=int((target<0).sum()),
            groups_per_ai_core=counts,wrong_core_merges=wrong,labeled_unresolved=sum(lookup[r]>=0 for r in h.unresolved_ray_ids),order_invariant=True)
        write(out/f'artifacts/corrected_ideal_groups_{i:02d}.json',asdict(h));rows.append(row)
    write(out/'metrics/pair_correction.json',dict(observations=rows,training_steps=0,meaning='Perfect partial AI core scores, not trained prediction or complete branch oracle',training_qualified=False))
    print(json.dumps(rows))

if __name__=='__main__':main()
