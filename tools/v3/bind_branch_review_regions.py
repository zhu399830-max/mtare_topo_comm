"""Bind AI-selected observation surface cores; no qualification or optimizer."""
from ai_branch_review import ROOT,RUN,CARD,CACHE,sha,write
import json
import numpy as np
import torch
from mtare_topo.representation.branch_relation_learning import observed_ray_pairs

def main():
    out=ROOT/RUN;card=json.loads((ROOT/CARD).read_text());review=json.loads((out/'artifacts/review.json').read_text());rows=[]
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='RUNNING'
    write(out/'config/region_binding.json',dict(code='tools/v3/bind_branch_review_regions.py',sha256=sha(ROOT/'tools/v3/bind_branch_review_regions.py'),review_sha256=sha(out/'artifacts/review.json'),qualification=False))
    for i,(e,r) in enumerate(zip(card['scope']['entries'],review['observations'])):
        with np.load(ROOT/e['student_path'],allow_pickle=False) as d:valid=d['valid_mask'][-1].astype(bool).reshape(-1)
        with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'][4*11520:5*11520]
        ids=np.flatnonzero(valid);x=xyz[ids]
        membership=np.stack([((x>=b['min_m'])&(x<=b['max_m'])).all(1) for b in r['regions']],1)
        labels=np.where(membership.sum(1)==1,membership.argmax(1),-1)
        pairs=observed_ray_pairs(torch.from_numpy(ids).long()).numpy()
        a=labels[pairs[:,0]];b=labels[pairs[:,1]];known=(a>=0)&(b>=0)
        targets=np.where(known,(a==b).astype(np.int8),-1)
        np.savez_compressed(out/f'artifacts/ai_core_{i:02d}.npz',ray_ids=ids,core_labels=labels,pairs=pairs,conditional_ai_targets=targets)
        row=dict(observation=i,core_ray_counts=[int((labels==j).sum()) for j in range(len(r['regions']))],unknown_rays=int((labels<0).sum()),
            same_core_pairs=int((targets==1).sum()),different_core_pairs=int((targets==0).sum()),unknown_pairs=int((targets<0).sum()),
            all_possible_cross_core_pairs=int((labels==0).sum()*(labels==1).sum()),training_qualified=False)
        rows.append(row)
    write(out/'metrics/core_binding.json',dict(observations=rows,meaning='AI core relations only; sparse pair graph is original, not GT-selected',training_steps=0))
    print(json.dumps(rows))

if __name__=='__main__':main()
