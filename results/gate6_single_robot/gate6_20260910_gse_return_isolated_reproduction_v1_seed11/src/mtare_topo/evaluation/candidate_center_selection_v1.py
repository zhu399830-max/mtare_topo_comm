"""Score-only XYZ greedy suppression, never connected components or GT filtering."""
import numpy as np
import torch

def select_candidates(positions,logits):
    xyz=np.asarray(positions,dtype=float);a=np.asarray(logits)
    if xyz.shape!=(len(a),3) or not np.isfinite(xyz).all() or not np.isfinite(a).all():raise ValueError('finite aligned candidates')
    probability=torch.as_tensor(a.copy()).sigmoid().numpy()
    before=np.flatnonzero(probability>=.5).tolist()
    order=sorted(before,key=lambda i:(-float(probability[i]),*xyz[i].tolist(),i))
    kept=[];suppressed=[]
    for i in order:
        blockers=[j for j in kept if np.linalg.norm(xyz[i]-xyz[j])<=2.]
        if blockers:
            j=blockers[0];suppressed.append(dict(candidate=i,suppressor=j,distance_m=float(np.linalg.norm(xyz[i]-xyz[j]))))
        else:kept.append(i)
    return dict(raw_count=len(a),threshold=.5,radius_m=2.,before=before,after=kept,suppressed=suppressed,
        threshold_count=len(before),kept_count=len(kept),gt_filter=False)
