"""Fixed 0.5 selection and 1m greedy suppression; preserve raw diagnostics.

Suppression can merge distinct nearby structures. Both pre/post scores and all
indices are mandatory evidence. This is not verified graph association.
"""
import numpy as np
from .gse_candidate_coverage import candidate_coverage


def score_candidate_selection(positions,probabilities,expected,*,complete_region):
    p=np.asarray(probabilities)
    if p.shape!=(len(positions),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('finite probabilities required')
    # Validate geometry and completeness through the common scoring interface.
    def score(x):return candidate_coverage(expected,x,complete_region=complete_region,expected_frame='current_sensor',candidate_frame='current_sensor')
    score(positions)
    selected=np.flatnonzero(p>=.5).tolist()
    order=sorted(selected,key=lambda i:(-float(p[i]),*positions[i].tolist(),i))
    kept=[];suppressed=[]
    for i in order:
        blockers=[j for j in kept if np.linalg.norm(positions[i]-positions[j])<=1.]
        if blockers:suppressed.append(dict(candidate=i,by=blockers[0]))
        else:kept.append(i)
    return dict(raw_selected_indices=selected,kept_indices=kept,suppressed=suppressed,
                raw=score(positions[selected]),after_suppression=score(positions[kept]),
                confidence_threshold=.5,suppression_radius_m=1.,scientific_gate_pass=False)
