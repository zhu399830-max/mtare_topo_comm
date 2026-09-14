"""Bounded geometry-only detection matching with explicit partial-label limits.

Caller supplies a frozen threshold and an independently qualified score region.
This function cannot establish annotation completeness or training eligibility.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def detection_counts(*, predicted_xyz_m, confidence, target_xyz_m, threshold,
                     maximum_error_m, score_center_m, score_radius_m, region_complete):
    p=np.asarray(predicted_xyz_m,dtype=np.float64)
    t=np.asarray(target_xyz_m,dtype=np.float64)
    c=np.asarray(confidence,dtype=np.float64)
    center=np.asarray(score_center_m,dtype=np.float64)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or t.ndim!=2 or t.shape[1:]!=(3,)
            or len(p)>64 or len(t)>64 or c.shape!=(len(p),) or center.shape!=(3,)
            or any(not np.isfinite(x).all() for x in (p,t,c,center))
            or np.any((c<0)|(c>1)) or type(region_complete) is not bool
            or not np.isfinite([threshold,maximum_error_m,score_radius_m]).all()
            or not 0<=threshold<=1 or maximum_error_m<=0 or not 0<score_radius_m<=10):
        raise ValueError('finite64-capacity positions and explicit frozen metric/region required')
    if len({tuple(x) for x in t})!=len(t):
        raise ValueError('duplicate target positions must be resolved by label producer')
    selected=np.flatnonzero(c>=threshold)
    # Content ordering for query permutations; confidence never enters geometry cost.
    selected=np.array(sorted(selected,key=lambda i:tuple(p[i])),dtype=int)
    order=np.array(sorted(range(len(t)),key=lambda i:tuple(t[i])),dtype=int)
    matches=[]
    if len(selected) and len(order):
        distances=np.linalg.norm(p[selected,None,:]-t[None,order,:],axis=2)
        valid=distances<=maximum_error_m
        # One invalid edge outweighs the sum of all possible valid distances.
        # Thus maximize valid cardinality first, then minimize geometric error.
        penalty=min(distances.shape)+1.
        cost=np.where(valid,distances/maximum_error_m,penalty)
        rows,cols=linear_sum_assignment(cost)
        matches=[(int(selected[i]),int(order[j]),float(distances[i,j]))
                 for i,j in zip(rows,cols) if valid[i,j]]
    matched={i for i,_,_ in matches}
    unmatched=[int(i) for i in selected if i not in matched]
    inside=np.linalg.norm(p-center,axis=1)<=score_radius_m
    known_fp=[i for i in unmatched if region_complete and inside[i]]
    unknown=[i for i in unmatched if not region_complete or not inside[i]]
    tp=len(matches);fn=len(t)-tp
    # A conditional score is not a full-scene success metric. Never provide a
    # full F1 when any selected prediction or target lies outside qualified area.
    all_targets_inside=bool(np.all(np.linalg.norm(t-center,axis=1)<=score_radius_m))
    all_predictions_inside=bool(np.all(inside[selected]))
    full=region_complete and all_targets_inside and all_predictions_inside
    denom=2*tp+len(known_fp)+fn
    return dict(true_positive=tp,false_negative=fn,known_region_false_positive=len(known_fp),
        unknown_unmatched_predictions=len(unknown),selected_predictions=len(selected),targets=len(t),
        matches=matches,known_false_positive_indices=known_fp,unknown_prediction_indices=unknown,
        full_f1=(2*tp/denom if denom else None) if full else None,
        full_scoring_available=full,region_completeness_supplied_not_verified=True,
        target_recall=tp/len(t) if len(t) else None,scientific_gate_pass=False)
