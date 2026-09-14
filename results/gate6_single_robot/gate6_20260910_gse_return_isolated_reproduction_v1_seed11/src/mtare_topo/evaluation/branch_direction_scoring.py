"""Independent one-anchor direction-set diagnostic, not whole-graph scoring."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def score_direction_set(predicted, observed_reference, *, reference_complete, matching_angle_deg):
    """Caller freezes selection threshold and angular tolerance before a run.

    With a partial reference set, unmatched predictions are unresolved, never
    false positives. This function cannot score missed/unmatched anchor nodes.
    """
    if type(reference_complete) is not bool or not np.isfinite(matching_angle_deg) or not 0 < matching_angle_deg < 180:
        raise ValueError('explicit completeness and finite angular tolerance required')
    arrays=[]
    for value in (predicted,observed_reference):
        a=np.asarray(value,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=3 or len(a)>64 or not np.isfinite(a).all():
            raise ValueError('bounded finite direction sets required')
        if not np.allclose(np.linalg.norm(a,axis=1),1.,rtol=0,atol=1e-6):
            raise ValueError('unit directed vectors required')
        arrays.append(a)
    p,t=arrays;angles=np.degrees(np.arccos(np.clip(p@t.T,-1.,1.)))
    # Maximize valid match count first, then minimize total angle. No training
    # assignment or prediction confidence participates in this correspondence.
    penalty=181.*(min(len(p),len(t))+1)
    row,col=linear_sum_assignment(angles+penalty*(angles>matching_angle_deg))
    pairs=[(int(i),int(j)) for i,j in zip(row,col) if angles[i,j]<=matching_angle_deg]
    tp=len(pairs);unmatched=len(p)-tp;fn=len(t)-tp
    fp=unmatched if reference_complete else 0
    denominator=2*tp+fp+fn
    return dict(tp=tp,fp=fp if reference_complete else None,fn=fn,
        unresolved_predictions=0 if reference_complete else unmatched,
        f1=(2*tp/denominator if denominator else None) if reference_complete else None,
        observed_reference_recall=tp/len(t) if len(t) else None,
        matched_angle_errors_deg=[float(angles[i,j]) for i,j in pairs],pairs=pairs,
        reference_complete=reference_complete,whole_structure_score=False)
