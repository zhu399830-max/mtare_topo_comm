"""Independent joint reference-node/direction diagnostic; no graph claims."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .branch_direction_scoring import score_direction_set


def score_anchor_branches(predicted_positions, predicted_directions, reference_positions,
                          reference_directions, *, anchor_scoreable,
                          association_allowed, anchor_reference_complete,
                          branch_reference_complete, matching_radius_m, matching_angle_deg):
    """Inputs contain already-selected predictions, not confidence-aided matches.

    Masks must come from independently bound observation/reference evidence.
    This pure scorer validates layout, not provenance. Unknown-reference
    association conflicts must be reflected in association_allowed.
    """
    if (type(anchor_reference_complete) is not bool or not np.isfinite(matching_radius_m)
            or not 0 < matching_radius_m <= 10):
        raise ValueError('explicit reference completeness and position radius required')
    positions=[]
    for value in (predicted_positions,reference_positions):
        a=np.asarray(value,dtype=float)
        if (a.ndim!=2 or a.shape[1]!=3 or len(a)>32 or not np.isfinite(a).all()
                or np.any(np.linalg.norm(a,axis=1)>10.+1e-6)):
            raise ValueError('bounded finite local anchor positions required')
        positions.append(a)
    p,t=positions
    scoreable=np.asarray(anchor_scoreable);allowed=np.asarray(association_allowed)
    if any(a.shape!=(len(p),) or a.dtype!=np.bool_ for a in (scoreable,allowed)):
        raise ValueError('explicit per-prediction boolean coverage masks required')
    if (len(predicted_directions)!=len(p) or len(reference_directions)!=len(t)
            or len(branch_reference_complete)!=len(t)
            or any(type(v) is not bool for v in branch_reference_complete)):
        raise ValueError('aligned direction sets and completeness required')
    empty=np.empty((0,3))
    # Validate every direction, including those under missed/unknown anchors.
    for directions in (*predicted_directions,*reference_directions):
        score_direction_set(directions,empty,reference_complete=False,matching_angle_deg=matching_angle_deg)
    distance=np.linalg.norm(p[:,None,:]-t[None,:,:],axis=2)
    eligible=(distance<=matching_radius_m)&allowed[:,None]
    penalty=float(min(len(p),len(t))+1)
    cost=distance/21.+penalty*(~eligible)
    row,col=linear_sum_assignment(cost)
    pairs=[(int(i),int(j)) for i,j in zip(row,col) if eligible[i,j]]
    # Reject competing optimal geometric correspondences rather than using
    # branch labels to break a tie in favour of an improved semantic score.
    best=float(cost[row,col].sum())
    for i,j in pairs:
        alternative=cost.copy();alternative[i,j]=np.inf
        try:
            ar,ac=linear_sum_assignment(alternative)
        except ValueError:
            continue
        other=float(alternative[ar,ac].sum())
        if abs(other-best)<=64*np.finfo(float).eps*max(1.,abs(best)):
            raise ValueError('ambiguous geometric anchor correspondence')
    matched_p={i for i,j in pairs};matched_t={j for i,j in pairs}
    anchor_fp=sum(bool(scoreable[i]) for i in range(len(p)) if i not in matched_p)
    anchor_unknown=sum(not bool(scoreable[i]) for i in range(len(p)) if i not in matched_p)
    branches=dict(tp=0,known_fp=0,fn=0,unresolved_predictions=0)
    errors=[]
    for i,j in pairs:
        result=score_direction_set(predicted_directions[i],reference_directions[j],
            reference_complete=branch_reference_complete[j],matching_angle_deg=matching_angle_deg)
        branches['tp']+=result['tp'];branches['known_fp']+=result['fp'] or 0
        branches['fn']+=result['fn'];branches['unresolved_predictions']+=result['unresolved_predictions']
        errors.extend(result['matched_angle_errors_deg'])
    # A missed node misses ALL observed branches under it, not zero branches.
    branches['fn']+=sum(len(reference_directions[j]) for j in range(len(t)) if j not in matched_t)
    for i in range(len(p)):
        if i not in matched_p:
            field='known_fp' if scoreable[i] else 'unresolved_predictions'
            branches[field]+=len(predicted_directions[i])
    full=anchor_reference_complete and all(branch_reference_complete) and not anchor_unknown and not branches['unresolved_predictions']
    denominator=2*branches['tp']+branches['known_fp']+branches['fn']
    branches['f1']=2*branches['tp']/denominator if full and denominator else None
    total_reference=sum(map(len,reference_directions))
    branches['observed_reference_recall']=branches['tp']/total_reference if total_reference else None
    return dict(anchors=dict(tp=len(pairs),known_fp=anchor_fp,fn=len(t)-len(pairs),
                            unresolved_predictions=anchor_unknown),
                branches=branches,anchor_pairs=pairs,matched_direction_errors_deg=errors,
                complete_reference_score=full,provenance_verified=False,graph_score=False)
