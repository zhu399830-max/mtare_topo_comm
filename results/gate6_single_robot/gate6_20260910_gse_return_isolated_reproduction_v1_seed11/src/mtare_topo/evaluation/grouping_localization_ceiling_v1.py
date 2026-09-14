"""Threshold-free one-to-one localization ceiling, not a deployment score.

All candidates are used for the first ceiling, selected candidates for the
second; neither receives unknown/background labels. Actual scoring remains
the original partial-reference scorer. Cardinality differences avoid counting
one prediction as recovery of multiple references.
"""
import numpy as np
import torch
from .grouping_center_scoring_v1 import center_score


def localization_funnel(positions, logits, targets, actual_score):
    q=np.asarray(positions,dtype=float);t=np.asarray(targets,dtype=float).reshape(-1,3)
    original_logits=np.asarray(logits)
    logits=np.asarray(logits,dtype=float)
    if logits.shape!=(len(q),) or not np.isfinite(logits).all():raise ValueError('aligned finite logits')
    keep=np.flatnonzero(torch.as_tensor(original_logits.copy()).sigmoid().numpy()>=.5)
    if keep.tolist()!=actual_score['query_indices']:raise ValueError('selection contract differs')
    def score(x):return center_score(x,t,np.ones(len(x),bool),np.ones(len(x),bool),radius=1.)
    raw=score(q);selected=score(q[keep]);actual=actual_score['tp']
    if not 0<=actual<=selected['tp']<=raw['tp']<=len(t):raise ValueError('invalid nested recall counts')
    if len(actual_score['pairs'])!=actual:raise ValueError('actual match cardinality mismatch')
    d=np.linalg.norm(q[:,None]-t[None],axis=2)
    matched={j for _,j in actual_score['pairs']}
    allowed=~np.asarray(actual_score['coverage']['possible_unconfirmed_reference_mask'],dtype=bool)
    rows=[]
    for j,target in enumerate(t):
        near=np.flatnonzero(d[:,j]<=1.);selected_near=keep[d[keep,j]<=1.]
        permitted_near=keep[(d[keep,j]<=1.)&allowed]
        category=('DETECTED' if j in matched else 'NO_CANDIDATE_WITHIN_1M' if not len(near)
            else 'NEAR_CANDIDATE_BELOW_THRESHOLD' if not len(selected_near)
            else 'SELECTED_NEIGHBOR_EXCLUDED_BY_COVERAGE' if not len(permitted_near)
            else 'ONE_TO_ONE_COMPETITION')
        nearest=int(np.argmin(d[:,j])) if len(q) else None
        rows.append(dict(reference_index=j,target_position_m=target.tolist(),category=category,
            nearest_slot=nearest,nearest_distance_m=float(d[nearest,j]) if nearest is not None else None,
            nearest_logit=float(logits[nearest]) if nearest is not None else None,
            all_near_slots=near.tolist(),selected_near_slots=selected_near.tolist(),
            coverage_permitted_near_slots=permitted_near.tolist(),
            near_logits=logits[near].tolist(),matched_actual=j in matched))
    return dict(reference_count=len(t),candidate_count=len(q),selected_count=len(keep),
        raw_one_to_one_tp=raw['tp'],selected_one_to_one_tp=selected['tp'],actual_tp=actual,
        localization_deficit=len(t)-raw['tp'],presence_deficit=raw['tp']-selected['tp'],
        coverage_or_matching_deficit=selected['tp']-actual,
        raw_pairs=raw['pairs'],selected_pairs=[(int(keep[i]),j) for i,j in selected['pairs']],
        actual_pairs=[(int(keep[i]),j) for i,j in actual_score['pairs']],references=rows,
        raw_ceiling_is_not_detector_recall=True,unknown_not_relabelled=True)
