"""Predeclared pilot checks; no threshold fitting, checkpoint selection or IDs.

Template uses fit predictions only, slot-wise with no missing-slot padding.
All-candidate error is a separate anti-template diagnostic, NOT detector recall.
Counterfactual scoring must keep recipient labels/grid and log donor provenance.
"""
import hashlib
import json
import numpy as np


def _positions(value):
    array=np.asarray(value)
    if (array.ndim!=2 or array.shape[1]!=3 or not 1<=len(array)<=32
            or array.dtype.kind!='f' or not np.isfinite(array).all()
            or np.any(np.linalg.norm(array,axis=1)>10.00001)):
        raise ValueError('nonempty bounded finite prediction positions required')
    return array.astype(np.float64)


def fit_template(records):
    if not records or any(r['split']!='fit' for r in records):
        raise ValueError('template construction is fit-only')
    values=[_positions(r['position_m']) for r in records]
    size=max(map(len,values))
    counts=np.asarray([sum(len(v)>i for v in values) for i in range(size)],np.int64)
    template=np.stack([np.mean([v[i] for v in values if len(v)>i],axis=0) for i in range(size)])
    return dict(position_m=template,slot_support_counts=counts,fit_observations=len(values))


def fixed_correspondence_shuffle(sources):
    """Hash-order cyclic derangement, seed0, chosen without labels/predictions."""
    if len(sources)<2:raise ValueError('at least two unique sources required')
    encoded=[json.dumps(s,sort_keys=True,separators=(',',':'),allow_nan=False) for s in sources]
    if len(set(encoded))!=len(encoded):raise ValueError('duplicate diagnostic source')
    order=sorted(range(len(sources)),key=lambda i:(hashlib.sha256(('0:'+encoded[i]).encode()).digest(),encoded[i]))
    donor=np.empty(len(sources),np.int64)
    for i,recipient in enumerate(order):donor[recipient]=order[(i+1)%len(order)]
    return donor


def decision(*,observations,anchor_tp,anchor_fn,branch_tp,branch_fn,
             anchor_known_fp,branch_known_fp,unresolved_anchors,unresolved_branches,
             candidate_mean_error_m,template_mean_error_m,shuffled_anchor_tp,shuffled_anchor_fn):
    counts=(observations,anchor_tp,anchor_fn,branch_tp,branch_fn,anchor_known_fp,branch_known_fp,
            unresolved_anchors,unresolved_branches,shuffled_anchor_tp,shuffled_anchor_fn)
    if any(type(v) is not int or v<0 for v in counts) or observations!=282:
        raise ValueError('exact282 fit population and nonnegative integer counts required')
    if anchor_tp+anchor_fn!=shuffled_anchor_tp+shuffled_anchor_fn:
        raise ValueError('shuffle must preserve recipient reference population')
    if anchor_tp+anchor_fn!=186 or branch_tp+branch_fn!=598:
        raise ValueError('frozen fit reference population must not drift')
    errors=np.asarray([candidate_mean_error_m,template_mean_error_m],float)
    if not np.isfinite(errors).all() or np.any(errors<0):raise ValueError('finite actual/template errors required')
    ar=anchor_tp/max(1,anchor_tp+anchor_fn);br=branch_tp/max(1,branch_tp+branch_fn)
    sr=shuffled_anchor_tp/max(1,shuffled_anchor_tp+shuffled_anchor_fn)
    fp=(anchor_known_fp+branch_known_fp)/observations
    advantage=template_mean_error_m-candidate_mean_error_m
    checks=dict(anchor_recall=10*anchor_tp>=9*(anchor_tp+anchor_fn),
                branch_recall=10*branch_tp>=9*(branch_tp+branch_fn),
                known_fp_per_observation=10*(anchor_known_fp+branch_known_fp)<=observations,
                anti_template=advantage>=.20,
                input_dependence=10*(anchor_tp-shuffled_anchor_tp)>=anchor_tp+anchor_fn)
    return dict(pilot_pass=all(checks.values()),checks=checks,anchor_recall=ar,branch_recall=br,
                known_fp_per_observation=fp,template_advantage_m=advantage,shuffle_recall_drop=ar-sr,
                unresolved_anchors=unresolved_anchors,unresolved_branches=unresolved_branches,
                independent_generalization=False,relation_advantage=False,graph_advantage=False)
