"""Fixed-position geometric supervision; no teacher information in forward."""
import numpy as np
import torch
from torch.nn import functional as F
from mtare_topo.evaluation.gse_synthetic_field_scoring import match_positions


def candidate_targets(positions, expected, *, complete_region):
    if type(complete_region)!=bool:raise ValueError('explicit completeness required')
    for x in (positions,expected):
        if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('finite N,3 geometry required')
    match=match_positions(expected,positions,1.)
    target=np.zeros(len(positions),dtype=np.float32)
    known=np.full(len(positions),complete_region,dtype=bool)
    # A local support heatmap, not arbitrary identity among duplicate proposals.
    # Within1m confidence target decays linearly with distance; identical
    # candidates receive identical targets. Independent scoring stays one-to-one.
    if len(expected) and len(positions):
        distance=np.linalg.norm(positions[:,None]-expected[None],axis=-1).min(axis=1)
        target=np.maximum(1.-distance,0.).astype(np.float32)
        known|=target>0
    return target,known,match['fn']


def balanced_candidate_loss(logits,target,known):
    if (logits.shape!=target.shape or known.shape!=logits.shape or known.dtype!=torch.bool
            or logits.dtype!=target.dtype or any(x.device!=logits.device for x in (target,known))
            or not torch.isfinite(logits).all() or not torch.isfinite(target).all()
            or ((target<0)|(target>1)).any()):raise ValueError('masked candidate targets in[0,1] required')
    losses=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    positive=known&(target>0);negative=known&(target==0)
    terms=[losses[m].mean() for m in (positive,negative) if m.any()]
    return (torch.stack(terms).mean() if terms else logits.sum()*0),dict(positive=int(positive.sum()),negative=int(negative.sum()))
