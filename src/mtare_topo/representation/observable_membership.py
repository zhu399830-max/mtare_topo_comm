"""Partial positive support is not an exclusive segmentation label."""
import numpy as np

def bind_observable_support(xyz, valid, boxes):
    xyz=np.asarray(xyz);valid=np.asarray(valid)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or valid.shape!=(len(xyz),) or valid.dtype!=bool:
        raise ValueError('point/valid shape')
    if not boxes:raise ValueError('at least one support region')
    hits=[]
    usable=valid&np.isfinite(xyz).all(axis=1)
    for box in boxes:
        lo=np.asarray(box['min_m']);hi=np.asarray(box['max_m'])
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or (lo>hi).any():
            raise ValueError('finite ordered box')
        hits.append(usable&((xyz>=lo)&(xyz<=hi)).all(axis=1))
    hits=np.stack(hits,axis=1)
    # Geometric overlap does not establish shared semantic membership.
    unique=hits.sum(axis=1)==1
    support=np.full(hits.shape,-1,dtype=np.int8)
    support[hits&unique[:,None]]=1
    return dict(support=support,known=support==1,overlap_unresolved=hits.sum(axis=1)>1,
                unassigned=~(support==1).any(axis=1),valid=usable)
