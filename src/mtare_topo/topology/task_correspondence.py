"""Shared score-to-task adapter. Similarity never certifies physical merging."""
import numpy as np


def mutual_unique_matches(scores):
    """Rows=current, columns=previous. Strict mutual positive maxima only.

    Exact ties/undefined rows remain provisional. This is a fixed diagnostic
    policy, not a calibrated association probability or deployment guarantee.
    """
    scores=np.asarray(scores,float)
    if scores.ndim!=2 or np.isinf(scores).any():raise ValueError('matrix with finite/unknown scores required')
    result={i:None for i in range(len(scores))}
    if not scores.size:return result
    safe=np.where(np.isnan(scores),-np.inf,scores)
    for i,row in enumerate(safe):
        best=row.max();columns=np.flatnonzero(row==best)
        if best<=0 or len(columns)!=1:continue
        j=int(columns[0]);rows=np.flatnonzero(safe[:,j]==safe[:,j].max())
        if len(rows)==1 and int(rows[0])==i:result[i]=j
    return result


def cosine_matrix(current,previous):
    out=np.full((len(current),len(previous)),np.nan)
    for i,a in enumerate(current):
        for j,b in enumerate(previous):
            if a is None or b is None:continue
            a=np.asarray(a,float);b=np.asarray(b,float)
            if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('descriptor shape/values')
            norm=np.linalg.norm(a)*np.linalg.norm(b)
            if norm>0:out[i,j]=np.clip(a@b/norm,-1,1)
    return out
