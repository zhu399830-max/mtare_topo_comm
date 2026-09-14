"""Loss-only pure-patch target binding; does not qualify source evidence.

Candidates must describe each actual return, not a ray-crossing source or a
window/structure ID. The caller supplies independently bound evidence_known.
Mixed surfaces are withheld rather than assigned their majority source.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurfaceAffinityTargets:
    values: np.ndarray
    known: np.ndarray
    pure_patch: np.ndarray
    patch_reason: tuple
    structural_membership: bool = False


def bind_pure_patch_targets(point_source_candidates, evidence_known,
                            patch_point_indices, neighbor_index):
    n=len(point_source_candidates)
    if (not isinstance(evidence_known,np.ndarray) or evidence_known.dtype!=np.bool_
            or evidence_known.shape!=(n,)):
        raise ValueError('explicit per-return evidence qualification mask required')
    sources=[]
    for candidates in point_source_candidates:
        if not isinstance(candidates,(tuple,list)) or any(type(s) is not str or not s for s in candidates):
            raise ValueError('explicit source candidates per return')
        if len(set(candidates))!=len(candidates):raise ValueError('duplicate candidates')
        sources.append(tuple(candidates))
    m=len(patch_point_indices)
    if (not isinstance(neighbor_index,np.ndarray) or neighbor_index.dtype.kind not in 'iu'
            or neighbor_index.shape!=(m,8) or np.any(neighbor_index < -1)
            or np.any(neighbor_index>=m)):
        raise ValueError('original bounded eight-neighbor layout required')
    owner=[];reason=[]
    for indices in patch_point_indices:
        ids=np.asarray(indices)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=n):
            raise ValueError('original return indices required')
        if len(set(map(int,ids)))!=len(ids):raise ValueError('duplicate point within patch')
        if not len(ids):o=None;r='EMPTY_PATCH'
        elif not evidence_known[ids].all():o=None;r='UNQUALIFIED_RETURN'
        elif any(len(sources[i])!=1 for i in ids):o=None;r='AMBIGUOUS_OR_MISSING_SOURCE'
        else:
            unique={sources[i][0] for i in ids}
            o=next(iter(unique)) if len(unique)==1 else None
            r='PURE' if o is not None else 'MIXED_SOURCES'
        owner.append(o);reason.append(r)
    values=np.zeros((m,8),dtype=np.float32);known=np.zeros((m,8),dtype=bool)
    for i in range(m):
        for k,j in enumerate(neighbor_index[i]):
            if j<0 or j==i or owner[i] is None or owner[j] is None:continue
            known[i,k]=True;values[i,k]=float(owner[i]==owner[j])
    pure=np.asarray([o is not None for o in owner],dtype=bool)
    for a in (values,known,pure):a.setflags(write=False)
    return SurfaceAffinityTargets(values,known,pure,tuple(reason))
