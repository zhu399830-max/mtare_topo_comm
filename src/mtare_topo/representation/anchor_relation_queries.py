"""All unordered anchor-observation pairs, excluding self pairs."""
import numpy as np

def complete_anchor_pairs(ray_ids, anchor_ids):
    ids=np.asarray(ray_ids);anchors=np.asarray(anchor_ids)
    if ids.ndim!=1 or anchors.ndim!=1 or len(np.unique(ids))!=len(ids) or len(np.unique(anchors))!=len(anchors):raise ValueError('unique one-dimensional identities')
    lookup={int(r):i for i,r in enumerate(ids)}
    if any(int(a) not in lookup for a in anchors):raise ValueError('foreign anchor')
    rows=[];all_indices=np.arange(len(ids),dtype=np.int64)
    for a in anchors:
        i=lookup[int(a)];others=all_indices[all_indices!=i]
        rows.append(np.stack((np.minimum(i,others),np.maximum(i,others)),1))
    return np.unique(np.concatenate(rows),axis=0) if rows else np.empty((0,2),dtype=np.int64)
