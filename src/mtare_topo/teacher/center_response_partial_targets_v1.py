"""Loss-side grid assignment only, not a new structural teacher.

The caller must authenticate existing positive positions and query-wise
negative evidence before invoking this helper. No data access here. It must
never enter a forward path or generate candidate cells from ground truth.
"""
import numpy as np
from mtare_topo.representation.center_response_lattice_v1 import lattice,cell_indices,STEP,SIZE


def partial_lattice_targets(positive_positions_m,*,confirmed_negative_mask):
    negative=np.asarray(confirmed_negative_mask)
    if negative.shape!=(SIZE**3,) or negative.dtype!=np.bool_:
        raise ValueError('explicit full-lattice confirmed-negative mask required')
    fixed=lattice()
    if np.any(negative & ~fixed.intersects_domain):
        raise ValueError('negative evidence outside local domain')
    ids=cell_indices(positive_positions_m)
    if len(ids)>32:raise ValueError('positive capacity exceeded; no target truncation')
    if len(np.unique(ids))!=len(ids):
        raise ValueError('two positive references share one cell; do not merge labels')
    state=np.full(SIZE**3,-1,np.int8)
    state[negative]=0
    state[ids]=1
    offsets=np.asarray(positive_positions_m,dtype=float)-fixed.centers_m[ids]
    if np.any(np.abs(offsets)>STEP/2.+1e-10):raise ValueError('cell-local target outside fixed offset range')
    return dict(state=state,positive_indices=ids,offsets_m=offsets,
                positive_count=len(ids),negative_count=int(np.count_nonzero(state==0)),
                unknown_domain_count=int(np.count_nonzero((state==-1)&fixed.intersects_domain)),
                outside_count=int(np.count_nonzero(~fixed.intersects_domain)),
                source_authenticated=False,full_annotation=False)
