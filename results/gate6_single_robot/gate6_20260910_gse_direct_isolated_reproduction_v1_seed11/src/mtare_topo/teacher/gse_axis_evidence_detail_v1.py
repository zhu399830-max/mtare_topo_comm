"""Lossless diagnostic evidence for an existing axis-support decision.

No relabeling or re-evaluation: explain which cells lack ray support and which
are merely numerical boundary alternatives. UNKNOWN is not an obstacle label.
"""
import numpy as np

from mtare_topo.representation.gse_surface_ray_evidence_v1 import query_patch_gaps


def axis_evidence_detail(grid, support):
    query_patch_gaps(grid, np.empty((0,3), dtype=np.float64),
                    np.empty((0,8), dtype=np.int64), np.empty((0,8), dtype=bool))
    if support.grid_content_sha256 != grid.content_sha256:
        raise ValueError("axis support and ray grid differ")
    cells=np.asarray(support.cell_indices,dtype=np.int64)
    if np.any(cells<0) or np.any(cells>=80**3) or len(set(cells.tolist()))!=len(cells):
        raise ValueError("invalid unique axis cells")
    state=grid.state.reshape(-1)[cells]
    if (int(np.count_nonzero(state==1))!=support.free_cells or
        int(np.count_nonzero(state==2))!=support.occupied_cells or
        int(np.count_nonzero(state==0))!=support.unknown_cells):
        raise ValueError("axis summary differs from grid")
    ambiguous=set(support.ambiguous_cells)
    if not ambiguous.issubset(set(cells.tolist())):
        raise ValueError("ambiguity outside inspected cells")
    free=grid.free_frame_bits.reshape(-1); occupied=grid.occupied_frame_bits.reshape(-1)
    return dict(status_unchanged=support.status,grid_content_sha256=grid.content_sha256,
        grid_numerical_bound_m=grid.numerical_bound_m,semantic_label=None,
        cells=[dict(index=int(c),ijk=[int(c//6400),int(c//80%80),int(c%80)],
                    state=int(s),free_frame_bits=int(free[c]),occupied_frame_bits=int(occupied[c]),
                    query_boundary_ambiguous=int(c) in ambiguous,
                    segment_indices=[i for i,group in enumerate(support.segment_cells) if int(c) in group])
               for c,s in zip(cells,state)],
        limitation="Zero evidence does not distinguish occlusion, angular sparsity, or withheld boundary rays; inspect original rays for that attribution.")
