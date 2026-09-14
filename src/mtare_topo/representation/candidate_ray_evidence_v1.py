"""Evidence at predicted XYZ; never a detector, filter, or traversability test."""
from dataclasses import dataclass
import numpy as np
from .gse_surface_ray_evidence_v1 import (
    _xyz, _point_cells, _numerical_bound, query_patch_gaps,
)


@dataclass(frozen=True)
class CandidateRayEvidence:
    position_m: tuple
    cell_indices: tuple
    states: tuple
    free_frame_bits: tuple
    occupied_frame_bits: tuple
    boundary_ambiguous: bool
    touches_outside: bool
    grid_content_sha256: str
    structure_confirmed: bool = False
    physical_connectivity: bool = False


def candidate_ray_evidence(grid, positions_m):
    """Preserve query order and all boundary alternatives without aggregating.

    Reuse the existing gap validator on an empty query population to validate
    the grid's shape, state/bit consistency and content hash. No gaps or new
    rays are generated. Query coordinates are never clipped to the grid.
    Caller authenticates the sensor frame and numerical source binding.
    """
    query_patch_gaps(grid, np.empty((0, 3), dtype=np.float64),
                     np.empty((0, 8), dtype=np.int64),
                     np.empty((0, 8), dtype=bool))
    _xyz(positions_m, 'candidate positions')
    if len(positions_m) > 32 or not np.isfinite(positions_m).all():
        raise ValueError('at most32 finite candidate positions required')
    tolerance = max(grid.numerical_bound_m, _numerical_bound(positions_m))
    state, free, occupied = (a.reshape(-1) for a in
                            (grid.state, grid.free_frame_bits, grid.occupied_frame_bits))
    result = []
    for point in positions_m:
        cells = _point_cells(point, tolerance)
        outside = bool(np.any(np.abs(point) >= 10. - tolerance))
        result.append(CandidateRayEvidence(
            tuple(float(x) for x in point), tuple(int(i) for i in cells),
            tuple(int(state[i]) for i in cells), tuple(int(free[i]) for i in cells),
            tuple(int(occupied[i]) for i in cells), len(cells) > 1 or outside,
            outside, grid.content_sha256))
    return tuple(result)
