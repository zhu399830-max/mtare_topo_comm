"""Reference-conditioned duplicate coverage, separate from observed background.

An independently confirmed target supplies structure evidence; its centre need
not be a LiDAR return. This does not certify the whole physical scene. Unknown
reference competitors still veto coverage, and empty space still needs original
ray evidence. V1 remains available for immutable historical comparisons.
"""
import numpy as np
from .gse_reference_query_coverage_v1 import reference_query_coverage as previous
from .gse_reference_query_coverage_v1 import _bound_query_coverage


def reference_query_coverage(**kwargs):
    result = previous(**kwargs)  # retain all validation and background evidence
    q = np.asarray(kwargs['query_xyz_m'], dtype=float)
    refs = np.asarray(kwargs['all_reference_xyz_m'], dtype=float).reshape(-1,3)
    indices = kwargs['confirmed_reference_indices']
    supported = (np.linalg.norm(q[:,None,:]-refs[None,indices,:],axis=2)
                 <= kwargs['matching_radius_m']).any(axis=1)
    inside = np.linalg.norm(q-np.asarray(kwargs['score_center_m']),axis=1)<=kwargs['score_radius_m']
    conflict = np.asarray(result['possible_unconfirmed_reference_mask'],dtype=bool)
    structure = inside & supported & ~conflict
    result.update(query_scoreable_mask=(np.asarray(result['query_scoreable_mask'],dtype=bool)|structure).tolist(),
        confirmed_structure_coverage_mask=structure.tolist(),
        coverage_definition='v2_observed_background_or_confirmed_structure_without_unknown_competitor')
    return result


def bound_anchor_query_coverage(bundle, grid, query_xyz_m, **kwargs):
    return _bound_query_coverage(bundle,grid,query_xyz_m,kind='anchors',
        coverage_function=reference_query_coverage,**kwargs)


def bound_opening_query_coverage(bundle, grid, query_xyz_m, **kwargs):
    return _bound_query_coverage(bundle,grid,query_xyz_m,kind='openings',
        coverage_function=reference_query_coverage,**kwargs)
