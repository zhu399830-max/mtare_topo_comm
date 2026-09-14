"""Reference competition without same-source incidental-section duplicates.

Nonowning contours remain stored and can still block a different source.
Opposite arcs and all owning references remain independent competitors.
"""
from copy import deepcopy
from .gse_window_opening_diagnostic_v1 import _diagnose_observation
from .gse_window_opening_diagnostic_v2 import propose_bound_window_openings


def resolve_reference_competition(proposals):
    rows = deepcopy(proposals)
    for i, row in enumerate(rows):
        if type(row.get('axis_reference_owned')) is not bool:
            raise ValueError('explicit contour-reference ownership required')
        competitors = set()
        excluded = []
        for j, other in enumerate(rows):
            if i == j:
                continue
            if type(other.get('axis_reference_owned')) is not bool:
                raise ValueError('explicit contour-reference ownership required')
            if (other['primitive_id_teacher_only'] == row['primitive_id_teacher_only']
                    and not other['axis_reference_owned']):
                excluded.append(j)
            else:
                competitors.update(other['geometric_outward_crossing_ray_indices'])
        row['same_source_nonreference_contour_indices'] = excluded
        row['competing_crossing_ray_indices'] = [r for r in row['outward_crossing_ray_indices'] if r in competitors]
        row['exclusive_outward_crossing_ray_indices'] = [r for r in row['outward_crossing_ray_indices'] if r not in competitors]
        row['proposal_supported_after_competition'] = bool(row['exclusive_outward_crossing_ray_indices'] and row['surface_return_ray_indices'])
    return rows


def diagnose_observation(bundle):
    return _diagnose_observation(bundle, propose_bound_window_openings, resolve_reference_competition)
