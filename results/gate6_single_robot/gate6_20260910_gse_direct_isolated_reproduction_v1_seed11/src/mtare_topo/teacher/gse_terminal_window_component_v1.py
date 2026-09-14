"""Necessary source-geometry condition, never sufficient membership evidence.

Use every source-axis ROI crossing, including unobserved crossings. Filtering
to detected openings first would attach a terminal to a later re-entry branch.
The returned component is not free-space, ground support, or a positive label.
"""
import numpy as np

from .gse_roi_crossings_v1 import roi_crossings


def terminal_window_component(points_m, *, endpoint_index, center_m):
    if type(endpoint_index) is not int or endpoint_index not in (0, 1):
        raise ValueError('explicit source endpoint 0 or 1 required')
    roots = roi_crossings(points_m, center_m=center_m)
    points = np.asarray(points_m, dtype=np.float64)
    center = np.asarray(center_m, dtype=np.float64)
    result = dict(status='UNKNOWN', opening_reference_arc_m=None,
                  opening_position_m=None, source_arc_interval_m=None,
                  membership=None, physical_traversability=False)
    if roots.ambiguous_segment_indices:
        return dict(result, reason='AMBIGUOUS_SOURCE_ROI_CROSSING')
    endpoint = points[0 if endpoint_index == 0 else -1]
    if np.linalg.norm(endpoint-center) >= roots.radius_m:
        return dict(result, reason='SOURCE_ENDPOINT_NOT_INSIDE_ROI')
    if not len(roots.source_arc_m):
        return dict(result, reason='NO_WINDOW_CROSSING_FROM_TERMINAL_COMPONENT')
    index = int(np.argmin(roots.source_arc_m) if endpoint_index == 0
                else np.argmax(roots.source_arc_m))
    crossing = float(roots.source_arc_m[index])
    endpoint_arc = (0. if endpoint_index == 0 else
                    float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    return dict(result, status='REFERENCE_COMPONENT_ONLY',
                reason='FIRST_ROI_EXIT_FROM_SOURCE_ENDPOINT',
                opening_reference_arc_m=crossing,
                opening_position_m=roots.positions_m[index].tolist(),
                source_arc_interval_m=sorted([endpoint_arc, crossing]))
