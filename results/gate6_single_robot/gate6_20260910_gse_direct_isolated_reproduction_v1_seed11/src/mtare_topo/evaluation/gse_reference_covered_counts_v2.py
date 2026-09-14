"""Source-bound conditional counts using confirmed structure support V2."""
import numpy as np
from .gse_reference_covered_counts_v1 import covered_detection_counts
from mtare_topo.teacher.gse_reference_query_coverage_v2 import (
    bound_anchor_query_coverage, bound_opening_query_coverage)


def score_bound_references(*, kind, bundle, grid, produced_targets, manifest_row,
                           predicted_xyz_m, confidence, threshold, maximum_error_m):
    if kind not in ('anchors','openings'):
        raise ValueError('explicit anchors or openings required')
    coverage_function = bound_anchor_query_coverage if kind == 'anchors' else bound_opening_query_coverage
    coverage = coverage_function(bundle,grid,predicted_xyz_m,produced_targets=produced_targets,
        manifest_row=manifest_row,matching_radius_m=maximum_error_m)
    record = produced_targets['record']; region = record['score_region']
    counts = covered_detection_counts(query_scoreable_mask=coverage['query_scoreable_mask'],
        predicted_xyz_m=predicted_xyz_m,confidence=confidence,
        target_xyz_m=np.asarray([x['position_m'] for x in record[kind]],dtype=float).reshape(-1,3),
        threshold=threshold,maximum_error_m=maximum_error_m,
        score_center_m=region['center_m'],score_radius_m=region['radius_m'])
    counts['reference_coverage_supplied_not_verified'] = False
    return dict(kind=kind,counts=counts,coverage=coverage,threshold=threshold,
        maximum_error_m=maximum_error_m,calibration_verified=False,full_f1=None,
        scientific_gate_pass=False)
