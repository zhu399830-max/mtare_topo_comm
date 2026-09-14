"""Conditional reference counts; never substitutes for full-scene F1."""
import numpy as np
from .gse_surface_detection_metrics_v1 import detection_counts


def covered_detection_counts(*, query_scoreable_mask, **kwargs):
    if 'region_complete' in kwargs:
        raise ValueError('query coverage cannot assert whole-region completeness')
    result = detection_counts(**kwargs, region_complete=False)
    mask = np.asarray(query_scoreable_mask)
    p = np.asarray(kwargs['predicted_xyz_m'])
    if mask.dtype != bool or mask.shape != (len(p),):
        raise ValueError('one explicit boolean coverage value per original query required')
    inside = np.linalg.norm(p-np.asarray(kwargs['score_center_m']),axis=1)<=kwargs['score_radius_m']
    if np.any(mask & ~inside):
        raise ValueError('coverage cannot extend beyond declared scoring sphere')
    unmatched = result['unknown_prediction_indices']
    known = [i for i in unmatched if mask[i]]
    unknown = [i for i in unmatched if not mask[i]]
    selected = np.asarray(kwargs['confidence'])>=kwargs['threshold']
    return dict(result, known_region_false_positive=len(known),
        known_false_positive_indices=known, unknown_unmatched_predictions=len(unknown),
        unknown_prediction_indices=unknown,
        selected_queries_with_reference_coverage=int(np.sum(selected & mask)),
        selected_queries_without_reference_coverage=int(np.sum(selected & ~mask)),
        reference_coverage_supplied_not_verified=True, full_f1=None,
        full_scoring_available=False, scientific_gate_pass=False)


def score_bound_anchor_references(*, bundle, grid, produced_targets, manifest_row,
                                   predicted_xyz_m, confidence, threshold):
    from mtare_topo.teacher.gse_reference_query_coverage_v1 import bound_anchor_query_coverage
    record = produced_targets['record']; region = record['score_region']
    rows = {}
    for radius in (1., 2., 4.):
        coverage = bound_anchor_query_coverage(bundle, grid, predicted_xyz_m,
            produced_targets=produced_targets, manifest_row=manifest_row, matching_radius_m=radius)
        counts = covered_detection_counts(query_scoreable_mask=coverage['query_scoreable_mask'],
            predicted_xyz_m=predicted_xyz_m, confidence=confidence,
            target_xyz_m=np.asarray([a['position_m'] for a in record['anchors']],dtype=float).reshape(-1,3),
            threshold=threshold, maximum_error_m=radius,
            score_center_m=region['center_m'], score_radius_m=region['radius_m'])
        counts['reference_coverage_supplied_not_verified'] = False
        rows[str(radius)] = dict(counts=counts, coverage=coverage)
    return dict(anchor_main_4m=rows['4.0'], anchor_sensitivity_1m=rows['1.0'],
        anchor_sensitivity_2m=rows['2.0'], threshold=threshold,
        calibration_verified=False, full_f1=None, scientific_gate_pass=False)


def score_bound_opening_references(*, bundle, grid, produced_targets, manifest_row,
                                    predicted_xyz_m, confidence, threshold, maximum_error_m):
    """Opening radius is supplied by its own frozen contract, never anchor4m."""
    from mtare_topo.teacher.gse_reference_query_coverage_v1 import bound_opening_query_coverage
    coverage=bound_opening_query_coverage(bundle,grid,predicted_xyz_m,
        produced_targets=produced_targets,manifest_row=manifest_row,matching_radius_m=maximum_error_m)
    record=produced_targets['record']; region=record['score_region']
    counts=covered_detection_counts(query_scoreable_mask=coverage['query_scoreable_mask'],
        predicted_xyz_m=predicted_xyz_m,confidence=confidence,
        target_xyz_m=np.asarray([o['position_m'] for o in record['openings']],dtype=float).reshape(-1,3),
        threshold=threshold,maximum_error_m=maximum_error_m,
        score_center_m=region['center_m'],score_radius_m=region['radius_m'])
    counts['reference_coverage_supplied_not_verified']=False
    return dict(counts=counts,coverage=coverage,maximum_error_m=maximum_error_m,
        threshold=threshold,calibration_verified=False,full_f1=None,scientific_gate_pass=False)
