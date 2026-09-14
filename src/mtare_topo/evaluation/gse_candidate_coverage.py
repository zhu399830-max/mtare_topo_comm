"""Geometry-only proposal diagnostics, NOT detector or topology qualification.

Candidate production is external and must not receive scoring targets. Every
candidate is retained; no confidence sorting, GT cropping, or top-k selection.
Incomplete annotations permit coverage, not precision or false-positive claims.
"""
import numpy as np

from .gse_synthetic_field_scoring import match_positions


def _positions(value):
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 3
            or value.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(value).all()):
        raise ValueError('finite floating N,3 coordinates required')
    return value.astype(np.float64, copy=False)


def candidate_coverage(expected, candidates, *, complete_region,
                       expected_frame, candidate_frame):
    """Score 1/2/4m correspondence with explicit common coordinate frame.

    Coverage allows several targets to be close to one candidate, while the
    one-to-one recall does not. Reporting both exposes proposal collapse.
    Empty-target recall and zero-candidate precision are undefined, not 1.
    Completeness is a caller-bound annotation contract, not inferred here.
    """
    x, y = _positions(expected), _positions(candidates)
    if type(complete_region) is not bool:
        raise ValueError('explicit boolean annotation completeness required')
    if (not isinstance(expected_frame, str) or not expected_frame.strip()
            or expected_frame != candidate_frame):
        raise ValueError('explicit identical coordinate frames required')
    nearest = (np.linalg.norm(x[:, None] - y[None, :], axis=-1).min(axis=1)
               if len(x) and len(y) else None)
    scores = {}
    for radius in (1., 2., 4.):
        matched = match_positions(x, y, radius)
        errors = [p[2] for p in matched['pairs']]
        scores[str(radius)] = dict(
            matched=matched['tp'], missed=matched['fn'],
            unmatched_candidates=matched['fp'],
            false_positives=matched['fp'] if complete_region else None,
            one_to_one_recall=matched['tp']/len(x) if len(x) else None,
            coverage_recall=(float(np.mean(nearest <= radius)) if nearest is not None
                             else (0. if len(x) else None)),
            precision=(matched['tp']/len(y) if complete_region and len(y) else None),
            f1=matched['f1'] if complete_region else None,
            matched_position_errors_m=errors,
            matched_position_mae_m=float(np.mean(errors)) if errors else None,
            pairs=matched['pairs'])
    return dict(schema='gse_candidate_coverage_v1', coordinate_frame=expected_frame,
                expected_count=len(x), candidate_count=len(y),
                complete_region=complete_region, thresholds_m=[1., 2., 4.],
                scores=scores, scientific_gate_pass=False,
                limitation='Proposal coverage is not learned detection, graph quality, '
                           'or evidence of observation-only candidate provenance.')
