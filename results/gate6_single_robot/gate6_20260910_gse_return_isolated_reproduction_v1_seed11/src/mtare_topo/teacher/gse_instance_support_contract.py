"""Loss-side support preconditions, not a teacher or instance predictor.

An explicit relation mask does not authenticate its labels. This validator
never assigns a return to its nearest anchor or invents background labels.
"""
import numpy as np


def inspect_instance_support(points, anchors, *, positive, known):
    for xyz in (points, anchors):
        if (not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] != 3
                or not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all()):
            raise ValueError('finite floating N,3 coordinates required')
    shape = (len(points), len(anchors))
    for mask in (positive, known):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError('explicit point-to-anchor positive and known masks required')
    if np.any(positive & ~known):
        raise ValueError('positive support cannot be unknown')
    counts = positive.sum(axis=0)
    return dict(
        positive_pairs=int(positive.sum()),
        negative_pairs=int((known & ~positive).sum()),
        unknown_pairs=int((~known).sum()),
        multiply_supported_points=int((positive.sum(axis=1) > 1).sum()),
        unsupported_anchor_indices=np.flatnonzero(counts == 0).tolist(),
        center_vote_supervision_available=bool(len(anchors) > 0 and np.all(counts > 0)),
        teacher_qualified=False,
        training_authorized=False,
    )
