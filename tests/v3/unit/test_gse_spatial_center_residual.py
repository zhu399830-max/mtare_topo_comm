from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_spatial_center_residual import (
    cross_traversal_pairs,
    pair_distance_metrics,
)


def test_cross_traversal_pairs_exclude_same_trace_and_preserve_identity():
    identity = np.asarray(["a", "a", "a", "b", "b", "c"])
    traversal = np.asarray(["x", "x", "y", "u", "v", "only"])
    pairs = cross_traversal_pairs(identity, traversal)
    # Identity c is valid inventory but cannot define a cross-traversal pair.
    assert pairs.identity_names == ("a", "b")
    assert pairs.pair_index.tolist() == [[0, 2], [1, 2], [3, 4]]
    assert pairs.identity_code.tolist() == [0, 0, 1]


def test_pair_distance_metrics_macro_average_is_identity_balanced():
    identity = np.asarray(["a", "a", "a", "b", "b"])
    traversal = np.asarray(["x", "x", "y", "u", "v"])
    centers = np.asarray([[0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], [6, 0, 0]])
    metrics = pair_distance_metrics(centers, cross_traversal_pairs(identity, traversal), threshold_m=4)
    assert metrics["pair_count"] == 3
    assert metrics["pair_within_fraction"] == 2 / 3
    assert metrics["identity_macro_within_fraction"] == 0.5
    assert metrics["identity_macro_mean_distance_m"] == 4.0
