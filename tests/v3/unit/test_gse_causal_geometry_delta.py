from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_causal_geometry_delta import (
    binary_roc_auc,
    fixed_negative_quantile_threshold,
    signed_lag_delta,
    summarize_binary_score,
)


def test_binary_auc_handles_perfect_reverse_and_tied_scores() -> None:
    assert binary_roc_auc([0.0, 0.1, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert binary_roc_auc([0.8, 0.9, 0.0, 0.1], [False, False, True, True]) == 0.0
    assert binary_roc_auc([1.0, 1.0, 1.0, 1.0], [False, False, True, True]) == 0.5


def test_binary_summary_and_fit_only_threshold_are_deterministic() -> None:
    summary = summarize_binary_score([0.0, 0.2, 0.8, 1.0], [False, False, True, True])
    assert summary.positive_count == 2
    assert summary.negative_count == 2
    assert summary.roc_auc == 1.0
    assert fixed_negative_quantile_threshold([0.0, 0.1, 0.2], false_positive_rate=0.01) == 0.2


def test_signed_delta_checks_shape_and_finiteness() -> None:
    current = np.asarray([[2.0, 4.0, 1.0, 0.2]])
    past = np.asarray([[1.0, 3.0, 2.0, 0.1]])
    assert np.allclose(signed_lag_delta(current, past), [[1.0, 1.0, -1.0, 0.1]])
    with pytest.raises(ValueError):
        signed_lag_delta(current[:, :3], past[:, :3])
