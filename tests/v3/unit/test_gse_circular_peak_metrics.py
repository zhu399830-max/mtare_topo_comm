from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_circular_peak_metrics import (
    action_macro_f1,
    apply_peak_threshold,
    circular_local_maxima,
    ranked_peak_metrics,
    select_safe_threshold,
)


def test_circular_local_maxima_handles_wrap_and_plateau_deterministically():
    values = np.zeros((1, 180), dtype=np.float64)
    values[0, 179] = 0.9
    values[0, 0] = 0.8
    values[0, 40:42] = 0.7
    maxima = circular_local_maxima(values)
    assert maxima[0, 179]
    assert not maxima[0, 0]
    assert maxima[0, 41]
    assert not maxima[0, 40]


def test_safe_threshold_groups_ties_and_transfers_exactly():
    confidence = np.zeros((2, 180), dtype=np.float64)
    target = np.zeros_like(confidence, dtype=np.uint8)
    confidence[0, 10] = 0.9; target[0, 10] = 1
    confidence[0, 40] = 0.8; target[0, 40] = 1
    confidence[1, 70] = 0.8
    confidence[1, 100] = 0.7; target[1, 100] = 1
    chosen = select_safe_threshold(confidence, target, precision_floor=0.66)
    assert chosen is not None
    assert chosen["threshold"] == 0.7
    selected, metrics = apply_peak_threshold(confidence, target, chosen["threshold"])
    assert int(selected.sum()) == 4
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 1.0
    ranked = ranked_peak_metrics(confidence, target)
    assert ranked["teacher_peaks"] == 3
    assert 0.0 < ranked["average_precision"] <= 1.0


def test_action_macro_f1_penalizes_zero_peak_provisional():
    result = action_macro_f1(np.array([1, 2, 3, 0]), np.array([1, 2, 3, 1]))
    assert result["provisional_predictions"] == 1
    assert 0.0 < result["macro_f1"] < 1.0
