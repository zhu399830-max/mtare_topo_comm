from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_slope_risk_calibration import (
    apply_residual_scale,
    select_maximin_residual_scale,
)


def _seed(target, corrected):
    target = np.asarray(target, dtype=np.float32)
    return {
        "target_slope_deg": target,
        "five_frame_prior_slope_deg": np.zeros_like(target),
        "corrected_slope_deg": np.asarray(corrected, dtype=np.float32),
        "parent_id": np.asarray(["world_a", "world_a", "world_b", "world_b"]),
    }


def test_maximin_calibration_selects_shared_under_relaxation():
    # Both parents have target 1 and an over-aggressive residual of 2, so the
    # unique grid optimum is a one-half residual scale.
    rows = [_seed([1, 1, 1, 1], [2, 2, 2, 2]) for _ in range(3)]
    result = select_maximin_residual_scale(rows, grid_denominator=100)
    assert result.residual_scale == 0.5
    assert result.grid_index == 50
    assert result.worst_parent_relative_improvement == pytest.approx(1.0)
    assert set(result.parent_relative_improvement) == {"world_a", "world_b"}


def test_apply_residual_scale_preserves_endpoints_and_dtype():
    prior = np.asarray([-2.0, 3.0], dtype=np.float32)
    corrected = np.asarray([2.0, 1.0], dtype=np.float32)
    assert np.array_equal(apply_residual_scale(prior, corrected, 0.0), prior)
    assert np.array_equal(apply_residual_scale(prior, corrected, 1.0), corrected)
    assert np.allclose(apply_residual_scale(prior, corrected, 0.25), [-1.0, 2.5])
    assert apply_residual_scale(prior, corrected, 0.25).dtype == np.float32


def test_calibration_rejects_parent_or_scale_drift():
    rows = [_seed([1, 1, 1, 1], [2, 2, 2, 2]) for _ in range(3)]
    rows[2] = dict(rows[2])
    rows[2]["parent_id"] = np.asarray(["world_a"] * 4)
    with pytest.raises(ValueError, match="same non-empty parent set"):
        select_maximin_residual_scale(rows)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        apply_residual_scale(np.asarray([0.0]), np.asarray([1.0]), 1.1)
