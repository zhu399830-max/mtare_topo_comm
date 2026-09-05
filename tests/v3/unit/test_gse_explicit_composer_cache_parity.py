from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.data.gse_explicit_composer_cache_parity import (
    BINARY16_ABSOLUTE_EPSILON,
    compare_explicit_cache_to_sealed_development,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "global_sequence_index": np.arange(2, dtype=np.int64),
        "token_bearing_deg": np.zeros((2, 5, 6), dtype=np.float16),
        "token_existence_logits": np.zeros((2, 5, 6), dtype=np.float16),
        "token_opening_width_m": np.ones((2, 5, 6), dtype=np.float16),
        "token_vertical_profile_m": np.zeros((2, 5, 6, 4), dtype=np.float16),
        "token_geometry_uncertainty": np.ones((2, 5, 6, 5), dtype=np.float16),
        "geometry": np.ones((2, 4), dtype=np.float32),
        "observation_uncertainty": np.full(2, .5, dtype=np.float16),
        "token_count_probability": np.full((2, 5, 7), np.float16(1 / 7), dtype=np.float16),
        "transport_row_probability": np.full((2, 4, 6, 7), np.float16(1 / 7), dtype=np.float16),
        "transport_reveal_probability": np.full((2, 4, 6), .5, dtype=np.float16),
    }


def test_one_binary16_epsilon_and_equal_decisions_pass() -> None:
    sealed = _arrays(); actual = {name: value.copy() for name, value in sealed.items()}
    actual["transport_row_probability"][0, 0, 0, 0] += np.float16(BINARY16_ABSOLUTE_EPSILON)
    result = compare_explicit_cache_to_sealed_development(actual, sealed)
    assert result["all_primary_fields_exact"]
    assert result["all_discrete_decisions_equal"]
    assert result["maximum_absolute_error"] == BINARY16_ABSOLUTE_EPSILON


def test_primary_drift_derived_excess_and_decision_flip_each_fail() -> None:
    sealed = _arrays(); actual = {name: value.copy() for name, value in sealed.items()}
    actual["geometry"][0, 0] += 1e-6
    with pytest.raises(RuntimeError, match="primary"):
        compare_explicit_cache_to_sealed_development(actual, sealed)

    actual = {name: value.copy() for name, value in sealed.items()}
    actual["transport_row_probability"][0, 0, 0, 0] += np.float16(2 * BINARY16_ABSOLUTE_EPSILON)
    with pytest.raises(RuntimeError, match="exceeds"):
        compare_explicit_cache_to_sealed_development(actual, sealed)

    actual = {name: value.copy() for name, value in sealed.items()}
    actual["transport_reveal_probability"][0, 0, 0] = np.float16(.499)
    with pytest.raises(RuntimeError, match="decision"):
        compare_explicit_cache_to_sealed_development(actual, sealed)

