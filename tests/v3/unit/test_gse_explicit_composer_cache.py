from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.data.gse_explicit_composer_cache import (
    EXPLICIT_CACHE_FIELDS,
    validate_explicit_composer_world_cache,
)


def _arrays(rows: int = 3) -> dict[str, np.ndarray]:
    return {
        "global_sequence_index": np.arange(rows, dtype=np.int64),
        "token_bearing_deg": np.zeros((rows, 5, 6), dtype=np.float16),
        "token_existence_logits": np.zeros((rows, 5, 6), dtype=np.float16),
        "token_opening_width_m": np.ones((rows, 5, 6), dtype=np.float16),
        "token_vertical_profile_m": np.zeros((rows, 5, 6, 4), dtype=np.float16),
        "token_geometry_uncertainty": np.ones((rows, 5, 6, 5), dtype=np.float16),
        "token_count_probability": np.full((rows, 5, 7), 1 / 7, dtype=np.float16),
        "transport_row_probability": np.full((rows, 4, 6, 7), 1 / 7, dtype=np.float16),
        "transport_reveal_probability": np.full((rows, 4, 6), 0.5, dtype=np.float16),
        "geometry": np.ones((rows, 4), dtype=np.float32),
        "observation_uncertainty": np.full(rows, 0.5, dtype=np.float16),
    }


def test_explicit_cache_accepts_only_complete_geometry_state() -> None:
    arrays = _arrays()
    result = validate_explicit_composer_world_cache(arrays, expected_rows=3)
    assert result["rows"] == 3
    assert result["fields"] == EXPLICIT_CACHE_FIELDS
    assert result["forbidden_fields_present"] == []


@pytest.mark.parametrize("forbidden", ("event_logits", "place_descriptor", "token_descriptor", "pose", "identity"))
def test_explicit_cache_rejects_every_bypass_field(forbidden: str) -> None:
    arrays = _arrays()
    arrays[forbidden] = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected"):
        validate_explicit_composer_world_cache(arrays)


def test_explicit_cache_rejects_missing_or_nonprobability_arrays() -> None:
    arrays = _arrays(); del arrays["geometry"]
    with pytest.raises(ValueError, match="missing"):
        validate_explicit_composer_world_cache(arrays)
    arrays = _arrays(); arrays["transport_reveal_probability"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="probability bounds"):
        validate_explicit_composer_world_cache(arrays)

