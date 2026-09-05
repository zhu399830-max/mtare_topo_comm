from __future__ import annotations

import json

import numpy as np
import pytest

from mtare_topo.data.gse_spatial_event_center_cache import (
    SpatialEventCenterCache,
    compact_five_frame_references,
)


def test_compact_five_frame_references_is_exact():
    frames = np.asarray([2, 3, 7, 8, 9, 10], dtype=np.int64)
    references = np.asarray([[2, 3, 7, 8, 9], [3, 7, 8, 9, 10]], dtype=np.int64)
    np.testing.assert_array_equal(
        compact_five_frame_references(references, frames),
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
    )
    with pytest.raises(RuntimeError, match="absent"):
        compact_five_frame_references(np.asarray([[1, 2, 3, 7, 8]]), frames)


def test_spatial_cache_gathers_exact_five_frames(tmp_path):
    manifest = {
        "schema_version": "gse_spatial_event_center_cache_v1",
        "history_frames": 5, "encoder_dim": 128,
        "directional_bins": 36, "elevation_bins": 2,
        "unique_frames": 7, "causal_observations": 2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "pooled.npy", np.arange(7 * 128, dtype=np.float16).reshape(7, 128))
    np.save(tmp_path / "azimuth.npy", np.zeros((7, 128, 36), dtype=np.float16))
    np.save(tmp_path / "elevation.npy", np.zeros((7, 128, 2), dtype=np.float16))
    np.save(tmp_path / "compact_reference_index.npy", np.asarray([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]], dtype=np.int32))
    cache = SpatialEventCenterCache(tmp_path)
    batch = cache.gather(np.asarray([1]))
    assert batch["pooled"].shape == (1, 5, 128)
    assert batch["azimuth"].shape == (1, 5, 128, 36)
    assert batch["elevation"].shape == (1, 5, 128, 2)
    assert batch["pooled"][0, 0, 0] == np.float32(2 * 128)
