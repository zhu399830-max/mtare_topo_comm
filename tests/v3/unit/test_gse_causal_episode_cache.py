from __future__ import annotations

import json

import numpy as np
import pytest

from mtare_topo.data.gse_causal_episode_cache import (
    CausalEpisodeEmbeddingDataset,
    compact_global_references,
    materialize_boundary_targets,
)


def test_compact_global_references_preserves_padding_and_sparse_ids() -> None:
    refs = np.asarray([[-1] * 7 + [10, 12, 14, 20, 22], list(range(30, 42))], dtype=np.int64)
    mask = refs >= 0
    frames = np.asarray([10, 12, 14, 20, 22] + list(range(30, 42)), dtype=np.int64)
    compact = compact_global_references(refs, mask, frames)
    assert np.all(compact[0, :7] == -1)
    np.testing.assert_array_equal(compact[0, -5:], np.arange(5))
    np.testing.assert_array_equal(compact[1], np.arange(5, 17))


def test_compact_global_references_rejects_missing_frame() -> None:
    refs = np.full((1, 12), -1, dtype=np.int64)
    refs[0, -1] = 9
    with pytest.raises(RuntimeError, match="absent"):
        compact_global_references(refs, refs >= 0, np.asarray([1, 3, 5]))


def test_embedding_dataset_reads_masked_cache_and_compacts_episode_ids(tmp_path) -> None:
    frame_count, observation_count = 14, 3
    np.save(tmp_path / "pooled.npy", np.arange(frame_count * 128, dtype=np.float16).reshape(frame_count, 128))
    np.save(tmp_path / "directional.npy", np.ones((frame_count, 128, 36), dtype=np.float16))
    refs = np.full((observation_count, 12), -1, dtype=np.int32)
    mask = np.zeros_like(refs, dtype=np.bool_)
    for row, length in enumerate((5, 6, 12)):
        refs[row, -length:] = np.arange(length)
        mask[row, -length:] = True
    np.save(tmp_path / "compact_reference_index.npy", refs)
    np.save(tmp_path / "valid_history_mask.npy", mask)
    np.save(tmp_path / "episode_id.npy", np.asarray([-1, 7, 9], dtype=np.int64))
    np.save(tmp_path / "event_index.npy", np.asarray([0, 3, 4], dtype=np.int8))
    np.save(tmp_path / "boundary_offset_m.npy", np.asarray([0, 0, 4.5], dtype=np.float32))
    np.save(tmp_path / "boundary_valid.npy", np.asarray([False, False, True]))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "gse_causal_episode_spatial_cache_v1",
        "history_frames": 12, "encoder_dim": 128, "directional_bins": 36,
        "unique_frames": frame_count, "causal_observations": observation_count,
    }))
    baseline = np.zeros((observation_count, 146), dtype=np.float32)
    baseline[:, :5] = 0.2
    baseline_path = tmp_path / "baseline.npy"
    np.save(baseline_path, baseline)
    dataset = CausalEpisodeEmbeddingDataset(tmp_path, baseline_path, np.asarray([0, 2]))
    assert dataset.episode_id.tolist() == [-1, 0]
    sample = dataset[1]
    assert sample["pooled"].shape == (12, 128)
    assert sample["directional"].shape == (12, 128, 36)
    assert sample["boundary_valid"]
    np.testing.assert_allclose(np.exp(sample["baseline_event_logits"]), 0.2)


def test_boundary_targets_keep_pre_boundary_event_but_mask_its_regression() -> None:
    teacher = [
        {"traversal_id": "t0", "sequence_index": 3, "event": "geometry_transition"},
        {"traversal_id": "t0", "sequence_index": 4, "event": "geometry_transition"},
        {"traversal_id": "t0", "sequence_index": 5, "event": "corridor"},
    ]
    timing = [
        {"traversal_id": "t0", "sequence_index": 3, "metres_after_boundary": -0.5},
        {"traversal_id": "t0", "sequence_index": 4, "metres_after_boundary": 0.5},
    ]
    target, valid, audit = materialize_boundary_targets(teacher, timing)
    np.testing.assert_array_equal(target, np.asarray([0.0, 0.5, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(valid, np.asarray([False, True, False]))
    assert audit == {
        "timing_rows": 2,
        "valid_backprojection_rows": 1,
        "pre_boundary_event_rows": 1,
    }


def test_boundary_targets_reject_out_of_history_offset() -> None:
    teacher = [{"traversal_id": "t0", "sequence_index": 1, "event": "geometry_transition"}]
    timing = [{"traversal_id": "t0", "sequence_index": 1, "metres_after_boundary": 11.5}]
    with pytest.raises(RuntimeError, match="outside"):
        materialize_boundary_targets(teacher, timing)
