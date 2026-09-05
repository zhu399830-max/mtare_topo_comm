"""Contracts for deterministic AEE-mask-matched Cano training views."""

from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.aee_mask_matched_views import (
    build_aee_mask_matched_cano_views,
    expand_domain_adaptation_views,
)


def _sample(frame_id: str, domain: str, valid: np.ndarray) -> dict[str, object]:
    student = np.empty((2, 16, 720), dtype=np.float32)
    student[0] = 0.2
    student[1] = valid.astype(np.float32)
    return {
        "student": student,
        "direction_target": np.zeros(720, dtype=np.float32),
        "count_target": np.int64(0),
        "role_target": np.int64(0),
        "frame_id": frame_id,
        "headings_robot_deg": (0.0,),
        "domain": domain,
    }


def _batch() -> list[dict[str, object]]:
    dense = np.ones((16, 720), dtype=bool)
    sparse_a = dense.copy()
    sparse_a[:, ::2] = False
    sparse_b = dense.copy()
    sparse_b[:, ::3] = False
    return [
        _sample("cano:b", "cano", dense),
        _sample("aee:b", "aee", sparse_b),
        _sample("cano:a", "cano", dense),
        _sample("aee:a", "aee", sparse_a),
    ]


def test_mask_matching_is_deterministic_sparse_and_does_not_mutate_sources() -> None:
    batch = _batch()
    originals = [np.asarray(item["student"]).copy() for item in batch]
    first, first_provenance = build_aee_mask_matched_cano_views(batch, seed=17)
    second, second_provenance = build_aee_mask_matched_cano_views(batch, seed=17)
    assert first_provenance == second_provenance
    assert [item["view_id"] for item in first] == [item["view_id"] for item in second]
    assert len(first) == 2
    for left, right in zip(first, second):
        assert np.array_equal(left["student"], right["student"])
        student = np.asarray(left["student"])
        assert left["domain"] == "cano_mask_matched"
        assert left["loss_weight"] == pytest.approx(0.5)
        assert np.all(student[0, student[1] == 0] == 1.0)
        assert np.all(student[1] <= 1.0)
    for item, original in zip(batch, originals):
        assert np.array_equal(item["student"], original)


def test_mask_matching_never_invents_a_cano_return() -> None:
    cano_valid = np.ones((16, 720), dtype=bool)
    cano_valid[0, 0] = False
    aee_valid = np.ones((16, 720), dtype=bool)
    aee_valid[0, 1] = False
    samples = [
        _sample("cano", "cano", cano_valid),
        _sample("aee", "aee", aee_valid),
    ]
    views, _ = build_aee_mask_matched_cano_views(samples, seed=1)
    valid = np.asarray(views[0]["student"])[1]
    assert valid[0, 0] == 0
    assert valid[0, 1] == 0
    assert int(valid.sum()) == 16 * 720 - 2


def test_expansion_counts_independent_samples_and_augmented_views() -> None:
    batch = _batch()
    expanded, provenance = expand_domain_adaptation_views(batch, seed=3)
    domains = [item["domain"] for item in expanded]
    assert domains.count("cano_dense") == 2
    assert domains.count("cano_mask_matched") == 2
    assert domains.count("aee") == 2
    assert len(provenance) == 2
    assert sum(float(item["loss_weight"]) for item in expanded) == pytest.approx(4.0)
    original_aee = {str(item["frame_id"]): item for item in batch if item["domain"] == "aee"}
    assert all(
        np.array_equal(item["student"], original_aee[str(item["frame_id"])]["student"])
        for item in expanded
        if item["domain"] == "aee"
    )


@pytest.mark.parametrize(
    "samples",
    [[], [_sample("cano", "cano", np.ones((16, 720), dtype=bool))]],
)
def test_mask_matching_rejects_unbalanced_domains(samples) -> None:
    with pytest.raises(ValueError, match="equal positive"):
        build_aee_mask_matched_cano_views(samples, seed=0)
