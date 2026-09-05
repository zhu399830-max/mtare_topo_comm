from __future__ import annotations

import numpy as np

from execute_gse_token_validity_action_track_feasibility_v1 import _result_payload

from mtare_topo.evaluation.gse_token_validity_feasibility import (
    binary_ranking_metrics,
    causal_descriptor_track_mean_confidence,
    cross_seed_geometry_consensus_score,
)


def _tokens(rows: int = 3) -> np.ndarray:
    value = np.zeros((rows, 6, 40), dtype=np.float32)
    value[:, :, 0] = np.arange(1, 7, dtype=np.float32) / 10.0
    value[:, :, 1] = 1.0
    value[:, :, 3] = 2.0
    for slot in range(6):
        value[:, slot, 8 + slot] = 1.0
    return value


def test_binary_ranking_never_splits_score_ties() -> None:
    result = binary_ranking_metrics(
        np.asarray([1, 0, 1, 0], dtype=bool),
        np.asarray([0.9, 0.9, 0.8, 0.1]),
        precision_floor=0.6,
    )
    assert result["threshold_at_precision_floor"] == 0.8
    assert result["selected_tokens"] == 3
    assert result["recall_at_precision_floor"] == 1.0


def test_causal_track_mean_is_past_only_and_resets() -> None:
    value = _tokens()
    value[1, :, 0] += 0.2
    value[2, :, 0] += 0.4
    result = causal_descriptor_track_mean_confidence(
        value,
        np.asarray(["a", "a", "b"]),
        np.asarray([0, 1, 0]),
    )
    assert np.allclose(result[0], value[0, :, 0])
    assert np.allclose(result[1], (value[0, :, 0] + value[1, :, 0]) / 2.0)
    assert np.allclose(result[2], value[2, :, 0])


def test_cross_seed_consensus_is_permutation_invariant() -> None:
    base = _tokens(rows=1)[0]
    angle = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    base[:, 1] = np.sin(angle)
    base[:, 2] = np.cos(angle)
    seeds = np.stack((base, base[[2, 0, 5, 1, 4, 3]], base[[5, 4, 3, 2, 1, 0]]), axis=0)[None]
    score = cross_seed_geometry_consensus_score(seeds, chunk_size=1)
    assert score.shape == (1, 3, 6)
    assert np.all(np.floor(score / 2.0) == 2)


def test_result_payload_accepts_outer_and_inner(tmp_path) -> None:
    outer = tmp_path / "outer.json"
    inner = tmp_path / "inner.json"
    outer.write_text('{"result":{"value":1}}', encoding="utf-8")
    inner.write_text('{"value":2}', encoding="utf-8")
    assert _result_payload(outer) == {"value": 1}
    assert _result_payload(inner) == {"value": 2}
