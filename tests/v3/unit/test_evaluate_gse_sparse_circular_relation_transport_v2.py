import numpy as np

import evaluate_gse_sparse_circular_relation_transport_v2 as evaluation


def test_six_token_alignment_restores_permutation() -> None:
    reference = np.asarray([[[0, 20, 40, 60, 80, 100]]], dtype=np.int16)
    candidate = np.asarray([[[61, 101, 1, 81, 41, 21]]], dtype=np.int16)
    mapping = evaluation._align_six(reference, candidate)
    restored = evaluation._gather_token(candidate[..., None], mapping)[..., 0]
    assert restored.tolist() == [[[1, 21, 41, 61, 81, 101]]]


def test_numpy_circular_nms_is_distinct() -> None:
    logits = np.full((2, 5, 180), -10.0)
    for rank, index in enumerate((179, 0, 1, 40, 80, 120, 150, 20)):
        logits[..., index] = 10 - rank
    selected = evaluation._circular_nms(logits)
    shifted = evaluation._circular_nms(np.roll(logits, 5, axis=-1))
    assert np.array_equal(shifted, np.remainder(selected + 5, 180))
    distance = evaluation._circular_distance(selected[..., :, None], selected[..., None, :])
    distance += np.eye(6, dtype=distance.dtype) * 180
    assert distance.min() > 4


def test_truth_matching_is_wrap_aware_and_count_limited() -> None:
    identity = np.full(180, -1, dtype=np.int64)
    identity[0] = 10
    identity[41] = 20
    mapping, distance = evaluation._match_prediction_to_truth(
        np.asarray([179, 40, 80, 100, 120, 140]), 2, identity
    )
    assert mapping[:2].tolist() == [10, 20]
    assert distance[:2].tolist() == [1.0, 1.0]
    assert np.all(mapping[2:] == -1)


def test_localization_uses_continuous_deployed_bearing_not_hard_proposal_bin() -> None:
    ensemble = {
        "token_bin_index": np.asarray([[[2, 40, 80, 100, 120, 140]]], dtype=np.int16),
        "token_bearing_deg": np.asarray([[[1.0, 80.0, 160.0, 200.0, 240.0, 280.0]]]),
    }
    coordinate = evaluation._deployed_token_coordinates_bins(ensemble)
    identity = np.full(180, -1, dtype=np.int64)
    identity[0] = 10
    mapping, distance = evaluation._match_prediction_to_truth(coordinate[0, 0], 1, identity)
    assert mapping[0] == 10
    assert distance[0] == 0.5
    assert evaluation._circular_distance(ensemble["token_bin_index"][0, 0, 0], 0) == 2


def test_deployed_bearing_coordinates_wrap_at_full_circle() -> None:
    ensemble = {
        "token_bin_index": np.asarray([[[179, 20, 40, 60, 80, 100]]], dtype=np.int16),
        "token_bearing_deg": np.asarray([[[359.0, 40.0, 80.0, 120.0, 160.0, 200.0]]]),
    }
    coordinate = evaluation._deployed_token_coordinates_bins(ensemble)
    assert coordinate[0, 0, 0] == 179.5
    assert evaluation._circular_distance(coordinate[0, 0, 0], 0) == 0.5


def test_objective_recall_counts_unemitted_teacher_positives_as_false_negatives() -> None:
    metrics = evaluation._objective_recall_metrics(
        score=np.asarray([0.95, 0.90, 0.80]),
        truth=np.asarray([True, True, False]),
        threshold=0.85,
        objective_positive_total=5,
    )
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 3
    assert metrics["positives"] == 5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.4
    assert np.isclose(metrics["f1"], 4.0 / 7.0)


def test_relation_metrics_use_objective_teacher_denominator() -> None:
    relation = {
        "score": np.asarray([0.9, 0.8, 0.1]),
        "truth": np.asarray([True, False, False]),
        "positive_total": 4,
    }
    metrics = evaluation._relation_metrics(relation, threshold=0.85)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 3
    assert metrics["recall"] == 0.25


def test_structural_refusal_metrics_use_objective_teacher_denominator() -> None:
    values = (
        np.asarray([0.95, 0.80, 0.10]),
        np.asarray([True, False, False]),
        5,
    )
    metrics = evaluation._structural_metrics(values, threshold=0.90)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 4
    assert metrics["positives"] == 5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.2
    assert np.isclose(metrics["f1"], 1.0 / 3.0)
