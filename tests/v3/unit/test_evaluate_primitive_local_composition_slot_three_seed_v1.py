import numpy as np
import torch

from evaluate_primitive_local_composition_slot_three_seed_v1 import (
    PairBuffer,
    SlotBuffer,
)


def test_pair_buffer_counts_unscored_objective_positive_as_false_negative() -> None:
    buffer = PairBuffer()
    buffer.append(
        torch.tensor([0.9, 0.2]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    )
    result = buffer.finalize(total_positive=2)
    assert result["candidate_true_pairs"] == 1
    assert result["objective_true_pairs"] == 2
    assert result["best_f1"]["false_negative"] >= 1


def test_slot_buffer_reports_structured_clusters_and_collapse() -> None:
    best = torch.zeros(2, 64, dtype=torch.long)
    best[0, 2:4] = 1
    best[1, :2] = 2
    best[1, 2:4] = 3
    confidence = torch.zeros(2, 64)
    confidence[:, :4] = 0.9
    active = torch.zeros(2, 64, dtype=torch.bool)
    active[:, :4] = True
    teacher = torch.tensor([2, 2])
    buffer = SlotBuffer(); buffer.append(best, confidence, active, teacher)
    result = buffer.finalize(threshold=0.5)
    assert result["predicted_nontrivial_clusters"] == 4
    assert result["mean_predicted_nontrivial_clusters_per_row"] == 2.0
    assert result["maximum_predicted_cluster_size"] == 2
    assert result["single_slot_collapse_rows"] == 0
    assert result["teacher_clusters"] == 4


def test_slot_buffer_none_threshold_emits_no_candidate() -> None:
    buffer = SlotBuffer()
    buffer.append(
        torch.zeros(1, 64, dtype=torch.long),
        torch.ones(1, 64),
        torch.ones(1, 64, dtype=torch.bool),
        torch.tensor([4]),
    )
    result = buffer.finalize(threshold=None)
    assert result["accepted_endpoints"] == 0
    assert result["predicted_nontrivial_clusters"] == 0
    assert result["teacher_clusters"] == 4


def test_slot_buffer_quantiles_use_only_active_endpoints() -> None:
    buffer = SlotBuffer()
    confidence = torch.zeros(1, 64); confidence[0, :2] = torch.tensor([0.25, 0.75])
    active = torch.zeros(1, 64, dtype=torch.bool); active[0, :2] = True
    buffer.append(torch.zeros(1, 64, dtype=torch.long), confidence, active, torch.tensor([1]))
    result = buffer.finalize(threshold=0.5)
    assert np.isclose(result["endpoint_confidence_quantiles"]["q50"], 0.5)
