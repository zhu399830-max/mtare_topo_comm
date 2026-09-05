from __future__ import annotations

from pathlib import Path
import torch

from mtare_topo.evaluation.primitive_relation_failure_attribution import (
    pair_space_counts,
    relation_sweeps_for_mask,
    slot_population_counts,
)
from mtare_topo.evaluation.primitive_relation_metrics import select_threshold
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationPrediction


def _prediction() -> PrimitiveRelationPrediction:
    batch, slots = 1, 32
    attachment = torch.full((batch, slots, 2, slots, 2), -8.0)
    # One correct true relation (slot0 endpoint1 <-> slot1 endpoint0).
    attachment[0, 0, 1, 1, 0] = 8.0
    attachment[0, 1, 0, 0, 1] = 8.0
    # One high-scoring false relation involving redundant slot2.
    attachment[0, 0, 0, 2, 0] = 8.0
    attachment[0, 2, 0, 0, 0] = 8.0
    return PrimitiveRelationPrediction(
        existence_logits=torch.tensor([[8.0, 8.0, 8.0] + [-8.0] * 29]),
        axis_control_current_sensor_m=torch.zeros(batch, slots, 3, 3),
        endpoint_half_axes_m=torch.ones(batch, slots, 2, 2),
        endpoint_shape_exponent=torch.ones(batch, slots, 2),
        endpoint_descriptor=torch.zeros(batch, slots, 2, 32),
        geometry_uncertainty=torch.zeros(batch, slots, 8),
        endpoint_attachment_logits=attachment,
        disconnected_overlap_logits=torch.full((batch, slots, slots), -8.0),
        temporal_correspondence_logits=torch.zeros(batch, 5, slots, slots + 1),
        temporal_presence_logits=torch.zeros(batch, 5, slots),
    )


def _aligned() -> dict[str, torch.Tensor]:
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[0, :2] = True
    attachment = torch.zeros(1, 32, 2, 32, 2, dtype=torch.bool)
    attachment[0, 0, 1, 1, 0] = True; attachment[0, 1, 0, 0, 1] = True
    return {
        "mask": mask,
        "attachment": attachment,
        "overlap": torch.zeros(1, 32, 32, dtype=torch.bool),
    }


def test_pair_space_formula_and_slot_population() -> None:
    active = torch.zeros(2, 32, dtype=torch.bool)
    active[0, :3] = True; active[1, :2] = True
    target = torch.zeros_like(active); target[:, :2] = True
    assert pair_space_counts(active) == {
        "rows": 2, "primitives": 5, "attachment_pairs": 16, "overlap_pairs": 4,
    }
    population = slot_population_counts(active, target)
    assert population["matched_active"] == 4
    assert population["redundant_active"] == 1
    assert population["missed_targets"] == 0
    assert population["active_count_histogram"][2:4] == [1, 1]


def test_oracle_mask_removes_redundant_false_relation_without_changing_logits() -> None:
    prediction = _prediction(); aligned = _aligned()
    actual_mask = torch.sigmoid(prediction.existence_logits) >= 0.5
    actual = relation_sweeps_for_mask(prediction, aligned, actual_mask, thresholds=(0.5,))
    oracle = relation_sweeps_for_mask(prediction, aligned, aligned["mask"], thresholds=(0.5,))
    actual_selected = select_threshold(actual["attachment"], thresholds=(0.5,))
    oracle_selected = select_threshold(oracle["attachment"], thresholds=(0.5,))
    assert actual_selected["true_positive"] == oracle_selected["true_positive"] == 1
    assert actual_selected["false_positive"] == 1
    assert oracle_selected["false_positive"] == 0
    assert oracle_selected["f1"] == 1.0


def test_masks_fail_closed_on_wrong_shape() -> None:
    prediction = _prediction(); aligned = _aligned()
    try:
        relation_sweeps_for_mask(prediction, aligned, torch.ones(1, 31, dtype=torch.bool))
    except ValueError as error:
        assert "[batch,32]" in str(error)
    else:
        raise AssertionError("wrong mask shape must fail")


def test_tf32_corrective_is_c07_only_and_explicit() -> None:
    source = Path("tools/v3/evaluate_primitive_relation_three_seed_c07_tf32_corrective_v1.py").read_text()
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "torch.backends.cudnn.allow_tf32 = False" in source
    assert 'torch.set_float32_matmul_precision("highest")' in source
    assert "c08_loader" not in source
    assert '"c08_rows_read": 0' in source
