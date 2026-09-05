from __future__ import annotations

from dataclasses import replace

import torch

from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts,
    THRESHOLD_GRID,
    align_for_evaluation,
    existence_sweep,
    finalize_geometry_totals,
    geometry_batch_totals,
    nonlearning_batch_to_torch,
    relation_counts,
    relation_sweeps,
    select_threshold,
    temporal_batch_metrics,
)
from mtare_topo.representation.primitive_relation_losses import PrimitiveRelationLossTargets
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationPrediction
from mtare_topo.semantics.primitive_relation_nonlearning import (
    NonlearningPrimitiveRelationPrediction,
)


def _exact_pair() -> tuple[PrimitiveRelationPrediction, PrimitiveRelationLossTargets]:
    batch, slots = 1, 32
    mask = torch.zeros(batch, slots); mask[:, :2] = 1
    axis = torch.zeros(batch, slots, 3, 3)
    axis[0, 0, :, 0] = torch.tensor((1.0, 2.0, 3.0))
    axis[0, 1, :, 1] = torch.tensor((1.0, 2.0, 3.0))
    half_axes = torch.ones(batch, slots, 2, 2)
    exponent = torch.full((batch, slots, 2), 2.0)
    temporal = torch.zeros(batch, 5, slots); temporal[:, :, :2] = 1
    attachment = torch.zeros(batch, slots, 2, slots, 2)
    attachment[0, 0, 1, 1, 0] = 1; attachment[0, 1, 0, 0, 1] = 1
    overlap = torch.zeros(batch, slots, slots)
    targets = PrimitiveRelationLossTargets(
        mask, axis, half_axes, exponent, temporal, attachment, overlap,
    )
    existence = torch.full((batch, slots), -10.0); existence[:, :2] = 10.0
    attachment_logits = torch.full_like(attachment, -10.0)
    attachment_logits[attachment.bool()] = 10.0
    overlap_logits = torch.full_like(overlap, -10.0)
    temporal_logits = torch.full((batch, 5, slots, slots + 1), -10.0)
    for slot in range(slots):
        temporal_logits[:, :, slot, slot if slot < 2 else slots] = 10.0
    temporal_presence = torch.full((batch, 5, slots), -10.0)
    temporal_presence[:, :, :2] = 10.0
    prediction = PrimitiveRelationPrediction(
        existence, axis.clone(), half_axes.clone(), exponent.clone(),
        torch.zeros(batch, slots, 2, 32), torch.zeros(batch, slots),
        attachment_logits, overlap_logits, temporal_logits, temporal_presence,
    )
    return prediction, targets


def test_exact_prediction_has_exact_geometry_and_relations() -> None:
    prediction, targets = _exact_pair()
    aligned = align_for_evaluation(prediction, targets)
    existence = existence_sweep(prediction, aligned)
    selected = select_threshold(existence)
    assert selected["f1"] == 1.0
    totals = finalize_geometry_totals(
        geometry_batch_totals(prediction, aligned, existence_threshold=0.5),
    )
    assert totals["target_coverage"] == 1.0
    assert totals["surface_chamfer_m_mean"] == 0.0
    relation = relation_sweeps(prediction, aligned, existence_threshold=0.5)
    threshold_index = int(torch.argmin(torch.abs(torch.as_tensor(THRESHOLD_GRID) - 0.5)))
    assert relation["attachment"][threshold_index].f1 == 1.0
    assert relation["disconnected_overlap"][threshold_index].f1 == 1.0
    direct_relation = relation_counts(
        prediction, aligned, existence_threshold=0.5,
        attachment_threshold=0.5, overlap_threshold=0.5,
    )
    assert direct_relation["attachment"].f1 == 1.0
    assert direct_relation["disconnected_overlap"].f1 == 1.0
    temporal = temporal_batch_metrics(prediction, aligned, existence_threshold=0.5)
    assert temporal["presence"].f1 == 1.0
    assert temporal["correspondence_correct"] == temporal["correspondence_total"]


def test_hidden_matched_attachment_is_excluded_from_metric_population() -> None:
    prediction, targets = _exact_pair()
    observed = torch.zeros(1, 32, 2)
    observed[0, 0, 1] = 1
    hidden_targets = replace(targets, endpoint_observed=observed)
    aligned = align_for_evaluation(prediction, hidden_targets)
    threshold_index = int(torch.argmin(
        torch.abs(torch.as_tensor(THRESHOLD_GRID) - 0.5)
    ))
    swept = relation_sweeps(
        prediction, aligned, existence_threshold=0.5,
    )["attachment"][threshold_index]
    direct = relation_counts(
        prediction, aligned, existence_threshold=0.5,
        attachment_threshold=0.5, overlap_threshold=0.5,
    )["attachment"]
    assert swept == BinaryCounts()
    assert direct == BinaryCounts()


def test_safe_threshold_prefers_maximum_recall() -> None:
    counts = (
        BinaryCounts(9, 1, 1),
        BinaryCounts(8, 0, 2),
        BinaryCounts(7, 0, 3),
    )
    selected = select_threshold(
        counts, thresholds=(0.1, 0.2, 0.3), minimum_precision=0.98,
    )
    assert selected["available"] is True
    assert selected["threshold"] == 0.2
    assert selected["recall"] == 0.8


def test_safe_threshold_reports_unavailable() -> None:
    selected = select_threshold(
        (BinaryCounts(1, 1, 9),), thresholds=(0.5,), minimum_precision=0.98,
    )
    assert selected["available"] is False


def test_nonlearning_adapter_preserves_hard_structure() -> None:
    prediction, _ = _exact_pair()
    hard = NonlearningPrimitiveRelationPrediction(
        primitive_mask=(prediction.existence_logits[0] > 0).to(torch.uint8).numpy(),
        axis_control_current_sensor_m=prediction.axis_control_current_sensor_m[0].numpy(),
        endpoint_half_axes_m=prediction.endpoint_half_axes_m[0].numpy(),
        endpoint_shape_exponent=prediction.endpoint_shape_exponent[0].numpy(),
        geometry_uncertainty=prediction.geometry_uncertainty[0].numpy(),
        temporal_visibility=(prediction.temporal_presence_logits[0] > 0).to(torch.uint8).numpy(),
        temporal_destination=prediction.temporal_correspondence_logits[0].argmax(-1).to(torch.int8).numpy(),
        endpoint_attachment=(prediction.endpoint_attachment_logits[0] > 0).to(torch.uint8).numpy(),
        disconnected_overlap=(prediction.disconnected_overlap_logits[0] > 0).to(torch.uint8).numpy(),
        azimuth_support=torch.zeros(32, 720, dtype=torch.uint8).numpy(),
    )
    converted = nonlearning_batch_to_torch((hard,), device=torch.device("cpu"))
    assert torch.equal(converted.existence_logits > 0, prediction.existence_logits > 0)
    assert torch.equal(
        converted.endpoint_attachment_logits > 0,
        prediction.endpoint_attachment_logits > 0,
    )
    assert torch.equal(
        converted.temporal_correspondence_logits.argmax(-1),
        prediction.temporal_correspondence_logits.argmax(-1),
    )
