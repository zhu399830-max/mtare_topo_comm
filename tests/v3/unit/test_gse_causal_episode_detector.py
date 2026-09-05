from __future__ import annotations

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_causal_episode_detector import (
    CausalEpisodeDetector,
    causal_episode_multiple_instance_loss,
    encode_spatial_scan_embeddings,
    encode_spatial_scan_features,
    materialize_causal_episode_references,
    materialize_past_only_references,
)


def _row(sequence: int, event: str = "corridor", identity: str | None = None, traversal: str = "w:e:d0") -> dict:
    frame = sequence + 4
    current = (0 if traversal.endswith("d0") else 100) + frame
    return {
        "traversal_id": traversal,
        "sequence_index": sequence,
        "frame_index": frame,
        "global_frame_references": list(range(current - 4, current + 1)),
        "event": event,
        "identity": identity,
    }


def test_twelve_frame_references_are_left_padded_and_never_cross_traversal() -> None:
    rows = [_row(i) for i in range(9)] + [_row(i, traversal="w:e:d1") for i in range(2)]
    bank = materialize_causal_episode_references(rows)
    assert bank.global_frame_references.shape == (11, 12)
    np.testing.assert_array_equal(bank.global_frame_references[0, -5:], np.arange(5))
    assert np.all(bank.global_frame_references[0, :-5] == -1)
    np.testing.assert_array_equal(bank.global_frame_references[8], np.arange(1, 13))
    np.testing.assert_array_equal(bank.global_frame_references[9, -5:], np.arange(100, 105))


def test_deployment_reference_materializer_does_not_require_teacher_fields() -> None:
    rows = [_row(i) for i in range(9)]
    for row in rows:
        row.pop("event")
        row.pop("identity")
    bank = materialize_past_only_references(rows)
    np.testing.assert_array_equal(bank.global_frame_references[8], np.arange(1, 13))
    assert bank.valid_history_mask[0].sum() == 5


def test_deployment_reference_materializer_rejects_missing_sequence() -> None:
    rows = [_row(0), _row(2)]
    with pytest.raises(ValueError, match="contiguous"):
        materialize_past_only_references(rows)


def test_reference_materializer_assigns_one_id_per_contiguous_episode() -> None:
    rows = [
        _row(0), _row(1, "turn", "a"), _row(2, "turn", "a"), _row(3),
        _row(4, "turn", "a"), _row(5, "junction", "b"),
    ]
    bank = materialize_causal_episode_references(rows)
    assert bank.episode_id.tolist() == [-1, 0, 0, -1, 1, 2]
    assert bank.event_index.tolist() == [0, 3, 3, 0, 3, 1]


def test_reference_materializer_rejects_legacy_crossing() -> None:
    row = _row(0)
    row["global_frame_references"][-1] += 1
    with pytest.raises(ValueError, match="reference"):
        materialize_causal_episode_references([row])


def test_detector_zero_residual_reproduces_baseline_event_probability() -> None:
    torch.manual_seed(0)
    model = CausalEpisodeDetector()
    embeddings = torch.randn(3, 12, 128)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[0, :7] = False
    logits = torch.randn(3, 5)
    directional = torch.randn(3, 12, 128, 36)
    output = model(embeddings, directional, mask, logits)
    torch.testing.assert_close(output["event_probability"], torch.softmax(logits, dim=1), atol=1e-6, rtol=1e-6)
    assert torch.all((output["boundary_offset_m"] >= 0) & (output["boundary_offset_m"] <= 11))


def test_spatial_encoder_operates_on_exact_scan_frames() -> None:
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(2, 128, kernel_size=1),
        torch.nn.AvgPool2d((2, 4)),
    )
    result = encode_spatial_scan_embeddings(encoder, torch.randn(2, 12, 2, 16, 720))
    assert result.shape == (2, 12, 128)
    features = encode_spatial_scan_features(encoder, torch.randn(2, 12, 2, 16, 720))
    assert features["pooled"].shape == (2, 12, 128)
    assert features["directional"].shape == (2, 12, 128, 36)
    assert features["vertical"].shape == (2, 12, 128, 8)


def test_detector_rejects_non_suffix_history_mask() -> None:
    model = CausalEpisodeDetector()
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[0, 5] = False
    with pytest.raises(ValueError, match="suffix"):
        model(torch.randn(1, 12, 128), torch.randn(1, 12, 128, 36), mask, torch.randn(1, 5))


def test_episode_loss_needs_one_jointly_correct_frame_not_all_frames() -> None:
    structural = torch.tensor([-4.0, -4.0, 5.0, -5.0], requires_grad=True)
    conditional = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], requires_grad=True)
    outputs = {
        "structural_logit": structural,
        "conditional_event_logits": conditional,
        "boundary_offset_m": torch.tensor([0.0, 0.0, 3.0, 0.0]),
    }
    loss = causal_episode_multiple_instance_loss(
        outputs,
        torch.tensor([3, 3, 3, 0]),
        torch.tensor([0, 0, 0, -1]),
        boundary_offset_target_m=torch.tensor([0.0, 0.0, 3.0, 0.0]),
        boundary_offset_valid=torch.tensor([False, False, True, False]),
    )
    assert float(loss["positive_episode_joint"].detach()) < 0.02
    assert float(loss["corridor_negative"].detach()) < 0.02
    assert float(loss["boundary_offset"].detach()) == 0.0
    loss["total"].backward()
    assert structural.grad is not None and conditional.grad is not None
