from __future__ import annotations

import torch

from mtare_topo.representation.gse_causal_geometry_delta import (
    CausalGeometryDeltaEventHead,
    causal_geometry_delta_event_loss,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_last_block_adaptation import (
    EXPECTED_TOTAL_TRAINABLE_PARAMETERS,
    audit_last_block_gradients,
    configure_last_block_adaptation,
    forward_last_block_adaptation,
    trainable_parameters,
)


def test_only_last_encoder_block_and_fresh_head_are_trainable() -> None:
    backbone = GeometrySemanticEventNet()
    head = CausalGeometryDeltaEventHead()
    report = configure_last_block_adaptation(backbone, head)
    assert report["total_trainable_parameters"] == EXPECTED_TOTAL_TRAINABLE_PARAMETERS
    assert report["backbone_trainable_parameters"] == report["last_encoder_block_parameters"]
    assert all(name.startswith("encoder.5.") for name in report["trainable_backbone_names"])
    last_block, head_parameters = trainable_parameters(backbone, head)
    assert sum(parameter.numel() for parameter in last_block) == 271_104
    assert sum(parameter.numel() for parameter in head_parameters) == 444_425


def test_real_forward_backward_cannot_escape_last_block_boundary() -> None:
    torch.manual_seed(11)
    backbone = GeometrySemanticEventNet()
    head = CausalGeometryDeltaEventHead()
    configure_last_block_adaptation(backbone, head)
    student = torch.randn(2, 5, 2, 16, 720)
    output = forward_last_block_adaptation(backbone, head, student)
    losses = causal_geometry_delta_event_loss(
        output,
        torch.tensor([0, 4]),
        torch.tensor([[0.2, -0.1, 0.3, 0.001], [-0.2, 0.1, -0.3, -0.001]]),
        torch.tensor([True, True]),
    )
    losses["total"].backward()
    audit = audit_last_block_gradients(backbone, head)
    assert audit["unexpected_frozen_gradient_tensors"] == 0
    assert audit["trainable_tensors_with_gradient"] > 0
    for name, parameter in backbone.named_parameters():
        if not name.startswith("encoder.5."):
            assert parameter.grad is None


def test_zero_step_output_still_matches_original_backbone_distribution() -> None:
    torch.manual_seed(19)
    backbone = GeometrySemanticEventNet()
    head = CausalGeometryDeltaEventHead()
    configure_last_block_adaptation(backbone, head)
    student = torch.randn(1, 5, 2, 16, 720)
    with torch.no_grad():
        causal = backbone.encode_causal_features(student)
        baseline = torch.softmax(backbone.event_head(causal["context"]), dim=1)
        output = forward_last_block_adaptation(backbone, head, student)
    assert torch.allclose(output["event_probability"], baseline, atol=1e-6)
    assert torch.equal(output["geometry_delta"], torch.zeros(1, 4))
