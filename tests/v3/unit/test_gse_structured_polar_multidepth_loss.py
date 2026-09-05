from __future__ import annotations

import math

import torch

from mtare_topo.representation.gse_structured_polar_multidepth_event import (
    StructuredPolarMultiDepthEventEncoder,
    rasterize_structured_polar_targets,
)
from mtare_topo.representation.gse_structured_polar_multidepth_loss import (
    balanced_dense_presence_loss,
    rasterize_structured_polar_targets_vectorized,
    structured_polar_direct_loss,
)


def test_per_row_presence_balance_does_not_use_global_imbalance_weight():
    logits = torch.zeros(2, 180, 2)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 3, 0] = True
    assert torch.allclose(
        balanced_dense_presence_loss(logits, mask),
        torch.tensor(math.log(2.0)),
        atol=1e-7,
    )


def test_direct_loss_handles_empty_and_same_ray_depth_pair():
    torch.manual_seed(4)
    model = StructuredPolarMultiDepthEventEncoder()
    ranges = torch.full((2, 5, 16, 720), 50.0)
    valid = torch.zeros_like(ranges, dtype=torch.uint8)
    outputs = model(ranges, valid)
    event_type = torch.full((2, 16), -1, dtype=torch.long)
    xyz = torch.zeros(2, 16, 3)
    identity = torch.full((2, 16), -1, dtype=torch.long)
    mask = torch.zeros(2, 16, dtype=torch.uint8)
    angle = math.radians(0.75)
    for index, distance in enumerate((4.0, 42.0)):
        event_type[0, index] = index
        xyz[0, index] = torch.tensor((distance * math.cos(angle), distance * math.sin(angle), 0.0))
        identity[0, index] = 9 + index
        mask[0, index] = 1
    dense = rasterize_structured_polar_targets(
        {"event_type_index": event_type, "event_relative_xyz_m": xyz, "event_identity_index": identity, "event_mask": mask},
        outputs["free_range_profile_m"].detach(),
    )
    losses = structured_polar_direct_loss(
        outputs,
        dense,
        event_type_class_weights=torch.ones(2),
    )
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_vectorized_rasterizer_exactly_matches_readiness_reference():
    ranges = torch.full((2, 5, 16, 720), 50.0)
    valid = torch.zeros_like(ranges, dtype=torch.uint8)
    outputs = StructuredPolarMultiDepthEventEncoder()(ranges, valid)
    event_type = torch.full((2, 16), -1, dtype=torch.long)
    xyz = torch.zeros(2, 16, 3)
    identity = torch.full((2, 16), -1, dtype=torch.long)
    mask = torch.zeros(2, 16, dtype=torch.uint8)
    for index, (angle_deg, distance) in enumerate(((0.75, 42.0), (0.75, 4.0), (90.75, 12.0))):
        angle = math.radians(angle_deg)
        event_type[0, index] = index % 2
        xyz[0, index] = torch.tensor((distance * math.cos(angle), distance * math.sin(angle), 0.0))
        identity[0, index] = 20 + index
        mask[0, index] = 1
    source = {"event_type_index": event_type, "event_relative_xyz_m": xyz, "event_identity_index": identity, "event_mask": mask}
    reference = rasterize_structured_polar_targets(source, outputs["free_range_profile_m"].detach())
    vectorized = rasterize_structured_polar_targets_vectorized(source, outputs["free_range_profile_m"].detach())
    for name in reference:
        assert torch.equal(reference[name], vectorized[name])
