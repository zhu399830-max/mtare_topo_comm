from __future__ import annotations

import math
import unittest

import torch

from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
    _canonical_active_order,
    primitive_relation_losses,
)
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_SLOTS,
    PrimitiveRelationNet,
    register_causal_lidar_points,
)


def _student(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    ranges = 0.1 + 0.8 * torch.rand(batch, 5, 16, 720, generator=generator)
    valid = torch.ones_like(ranges)
    range_valid = torch.stack((ranges, valid), dim=2)
    translation = torch.zeros(batch, 5, 3)
    translation[:, 0] = torch.tensor((-1.0, 0.2, 0.1))
    translation[:, 1] = torch.tensor((-0.75, 0.1, 0.05))
    translation[:, 2] = torch.tensor((-0.5, 0.05, 0.0))
    translation[:, 3] = torch.tensor((-0.25, 0.0, 0.0))
    yaw = torch.zeros(batch, 5)
    yaw[:, :4] = torch.tensor((-4.0, -3.0, -2.0, -1.0))
    return range_valid, translation, yaw


def _targets(batch: int = 1) -> PrimitiveRelationLossTargets:
    mask = torch.zeros(batch, 32); mask[:, :3] = 1
    axis = torch.zeros(batch, 32, 3, 3)
    axis[:, 0] = torch.tensor(((1.0, -1.0, 0.0), (2.0, -1.0, 0.1), (3.0, -1.0, 0.2)))
    axis[:, 1] = torch.tensor(((1.0, 0.0, 0.0), (2.0, 0.0, 0.1), (3.0, 0.0, 0.2)))
    axis[:, 2] = torch.tensor(((1.0, 1.0, 0.0), (2.0, 1.0, 0.1), (3.0, 1.0, 0.2)))
    half_axes = torch.zeros(batch, 32, 2, 2); half_axes[:, :3] = torch.tensor(((1.2, 0.9), (1.0, 0.8)))
    exponent = torch.zeros(batch, 32, 2); exponent[:, 0] = 2.0; exponent[:, 1] = 6.0; exponent[:, 2] = torch.tensor((2.0, 8.0))
    temporal = torch.zeros(batch, 5, 32); temporal[:, :, :3] = 1; temporal[:, 0, 2] = 0
    attachment = torch.zeros(batch, 32, 2, 32, 2)
    attachment[:, 0, 1, 1, 0] = 1; attachment[:, 1, 0, 0, 1] = 1
    overlap = torch.zeros(batch, 32, 32); overlap[:, 0, 2] = 1; overlap[:, 2, 0] = 1
    return PrimitiveRelationLossTargets(mask, axis, half_axes, exponent, temporal, attachment, overlap)


def _permute_targets(targets: PrimitiveRelationLossTargets, permutation: torch.Tensor) -> PrimitiveRelationLossTargets:
    return PrimitiveRelationLossTargets(
        targets.primitive_mask[:, permutation],
        targets.axis_control_current_sensor_m[:, permutation],
        targets.endpoint_half_axes_m[:, permutation],
        targets.endpoint_shape_exponent[:, permutation],
        targets.temporal_visibility[:, :, permutation],
        targets.endpoint_attachment[:, permutation][:, :, :, permutation],
        targets.disconnected_overlap[:, permutation][:, :, permutation],
    )


class PrimitiveRelationModelTest(unittest.TestCase):
    def test_registration_rotation_contract(self) -> None:
        scans, translation, yaw = _student()
        points, valid = register_causal_lidar_points(scans, translation, yaw)
        shift = 72; angle = math.radians(0.5 * shift)
        rotated_scans = scans.roll(shift, dims=-1)
        rotated_translation = translation.clone()
        x, y = translation[..., 0].clone(), translation[..., 1].clone()
        rotated_translation[..., 0] = math.cos(angle) * x - math.sin(angle) * y
        rotated_translation[..., 1] = math.sin(angle) * x + math.cos(angle) * y
        rotated_points, rotated_valid = register_causal_lidar_points(rotated_scans, rotated_translation, yaw)
        expected = points.clone()
        x, y = points[..., 0].clone(), points[..., 1].clone()
        expected[..., 0] = math.cos(angle) * x - math.sin(angle) * y
        expected[..., 1] = math.sin(angle) * x + math.cos(angle) * y
        self.assertTrue(torch.equal(rotated_valid, valid.roll(shift, dims=-1)))
        self.assertLess(float((rotated_points - expected.roll(shift, dims=-2)).abs().max()), 2e-5)

    def test_output_schema_and_finiteness(self) -> None:
        torch.manual_seed(3); model = PrimitiveRelationNet().eval(); inputs = _student()
        with torch.no_grad(): output = model(*inputs)
        expected = {
            "existence_logits": (1, 32), "axis_control_current_sensor_m": (1, 32, 3, 3),
            "endpoint_half_axes_m": (1, 32, 2, 2), "endpoint_shape_exponent": (1, 32, 2),
            "endpoint_descriptor": (1, 32, 2, 32), "geometry_uncertainty": (1, 32),
            "endpoint_attachment_logits": (1, 32, 2, 32, 2),
            "disconnected_overlap_logits": (1, 32, 32),
            "temporal_correspondence_logits": (1, 5, 32, 33),
            "temporal_presence_logits": (1, 5, 32),
        }
        for name, shape in expected.items():
            value = getattr(output, name); self.assertEqual(tuple(value.shape), shape); self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 1_000_000)
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 3_000_000)

    def test_query_permutation_equivariance(self) -> None:
        torch.manual_seed(5); model = PrimitiveRelationNet().eval(); inputs = _student()
        permutation = torch.randperm(MAXIMUM_SLOTS, generator=torch.Generator().manual_seed(9))
        with torch.no_grad(): reference = model(*inputs); changed = model(*inputs, query_permutation=permutation)
        for name in ("existence_logits", "axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent", "endpoint_descriptor", "geometry_uncertainty", "temporal_presence_logits"):
            value = getattr(reference, name); actual = getattr(changed, name)
            dimension = 2 if name == "temporal_presence_logits" else 1
            self.assertTrue(torch.allclose(actual, value.index_select(dimension, permutation), atol=2e-5, rtol=2e-5), name)
        expected_attachment = reference.endpoint_attachment_logits[:, permutation][:, :, :, permutation]
        expected_overlap = reference.disconnected_overlap_logits[:, permutation][:, :, permutation]
        expected_temporal = torch.cat((reference.temporal_correspondence_logits[..., :32][:, :, permutation][:, :, :, permutation], reference.temporal_correspondence_logits[..., 32:][:, :, permutation]), dim=-1)
        self.assertTrue(torch.allclose(changed.endpoint_attachment_logits, expected_attachment, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(changed.disconnected_overlap_logits, expected_overlap, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(changed.temporal_correspondence_logits, expected_temporal, atol=2e-5, rtol=2e-5))

    def test_loss_is_target_permutation_invariant(self) -> None:
        torch.manual_seed(7); model = PrimitiveRelationNet().eval(); inputs = _student(); prediction = model(*inputs)
        targets = _targets(); permutation = torch.cat((torch.tensor((2, 0, 1)), torch.arange(3, 32)))
        first = primitive_relation_losses(prediction, targets, inputs[0])
        second = primitive_relation_losses(prediction, _permute_targets(targets, permutation), inputs[0])
        for name in first: self.assertAlmostEqual(float(first[name].detach()), float(second[name].detach()), places=6, msg=name)

    def test_teacher_canonical_order_is_slot_permutation_independent(self) -> None:
        targets = _targets()
        permutation = torch.cat((torch.tensor((2, 0, 1)), torch.arange(3, 32)))
        changed = _permute_targets(targets, permutation)
        active = torch.nonzero(targets.primitive_mask[0].bool()).flatten()
        changed_active = torch.nonzero(changed.primitive_mask[0].bool()).flatten()
        first = _canonical_active_order(targets, 0, active)
        second = _canonical_active_order(changed, 0, changed_active)
        self.assertTrue(torch.equal(
            targets.axis_control_current_sensor_m[0, first],
            changed.axis_control_current_sensor_m[0, second],
        ))
        self.assertTrue(torch.equal(
            targets.endpoint_attachment[0, first][:, :, first],
            changed.endpoint_attachment[0, second][:, :, second],
        ))

    def test_loss_is_endpoint_reversal_invariant(self) -> None:
        torch.manual_seed(11); model = PrimitiveRelationNet().eval(); inputs = _student(); prediction = model(*inputs)
        target = _targets(); axis = target.axis_control_current_sensor_m.clone(); axes = target.endpoint_half_axes_m.clone(); exponent = target.endpoint_shape_exponent.clone(); attachment = target.endpoint_attachment.clone()
        axis[:, 0] = axis[:, 0].flip(1); axes[:, 0] = axes[:, 0].flip(1); exponent[:, 0] = exponent[:, 0].flip(1)
        attachment[:, 0] = attachment[:, 0].flip(1); attachment[:, :, :, 0] = attachment[:, :, :, 0].flip(3)
        reversed_target = PrimitiveRelationLossTargets(target.primitive_mask, axis, axes, exponent, target.temporal_visibility, attachment, target.disconnected_overlap)
        first = primitive_relation_losses(prediction, target, inputs[0]); second = primitive_relation_losses(prediction, reversed_target, inputs[0])
        for name in first: self.assertAlmostEqual(float(first[name].detach()), float(second[name].detach()), places=6, msg=name)

    def test_six_losses_and_all_gradients_are_finite(self) -> None:
        torch.manual_seed(13); model = PrimitiveRelationNet(); inputs = _student(); prediction = model(*inputs)
        losses = primitive_relation_losses(prediction, _targets(), inputs[0])
        self.assertEqual(set(losses), {"primitive_set_parameters", "surface_reconstruction", "ray_free_space", "port_relations", "temporal_equivariance", "uncertainty_calibration", "total"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))

    def test_forbidden_student_shapes_fail_closed(self) -> None:
        model = PrimitiveRelationNet(); scans, translation, yaw = _student()
        with self.assertRaisesRegex(ValueError, "range_valid"): model(scans[..., :-1], translation, yaw)
        bad = translation.clone(); bad[:, -1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "exactly zero"): model(scans, bad, yaw)


if __name__ == "__main__":
    unittest.main()
