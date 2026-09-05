from __future__ import annotations

import unittest

import torch

from mtare_topo.representation.gse_graph import (
    GeometrySemanticEventNet,
    gse_multitask_loss,
    supervised_association_loss,
)


class GSEGraphModelTest(unittest.TestCase):
    def test_output_contract(self) -> None:
        torch.manual_seed(0)
        model = GeometrySemanticEventNet().eval()
        scans = torch.zeros((1, 5, 2, 16, 720), dtype=torch.float32)
        with torch.no_grad():
            outputs = model(scans)
        self.assertEqual(tuple(outputs["event_logits"].shape), (1, 5))
        self.assertEqual(tuple(outputs["local_axis"].shape), (1, 3))
        self.assertEqual(tuple(outputs["place_descriptor"].shape), (1, 128))
        self.assertEqual(tuple(outputs["exit_presence_logits"].shape), (1, 6))
        self.assertEqual(tuple(outputs["exit_heading_unit"].shape), (1, 6, 2))
        self.assertEqual(tuple(outputs["exit_vertical_profile"].shape), (1, 6, 4))
        self.assertEqual(tuple(outputs["exit_descriptor"].shape), (1, 6, 32))
        self.assertTrue(torch.isfinite(torch.cat([value.reshape(-1) for value in outputs.values()])).all())
        self.assertTrue((outputs["width_m"] > 0.0).all())
        self.assertTrue((outputs["height_m"] > 0.0).all())

    def test_future_or_wrong_history_shape_is_rejected(self) -> None:
        model = GeometrySemanticEventNet().eval()
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 6, 2, 16, 720)))

    def test_directional_outputs_are_circularly_equivariant(self) -> None:
        torch.manual_seed(4)
        model = GeometrySemanticEventNet().eval()
        scans = torch.rand((2, 5, 2, 16, 720))
        with torch.no_grad():
            original = model(scans)
            shifted = model(torch.roll(scans, shifts=4, dims=-1))
        angle = torch.deg2rad(torch.tensor(2.0))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        expected_axis = original["local_axis"].clone()
        expected_axis[:, 0] = cosine * original["local_axis"][:, 0] - sine * original["local_axis"][:, 1]
        expected_axis[:, 1] = sine * original["local_axis"][:, 0] + cosine * original["local_axis"][:, 1]
        expected_heading = original["exit_heading_unit"].clone()
        expected_heading[..., 0] = sine * original["exit_heading_unit"][..., 1] + cosine * original["exit_heading_unit"][..., 0]
        expected_heading[..., 1] = cosine * original["exit_heading_unit"][..., 1] - sine * original["exit_heading_unit"][..., 0]
        torch.testing.assert_close(shifted["local_axis"], expected_axis, atol=2e-4, rtol=0.0)
        torch.testing.assert_close(shifted["exit_heading_unit"], expected_heading, atol=2e-4, rtol=0.0)

    def test_supervised_association_loss(self) -> None:
        descriptors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        labels = torch.tensor([1, 1, 2, 2])
        loss = supervised_association_loss(descriptors, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        with self.assertRaises(ValueError):
            supervised_association_loss(descriptors, torch.tensor([1, 2, 3, 4]))

    def test_complete_multitask_loss_is_finite_and_backward(self) -> None:
        torch.manual_seed(2)
        model = GeometrySemanticEventNet()
        outputs = model(torch.rand((4, 5, 2, 16, 720)))
        target_heading = torch.zeros((4, 3, 2))
        target_heading[..., 1] = 1.0
        targets = {
            "event_index": torch.tensor([0, 1, 1, 2]),
            "local_axis": torch.tensor([[1.0, 0.0, 0.0]] * 4),
            "width_m": torch.full((4,), 5.0),
            "height_m": torch.full((4,), 4.0),
            "slope_deg": torch.zeros(4),
            "curvature_per_m": torch.full((4,), 0.02),
            "geometry_valid_mask": torch.tensor(
                [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                dtype=torch.bool,
            ),
            "exit_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool),
            "exit_width_valid_mask": torch.tensor(
                [[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
            ),
            "exit_heading_unit": target_heading,
            "exit_opening_width_m": torch.full((4, 3), 2.0),
            "exit_vertical_profile": torch.zeros((4, 3, 4)),
            "exit_identity": torch.tensor(
                [[100, 200, -1], [100, -1, -1], [300, -1, -1], [-1, -1, -1]]
            ),
            "association_identity": torch.tensor([-1, 10, 20, 20]),
            "association_valid_mask": torch.tensor([0, 1, 1, 1], dtype=torch.bool),
        }
        losses = gse_multitask_loss(outputs, targets)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        changed = dict(targets)
        changed["width_m"] = targets["width_m"].clone()
        changed["height_m"] = targets["height_m"].clone()
        changed["width_m"][0] = 5000.0
        changed["height_m"][0] = 5000.0
        changed_losses = gse_multitask_loss(outputs, changed)
        self.assertAlmostEqual(
            float(losses["geometry"].detach()),
            float(changed_losses["geometry"].detach()),
            places=7,
        )
        self.assertGreaterEqual(float(losses["exit_association"].detach()), 0.0)
        masked_exit_width = dict(targets)
        masked_exit_width["exit_opening_width_m"] = targets["exit_opening_width_m"].clone()
        masked_exit_width["exit_opening_width_m"][0, 0] = 5000.0
        masked_exit_width_losses = gse_multitask_loss(outputs, masked_exit_width)
        self.assertAlmostEqual(
            float(losses["exit_geometry"].detach()),
            float(masked_exit_width_losses["exit_geometry"].detach()),
            places=7,
        )
        losses["total"].backward()
        self.assertIsNotNone(model.event_head.weight.grad)

        weighted = gse_multitask_loss(
            outputs,
            targets,
            event_class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        self.assertTrue(torch.isfinite(weighted["total"]))
        with self.assertRaises(ValueError):
            gse_multitask_loss(outputs, targets, event_class_weights=torch.ones(4))


if __name__ == "__main__":
    unittest.main()
