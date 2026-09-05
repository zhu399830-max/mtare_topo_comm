from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    circular_gaussian_label,
    rasterize_generic_model_output,
    select_role_stratified_poses,
    sphere_polyline_intersections,
)
from mtare_topo.governance import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORLD = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260810_cano_readonly_audited_adapter_smoke_v1_seed0/"
    "artifacts/world_000"
)


class CanoSensorSmokeTest(unittest.TestCase):
    def test_sphere_polyline_intersections_returns_two_directions(self) -> None:
        points = np.asarray([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = sphere_polyline_intersections([0.0, 0.0, 0.0], 5.0, points)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.allclose(sorted(point[0] for point in result), [-5.0, 5.0]))

    def test_circular_gaussian_wraps_across_zero_degrees(self) -> None:
        label = circular_gaussian_label([359.5])
        self.assertEqual(label.shape, (720,))
        self.assertEqual(label[719], 1.0)
        self.assertGreater(label[0], 0.98)
        self.assertLess(label[360], 1e-6)

    def test_generic_model_output_raster_contract(self) -> None:
        ranges, valid, counts = rasterize_generic_model_output(
            [0.0, 0.49, 359.6], [-15.0, -13.0, 15.0], [2.0, 3.0, 4.0]
        )
        self.assertEqual(ranges.shape, (16, 720))
        self.assertEqual(valid.shape, (16, 720))
        self.assertEqual(int(valid.sum()), 3)
        self.assertEqual(ranges[0, 0], 2.0)
        self.assertEqual(ranges[1, 1], 3.0)
        self.assertEqual(ranges[15, 719], 4.0)
        self.assertEqual(counts["accepted"], 3)

    def test_audited_seed0_pose_manifest_is_exact_and_deterministic(self) -> None:
        graph = load_json(WORLD / "graph.json")
        splines = load_json(WORLD / "splines.json")
        fta = float((WORLD / "fta_dist.txt").read_text(encoding="utf-8"))
        first = select_role_stratified_poses(graph, splines, fta)
        second = select_role_stratified_poses(graph, splines, fta)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(
            [
                sum(pose["role"] == role for pose in first)
                for role in ("tunnel_interior", "junction_transition", "terminal_approach")
            ],
            [8, 8, 8],
        )
        self.assertEqual(
            {pose["role"]: pose["expected_branch_count"] for pose in first},
            {
                "tunnel_interior": 2,
                "junction_transition": 3,
                "terminal_approach": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
