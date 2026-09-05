from __future__ import annotations

import json
import math
import struct
import unittest
from pathlib import Path

import numpy as np

from mtare_topo.data.cano_gazebo_parity import (
    analytic_box_ranges,
    decode_organized_cloud,
    gazebo_world_sdf,
    parity_metrics,
    select_parity_poses,
)


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"


class CanoGazeboParityTest(unittest.TestCase):
    def _load_parent(self, parent_id: str):
        base = RUN / "artifacts/meshes" / parent_id / "primary"
        graph = json.loads((base / "graph.json").read_text())
        splines = json.loads((base / "splines.json").read_text())
        geometry = json.loads((base / "geometry_parameters.json").read_text())
        return graph, splines, geometry

    def test_exact_three_parent_pose_contract(self):
        expected = {
            "S01_flat_tree_small_C01": {"tunnel_interior": 3, "junction_transition": 2, "terminal_approach": 3},
            "S06_3d_branch_medium_C01": {"tunnel_interior": 3, "junction_transition": 3, "terminal_approach": 2},
            "S10_3d_complex_C01": {"tunnel_interior": 2, "junction_transition": 3, "terminal_approach": 3},
        }
        aggregate = {key: 0 for key in next(iter(expected.values()))}
        for parent_id, role_counts in expected.items():
            graph, splines, geometry = self._load_parent(parent_id)
            first = select_parity_poses(parent_id, graph, splines, geometry["fta_distance_m"])
            second = select_parity_poses(parent_id, graph, splines, geometry["fta_distance_m"])
            self.assertEqual(first, second)
            self.assertEqual(len(first), 8)
            observed = {role: sum(item["role"] == role for item in first) for role in role_counts}
            self.assertEqual(observed, role_counts)
            for role, count in observed.items():
                aggregate[role] += count
        self.assertEqual(aggregate, {"tunnel_interior": 8, "junction_transition": 8, "terminal_approach": 8})

    def test_sdf_freezes_nonduplicated_720_grid_and_cpu_plugin(self):
        pose = {"sensor_xyz_m": [1, 2, 3], "yaw_deg": 45}
        sdf = gazebo_world_sdf("file:///workspace/frozen.obj", [pose])
        self.assertIn("<samples>720</samples>", sdf)
        self.assertIn("<samples>16</samples>", sdf)
        self.assertIn("6.274458660919615", sdf)
        self.assertIn("libgazebo_ros_velodyne_laser.so", sdf)
        self.assertIn("<organize_cloud>true</organize_cloud>", sdf)
        self.assertIn("file:///workspace/frozen.obj", sdf)
        self.assertNotIn("gpu_ray", sdf)

    def test_decode_frozen_organized_cloud_layout(self):
        point_step = 22
        raw = bytearray(720 * 16 * point_step)
        for azimuth in range(720):
            angle = math.radians(azimuth * 0.5)
            for ring in range(16):
                offset = (azimuth * 16 + ring) * point_step
                struct.pack_into("<ffffHf", raw, offset, 10 * math.cos(angle), 10 * math.sin(angle), 0, 1, ring, 0)
        ranges, valid = decode_organized_cloud(bytes(raw), 16, 720, 22)
        self.assertEqual(ranges.shape, (16, 720))
        self.assertTrue(np.all(valid == 1))
        self.assertLess(float(np.max(np.abs(ranges - 10))), 1e-5)

    def test_analytic_box_and_frozen_parity_thresholds(self):
        ranges, valid = analytic_box_ranges()
        self.assertEqual(ranges.shape, (16, 720))
        self.assertTrue(np.all(valid == 1))
        metrics = parity_metrics(ranges, valid, ranges + np.float32(0.01), valid)
        self.assertTrue(metrics["passed"])
        failed = parity_metrics(ranges, valid, ranges + np.float32(0.30), valid)
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
