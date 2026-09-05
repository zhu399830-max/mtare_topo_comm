from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.gse_mesh_teacher_inventory import world_mesh_teacher_inventory


class GSEMeshTeacherInventoryTest(unittest.TestCase):
    @staticmethod
    def _step_tunnel_cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        result = np.empty(directions.shape[:2], dtype=np.float64)
        for row, (origin, ray_row) in enumerate(zip(origins, directions)):
            half_width = 2.0 if origin[0] < 15.0 else 3.0
            for column, direction in enumerate(ray_row):
                if abs(direction[2]) > abs(direction[1]):
                    distance = 3.0 if direction[2] > 0.0 else 1.0
                    projection = abs(direction[2])
                else:
                    distance = half_width
                    projection = abs(direction[1])
                result[row, column] = distance / projection
        return result

    def test_real_mesh_contract_labels_sustained_geometry_change(self) -> None:
        graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0], "degree": 1, "incident_tunnel_ids": [7]},
                {"id": "n1", "xyz": [30.0, 0.0, 0.0], "degree": 1, "incident_tunnel_ids": [7]},
            ],
            "edges": [{"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]}],
        }
        splines = {"tunnels": [{"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]}]}
        geometry = {
            "fta_distance_m": 0.0,
            "tunnels": [{"tunnel_id": 7, "radius_m": 2.0}],
        }
        result = world_mesh_teacher_inventory(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
            cast_distances=self._step_tunnel_cast,
        )
        self.assertEqual(result["directed_traversal_count"], 2)
        self.assertEqual(result["sequence_count"], 52)
        self.assertEqual(result["labelled_sequence_count"], 52)
        self.assertEqual(result["continuous_geometry_target_count"], 52)
        self.assertEqual(result["missing_continuous_geometry_target_count"], 0)
        self.assertEqual(result["complete_mesh_frame_count"], 60)
        self.assertEqual(result["incomplete_mesh_frame_count"], 0)
        self.assertGreater(result["event_counts"]["geometry_transition"], 0)
        self.assertGreater(result["event_counts"]["terminal"], 0)
        self.assertAlmostEqual(result["geometry"]["height_m"]["median"], 4.0)

    def test_c10_rejected_before_raycast(self) -> None:
        called = False

        def forbidden(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return np.zeros(directions.shape[:2])

        with self.assertRaises(ValueError):
            world_mesh_teacher_inventory(
                parent_id="S01_flat_tree_small_C10",
                split="validation",
                graph={"nodes": [], "edges": []},
                spline_document={"tunnels": []},
                geometry_parameters={},
                cast_distances=forbidden,
            )
        self.assertFalse(called)

    def test_short_traversal_without_five_frame_history_is_not_raycast(self) -> None:
        graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0], "degree": 1},
                {"id": "n1", "xyz": [10.0, 0.0, 0.0], "degree": 2},
                {"id": "n2", "xyz": [11.5, 0.0, 0.0], "degree": 1},
            ],
            "edges": [
                {"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]},
                {"id": "e1", "node_ids": ["n1", "n2"], "tunnel_ids": [8]},
            ],
        }
        splines = {
            "tunnels": [
                {"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]},
                {"tunnel_id": 8, "points": [[10.0, 0.0, 0.0], [11.5, 0.0, 0.0]]},
            ]
        }
        geometry = {
            "fta_distance_m": 0.0,
            "tunnels": [
                {"tunnel_id": 7, "radius_m": 2.0},
                {"tunnel_id": 8, "radius_m": 2.0},
            ],
        }
        batch_sizes: list[int] = []

        def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            batch_sizes.append(len(origins))
            return self._step_tunnel_cast(origins, directions)

        result = world_mesh_teacher_inventory(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
            cast_distances=cast,
        )
        self.assertEqual(result["directed_traversal_count"], 4)
        self.assertEqual(result["sequence_count"], 12)
        self.assertEqual(result["complete_mesh_frame_count"], 20)
        self.assertEqual(batch_sizes, [10, 10])


if __name__ == "__main__":
    unittest.main()
