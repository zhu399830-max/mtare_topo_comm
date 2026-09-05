from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.gse_teacher_manifest import world_teacher_manifest


class GSETeacherManifestTest(unittest.TestCase):
    @staticmethod
    def _cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        result = np.empty(directions.shape[:2], dtype=np.float64)
        for row, ray_row in enumerate(directions):
            for column, direction in enumerate(ray_row):
                if abs(direction[2]) > abs(direction[1]):
                    distance = 3.0 if direction[2] > 0.0 else 1.0
                    projection = abs(direction[2])
                else:
                    distance = 2.0
                    projection = abs(direction[1])
                result[row, column] = distance / projection
        return result

    def test_opposite_traversals_share_node_identity_and_short_edge_stays_manifested(self) -> None:
        graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0], "degree": 1},
                {"id": "n1", "xyz": [20.0, 0.0, 0.0], "degree": 3},
                {"id": "n2", "xyz": [40.0, 0.0, 0.0], "degree": 1},
                {"id": "n3", "xyz": [20.0, 1.5, 0.0], "degree": 1},
            ],
            "edges": [
                {"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]},
                {"id": "e1", "node_ids": ["n1", "n2"], "tunnel_ids": [8]},
                {"id": "e2", "node_ids": ["n1", "n3"], "tunnel_ids": [9]},
            ],
        }
        splines = {
            "tunnels": [
                {"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]},
                {"tunnel_id": 8, "points": [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]]},
                {"tunnel_id": 9, "points": [[20.0, 0.0, 0.0], [20.0, 1.5, 0.0]]},
            ]
        }
        geometry = {
            "fta_distance_m": 0.0,
            "tunnels": [
                {"tunnel_id": 7, "radius_m": 2.0},
                {"tunnel_id": 8, "radius_m": 2.0},
                {"tunnel_id": 9, "radius_m": 2.0},
            ],
        }
        result = world_teacher_manifest(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
            cast_distances=self._cast,
        )
        summary = result["summary"]
        self.assertEqual(summary["directed_traversal_count"], 6)
        self.assertEqual(summary["zero_sequence_traversal_count"], 2)
        self.assertEqual(summary["sequence_count"], 64)
        self.assertEqual(len(result["observations"]), 64)
        junction = [row for row in result["observations"] if row["event"] == "junction"]
        self.assertGreater(len(junction), 1)
        self.assertEqual({row["identity"] for row in junction}, {"S01_flat_tree_small_C01:node:n1"})
        positive = [pair for pair in result["association_pairs"] if pair["same_identity"]]
        self.assertGreater(len(positive), 0)
        self.assertTrue(any(pair["pair_kind"] == "cross_traversal_positive" for pair in positive))
        reversed_graph = {**graph, "edges": list(reversed(graph["edges"]))}
        repeated = world_teacher_manifest(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=reversed_graph,
            spline_document=splines,
            geometry_parameters=geometry,
            cast_distances=self._cast,
        )
        self.assertEqual(result, repeated)

    def test_c10_is_rejected_before_raycast(self) -> None:
        called = False

        def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return np.zeros(directions.shape[:2])

        with self.assertRaises(ValueError):
            world_teacher_manifest(
                parent_id="S01_flat_tree_small_C10",
                split="validation",
                graph={"nodes": [], "edges": []},
                spline_document={"tunnels": []},
                geometry_parameters={},
                cast_distances=cast,
            )
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
