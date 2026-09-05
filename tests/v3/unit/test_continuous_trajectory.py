from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.topology.continuous_trajectory import (
    build_spline_route,
    doubled_euler_traversals,
    project_to_polyline,
    resample_route,
)


class ContinuousTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = {
            "nodes": [
                {"id": "a", "xyz": [0.0, 0.1, 0.0], "degree": 1},
                {"id": "b", "xyz": [2.0, -0.1, 0.0], "degree": 1},
            ],
            "edges": [{"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [7]}],
        }
        self.splines = {"tunnels": [{"tunnel_id": 7, "points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}]}

    def test_projection_reports_arc_and_gap(self) -> None:
        result = project_to_polyline([0.5, 0.2, 0.0], np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float))
        self.assertAlmostEqual(result.arc_m, 0.5)
        self.assertAlmostEqual(result.error_m, 0.2)

    def test_doubled_route_keeps_edge_identity_and_continuity(self) -> None:
        traversals = doubled_euler_traversals(self.graph)
        self.assertEqual(len(traversals), 2)
        self.assertEqual({item["edge_id"] for item in traversals}, {"e0"})
        self.assertEqual(traversals[0]["to_node"], traversals[1]["from_node"])

    def test_route_includes_explicit_connectors_and_exact_resampling(self) -> None:
        route, traversals, connectors = build_spline_route(self.graph, self.splines, 0.5)
        self.assertAlmostEqual(sum(item["length_m"] for item in traversals), 4.4)
        self.assertAlmostEqual(max(item["length_m"] for item in connectors), 0.1)
        samples, tangents, distances = resample_route(route, 2.0)
        self.assertEqual(samples.shape, (3, 3))
        self.assertEqual(tangents.shape, (3, 3))
        np.testing.assert_allclose(distances, [0.0, 2.0, 4.0])

    def test_connector_limit_is_hard(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds connector contract"):
            build_spline_route(self.graph, self.splines, 0.05)


if __name__ == "__main__":
    unittest.main()
