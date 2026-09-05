from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.teacher.gse_exit_teacher import (
    ExitRouteGeometryContext,
    ExitRouteTarget,
    IncidentExitCandidate,
    audit_exit_route_targets,
    directed_exit_identity,
    incident_exit_candidates,
    make_objective_exit_target,
    spline_vertical_profile,
)
from mtare_topo.teacher.gse_mesh_geometry_teacher import MeshGeometryMeasurement


class GSEExitTeacherTest(unittest.TestCase):
    @staticmethod
    def _context() -> ExitRouteGeometryContext:
        return ExitRouteGeometryContext(
            parent_id="p",
            graph={
                "nodes": [
                    {"id": "a", "xyz": [0.0, 0.0, 0.0]},
                    {"id": "b", "xyz": [20.0, 0.0, 2.0]},
                ],
                "edges": [{"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [7]}],
            },
            spline_document={"tunnels": [{"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [20.0, 0.0, 2.0]]}]},
            geometry_parameters={"fta_distance_m": -1.5},
        )

    @staticmethod
    def _batch_rectangular_cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        del origins
        result = np.empty(directions.shape[:2], dtype=np.float64)
        for row_index, row in enumerate(directions):
            for ray_index, direction in enumerate(row):
                if abs(direction[2]) > abs(direction[1]):
                    distance = 4.0 if direction[2] > 0.0 else 1.0
                    projection = abs(direction[2])
                else:
                    distance = 3.0
                    projection = abs(direction[1])
                result[row_index, ray_index] = distance / max(projection, 1e-12)
        return result

    def test_directed_identity_distinguishes_opposite_exit(self) -> None:
        forward = directed_exit_identity(parent_id="p", edge_id="e", from_node_id="a", to_node_id="b")
        reverse = directed_exit_identity(parent_id="p", edge_id="e", from_node_id="b", to_node_id="a")
        self.assertNotEqual(forward, reverse)

    def test_vertical_profile_preserves_ramp_direction(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 2.0]])
        forward = spline_vertical_profile(points, current_arc_m=5.0, direction_sign=1)
        reverse = spline_vertical_profile(points, current_arc_m=15.0, direction_sign=-1)
        expected = np.asarray([2.5, 5.0, 7.5, 10.0]) * 2.0 / np.sqrt(404.0)
        np.testing.assert_allclose(forward, expected, atol=1e-12)
        np.testing.assert_allclose(reverse, -expected, atol=1e-12)

    def test_target_uses_mesh_opening_and_robot_relative_heading(self) -> None:
        measurement = MeshGeometryMeasurement(
            left_m=2.0,
            right_m=3.0,
            up_m=4.0,
            down_m=1.0,
            width_m=5.0,
            height_m=5.0,
            valid_rays=(5, 5, 5, 5),
        )
        target = make_objective_exit_target(
            identity="p:e:a->b",
            heading_world_deg=10.0,
            robot_yaw_deg=350.0,
            cross_section=measurement,
            vertical_profile_m=(0.1, 0.2, 0.3, 0.4),
            source_tunnel_ids=(7, 7),
        )
        self.assertAlmostEqual(target.heading_robot_deg, 20.0)
        self.assertAlmostEqual(target.opening_width_m, 5.0)
        self.assertEqual(target.source_tunnel_ids, ("7",))

    def test_node_candidates_are_incident_only_and_same_tunnel_edges_do_not_collapse(self) -> None:
        graph = {
            "nodes": [{"id": value} for value in ("a", "b", "c", "x", "y")],
            "edges": [
                {"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [7]},
                {"id": "e1", "node_ids": ["a", "c"], "tunnel_ids": [7]},
                {"id": "stacked", "node_ids": ["x", "y"], "tunnel_ids": [9]},
            ],
        }
        candidates = incident_exit_candidates(parent_id="p", graph=graph, current_edge_id="e0", selected_node_id="a")
        self.assertEqual([item.edge_id for item in candidates], ["e0", "e1"])
        self.assertEqual([item.to_node_id for item in candidates], ["b", "c"])
        self.assertEqual([item.source_tunnel_ids for item in candidates], [("7",), ("7",)])
        self.assertNotEqual(candidates[0].identity, candidates[1].identity)

    def test_corridor_candidates_preserve_both_current_edge_directions(self) -> None:
        graph = {
            "nodes": [{"id": value} for value in ("a", "b", "x", "y")],
            "edges": [
                {"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [7]},
                {"id": "nonincident", "node_ids": ["x", "y"], "tunnel_ids": [7]},
            ],
        }
        candidates = incident_exit_candidates(parent_id="p", graph=graph, current_edge_id="e0", selected_node_id=None)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({(item.from_node_id, item.to_node_id) for item in candidates}, {("a", "b"), ("b", "a")})
        self.assertEqual({item.edge_id for item in candidates}, {"e0"})

    def test_route_target_preserves_forward_and_reverse_ramp_semantics(self) -> None:
        context = self._context()
        graph = context.graph
        forward, reverse = incident_exit_candidates(
            parent_id="p", graph=graph, current_edge_id="e0", selected_node_id=None
        )
        target_forward = context.route_target(
            candidate=forward,
            current_axis_xyz_m=[0.0, 0.0, 0.0],
            robot_yaw_deg=350.0,
            selected_node_id=None,
        )
        target_reverse = context.route_target(
            candidate=reverse,
            current_axis_xyz_m=[20.0, 0.0, 2.0],
            robot_yaw_deg=170.0,
            selected_node_id=None,
        )
        self.assertAlmostEqual(target_forward.heading_robot_deg, 10.0)
        self.assertAlmostEqual(target_reverse.heading_robot_deg, 10.0)
        self.assertTrue(all(value > 0.0 for value in target_forward.vertical_profile_m))
        self.assertTrue(all(value < 0.0 for value in target_reverse.vertical_profile_m))
        self.assertGreater(target_forward.representative_axis_xyz_m[2], 0.0)
        self.assertLess(target_reverse.representative_axis_xyz_m[2], 2.0)

    def test_node_route_must_point_away_from_selected_incident_node(self) -> None:
        context = self._context()
        candidate = incident_exit_candidates(
            parent_id="p", graph=context.graph, current_edge_id="e0", selected_node_id="a"
        )[0]
        target = context.route_target(
            candidate=candidate,
            current_axis_xyz_m=[0.1, 0.0, 0.01],
            robot_yaw_deg=0.0,
            selected_node_id="a",
        )
        self.assertEqual(target.candidate.from_node_id, "a")
        self.assertGreater(target.representative_axis_xyz_m[0], 0.0)
        with self.assertRaisesRegex(ValueError, "point away"):
            context.route_target(
                candidate=candidate,
                current_axis_xyz_m=[0.1, 0.0, 0.01],
                robot_yaw_deg=0.0,
                selected_node_id="b",
            )

    def test_audit_separates_los_visibility_from_width_validity(self) -> None:
        candidate = IncidentExitCandidate("p:e:a->b", "e", "a", "b", ("7",))
        targets = (
            ExitRouteTarget(candidate, 0.0, (2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            ExitRouteTarget(candidate, 0.0, (4.0, 0.0, 0.0), (4.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        )

        calls = 0

        def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            if directions.shape[1] == 20:
                values = self._batch_rectangular_cast(origins, directions)
                values[1, :5] = np.inf
                return values
            return np.asarray([[np.inf], [3.0]], dtype=np.float64)

        audited = audit_exit_route_targets(
            targets,
            current_sensor_xyz_m=np.zeros((2, 3), dtype=np.float64),
            cast_distances=cast,
        )
        self.assertEqual(calls, 2)
        self.assertTrue(audited[0].visible)
        self.assertTrue(audited[0].opening_width_valid)
        self.assertAlmostEqual(audited[0].opening_width_m or 0.0, 6.0)
        self.assertFalse(audited[1].visible)
        self.assertFalse(audited[1].opening_width_valid)
        self.assertIsNone(audited[1].opening_width_m)
        self.assertEqual(audited[1].identity, candidate.identity)

    def test_audit_accepts_zero_distance_target_but_rejects_bad_batch_shape(self) -> None:
        candidate = IncidentExitCandidate("p:e:a->b", "e", "a", "b", ("7",))
        target = ExitRouteTarget(
            candidate, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
        )

        def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            if directions.shape[1] == 20:
                return self._batch_rectangular_cast(origins, directions)
            return np.asarray([np.inf], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "LOS distance shape"):
            audit_exit_route_targets(
                (target,),
                current_sensor_xyz_m=np.zeros((1, 3), dtype=np.float64),
                cast_distances=cast,
            )

        def valid_cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            if directions.shape[1] == 20:
                return self._batch_rectangular_cast(origins, directions)
            return np.asarray([[np.inf]], dtype=np.float64)

        audited = audit_exit_route_targets(
            (target,),
            current_sensor_xyz_m=np.zeros((1, 3), dtype=np.float64),
            cast_distances=valid_cast,
        )
        self.assertTrue(audited[0].visible)
        self.assertEqual(audited[0].line_of_sight_distance_m, 0.0)

    def test_route_and_audit_reject_nonfinite_inputs(self) -> None:
        context = self._context()
        candidate = incident_exit_candidates(
            parent_id="p", graph=context.graph, current_edge_id="e0", selected_node_id="a"
        )[0]
        with self.assertRaises(ValueError):
            context.route_target(
                candidate=candidate,
                current_axis_xyz_m=[np.nan, 0.0, 0.0],
                robot_yaw_deg=0.0,
                selected_node_id="a",
            )
        with self.assertRaises(ValueError):
            context.route_target(
                candidate=candidate,
                current_axis_xyz_m=[0.0, 0.0, 0.0],
                robot_yaw_deg=np.nan,
                selected_node_id="a",
            )


if __name__ == "__main__":
    unittest.main()
