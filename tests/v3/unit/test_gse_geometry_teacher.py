from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.semantics.geometric_semantics import StructuralEvent
from mtare_topo.teacher.gse_geometry_teacher import (
    GSETeacherConfig,
    classify_structural_event,
    local_geometry_target,
    node_has_geometry_transition,
)


class GSEGeometryTeacherTest(unittest.TestCase):
    def test_ramp_and_reverse_traversal(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 10.0]])
        forward = local_geometry_target(points, 7.0, tunnel_radius_m=3.0)
        reverse = local_geometry_target(points[::-1], 7.0, tunnel_radius_m=3.0)
        self.assertAlmostEqual(forward.width_m, 6.0)
        self.assertAlmostEqual(forward.height_m, 6.0)
        self.assertAlmostEqual(forward.slope_deg, 45.0)
        self.assertAlmostEqual(reverse.slope_deg, -45.0)
        np.testing.assert_allclose(reverse.axis, -np.asarray(forward.axis), atol=1e-12)
        self.assertAlmostEqual(forward.curvature_per_m, reverse.curvature_per_m)

    def test_turn_curvature(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
        target = local_geometry_target(points, 10.0, tunnel_radius_m=2.0, span_m=5.0)
        self.assertAlmostEqual(target.heading_change_deg, 90.0)
        self.assertAlmostEqual(target.curvature_per_m, np.pi / 20.0)

    def test_event_priority_and_transition(self) -> None:
        junction = {"id": "junction", "degree": 3, "incident_tunnel_ids": ["a", "b", "c"]}
        corridor = {"id": "transition", "degree": 2, "incident_tunnel_ids": ["a", "b"]}
        radii = {"a": 2.0, "b": 3.0, "c": 2.5}
        self.assertTrue(node_has_geometry_transition(corridor, radii, minimum_ratio=1.25))
        local = local_geometry_target(np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), 5.0, tunnel_radius_m=2.0)
        event, identity = classify_structural_event(
            traversal_arc_m=5.0,
            traversal_length_m=20.0,
            from_node=junction,
            to_node=corridor,
            local_geometry=local,
            tunnel_radius_by_id=radii,
        )
        self.assertEqual(event, StructuralEvent.JUNCTION)
        self.assertEqual(identity, "junction")
        event, identity = classify_structural_event(
            traversal_arc_m=15.0,
            traversal_length_m=20.0,
            from_node={"id": "plain", "degree": 2, "incident_tunnel_ids": ["a", "a"]},
            to_node=corridor,
            local_geometry=local,
            tunnel_radius_by_id=radii,
            geometry_transition=True,
            geometry_transition_identity="mesh_transition_0",
        )
        self.assertEqual(event, StructuralEvent.GEOMETRY_TRANSITION)
        self.assertEqual(identity, "mesh_transition_0")

    def test_turn_event_away_from_nodes(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0], [15.0, 15.0, 0.0]])
        local = local_geometry_target(points, 15.0, tunnel_radius_m=2.0, span_m=5.0)
        plain = {"id": "plain", "degree": 2, "incident_tunnel_ids": ["a", "a"]}
        event, identity = classify_structural_event(
            traversal_arc_m=15.0,
            traversal_length_m=30.0,
            from_node=plain,
            to_node=plain,
            local_geometry=local,
            tunnel_radius_by_id={"a": 2.0},
            config=GSETeacherConfig(node_event_radius_m=5.0),
        )
        self.assertEqual(event, StructuralEvent.TURN)
        self.assertIsNone(identity)


if __name__ == "__main__":
    unittest.main()
