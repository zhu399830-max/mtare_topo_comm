import tempfile
import unittest
from pathlib import Path

import numpy as np

from execute_cano_c08_hybrid_implicit_qualification_v3 import TRAJECTORIES, frame_identities, freeze_support_queries, geometry_metrics, load_json_list, optimize_window_trajectory, sanitize_visual_faces, solve_constrained_trajectory
from mtare_topo.data.local_implicit_union import LocalUnionWindow


class HybridImplicitQualificationTests(unittest.TestCase):
    def test_frame_identity_keeps_route_order_and_traversal(self):
        traversals = [
            {"route_start_m": 0.0, "route_end_m": 2.0, "traversal_index": 0, "edge_id": "e0", "tunnel_id": "7"},
            {"route_start_m": 2.0, "route_end_m": 4.0, "traversal_index": 1, "edge_id": "e1", "tunnel_id": "8"},
        ]
        result = frame_identities(np.asarray([0.0, 1.0, 2.0, 4.0]), traversals)
        self.assertEqual([item["traversal_index"] for item in result], [0, 0, 1, 1])
        self.assertEqual(result[2]["traversal_arc_m"], 0.0)

    def test_geometry_metrics_are_deterministic(self):
        xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        first = geometry_metrics(xyz)
        self.assertEqual(first, geometry_metrics(xyz.copy()))
        self.assertAlmostEqual(first["maximum_step_m"], 1.0)
        self.assertAlmostEqual(first["maximum_turn_deg"], 90.0)

    def test_list_root_loader_accepts_sealed_traversal_shape(self):
        sealed = load_json_list(TRAJECTORIES / "S01_flat_tree_small_C08_traversals.json")
        self.assertEqual(len(sealed), 108)
        self.assertEqual(sealed[0]["traversal_index"], 0)

    def test_list_root_loader_rejects_object_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.json"
            path.write_text('{"not":"a traversal list"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty list"):
                load_json_list(path)

    def test_visual_sanitation_removes_only_small_area_faces(self):
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1e-14, 0.0, 0.0]])
        faces = np.asarray([[0, 1, 2], [0, 3, 2]])
        sanitized, removed = sanitize_visual_faces(vertices, faces)
        self.assertEqual(removed, 1)
        np.testing.assert_array_equal(sanitized, faces[:1])

    def test_visual_sanitation_matches_v3r_failure_evidence(self):
        patch = TRAJECTORIES.parents[1] / "gate4_20260814_cano_c08_hybrid_implicit_qualification_v3r_seed0/artifacts/patch_S01_flat_tree_small_C08_node_0039_0.100m.npz"
        evidence = np.load(patch)
        sanitized, removed = sanitize_visual_faces(evidence["vertices_m"].astype(float), evidence["faces"].astype(np.int64))
        self.assertEqual(removed, 2)
        self.assertEqual(len(sanitized), 111524)

    def test_window_optimizer_enforces_step_with_minimum_xy_change(self):
        class UnitField:
            def ray_exit_distances(self, origins, directions, *args):
                return np.ones(len(origins))
        axis = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        center_sensor = axis.copy(); center_sensor[1, 2] = 1.0
        window = LocalUnionWindow("n", (2.0, 0.0, 0.0), (1,), 0.1)
        sensor, membership, evidence = optimize_window_trajectory(UnitField(), axis, center_sensor, -1.0, (window,))
        self.assertEqual(membership.tolist(), [False, True, False])
        self.assertLessEqual(np.linalg.norm(sensor[1] - sensor[0]), 2.1 + 1e-7)
        self.assertLess(evidence["maximum_xy_correction_m"], 0.25)
        self.assertGreater(evidence["maximum_xy_correction_m"], 0.0)

    @staticmethod
    def _identities(count):
        return [{"tunnel_id":1,"edge_id":"e","traversal_index":0} for _ in range(count)]

    def test_outer_optimizer_records_support_convergence(self):
        class UnitField:
            pass
        class UnitSupport:
            def support_heights(self, graph_axis, queries):
                return np.full(len(graph_axis),-1.0)
            def downward_distances(self, sensor, graph_axis, queries):
                return sensor[:,2]+1.0
        axis = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        window = LocalUnionWindow("n", (2.0, 0.0, 0.0), (1,), 0.1)
        _, _, evidence, _ = solve_constrained_trajectory(
            (UnitField(),),UnitSupport(),axis,-1.0,(window,),np.ones(len(axis),dtype=int),self._identities(len(axis))
        )
        self.assertEqual(evidence["support_xy_outer_iterations"], 1)
        self.assertEqual(len(evidence["support_xy_convergence"]), 1)
        self.assertTrue(evidence["outside_window_pose_exact"])

    def test_route_support_optimizer_keeps_window_exterior_bitwise_exact(self):
        class UnitField:
            pass
        class OffsetSupport:
            def support_heights(self, graph_axis, queries):
                return np.asarray([-1.0,-1.08,-1.0])
            def downward_distances(self, sensor, graph_axis, queries):
                return sensor[:,2]-self.support_heights(graph_axis,queries)
        axis = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        window = LocalUnionWindow("n", (2.0, 0.0, 0.0), (1,), 0.1)
        sensor,membership,evidence,_=solve_constrained_trajectory(
            (UnitField(),),OffsetSupport(),axis,-1.0,(window,),np.ones(len(axis),dtype=int),self._identities(len(axis))
        )
        nominal=axis.copy(); nominal[:,2]+=0.0
        np.testing.assert_array_equal(sensor[~membership],nominal[~membership])
        self.assertTrue(evidence["outside_window_pose_exact"])
        self.assertLessEqual(abs(sensor[1,2]+1.08-1.0),0.05+1e-9)

    def test_frozen_membership_rejects_overlapping_windows(self):
        axis = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        windows=(LocalUnionWindow("a",(2.0,0.0,0.0),(1,),0.2),LocalUnionWindow("b",(2.0,0.0,0.0),(1,),0.2))
        with self.assertRaisesRegex(RuntimeError,"non-unique frozen window membership"):
            freeze_support_queries(axis,self._identities(len(axis)),windows)


if __name__ == "__main__":
    unittest.main()
