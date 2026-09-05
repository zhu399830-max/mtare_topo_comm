from __future__ import annotations

from copy import deepcopy
import os
import subprocess
import sys
import unittest

import numpy as np

from mtare_topo.data.gse_exit_token_audit import (
    selected_node_id_from_observation,
    world_exit_token_audit,
)
from mtare_topo.data.gse_dataset_export import pack_world_teacher_targets
from mtare_topo.data.gse_sequence_inventory import enumerate_directed_traversals
from mtare_topo.data.gse_sensor_export import world_unique_frame_poses


class GSEExitTokenAuditTest(unittest.TestCase):
    @staticmethod
    def _world() -> tuple[dict, dict, dict]:
        graph = {
            "nodes": [
                {"id": "a", "xyz": [0.0, 0.0, 0.0], "degree": 3},
                {"id": "b", "xyz": [20.0, 0.0, 0.0], "degree": 1},
                {"id": "c", "xyz": [0.0, 20.0, 1.0], "degree": 1},
                {"id": "d", "xyz": [-20.0, 0.0, -1.0], "degree": 1},
            ],
            "edges": [
                {"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [0]},
                {"id": "e1", "node_ids": ["a", "c"], "tunnel_ids": [1]},
                {"id": "e2", "node_ids": ["a", "d"], "tunnel_ids": [2]},
            ],
        }
        splines = {
            "tunnels": [
                {"tunnel_id": 0, "points": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]},
                {"tunnel_id": 1, "points": [[0.0, 0.0, 0.0], [0.0, 20.0, 1.0]]},
                {"tunnel_id": 2, "points": [[0.0, 0.0, 0.0], [-20.0, 0.0, -1.0]]},
            ]
        }
        geometry = {"fta_distance_m": -1.5}
        return graph, splines, geometry

    @staticmethod
    def _open_tunnel_cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        del origins
        if directions.shape[1] == 1:
            return np.full(directions.shape[:2], np.inf, dtype=np.float64)
        result = np.empty(directions.shape[:2], dtype=np.float64)
        for row_index, row in enumerate(directions):
            for ray_index, direction in enumerate(row):
                if abs(direction[2]) > max(abs(direction[0]), abs(direction[1])):
                    distance = 3.0 if direction[2] > 0.0 else 1.0
                    projection = abs(direction[2])
                else:
                    distance = 2.0
                    projection = max(abs(direction[0]), abs(direction[1]))
                result[row_index, ray_index] = distance / max(projection, 1e-12)
        return result

    def test_node_identity_is_decoded_only_for_node_events(self) -> None:
        row = {"parent_id": "p", "event": "junction", "identity": "p:node:a"}
        self.assertEqual(selected_node_id_from_observation(row, graph_node_ids={"a"}), "a")
        self.assertIsNone(
            selected_node_id_from_observation(
                {"parent_id": "p", "event": "turn", "identity": "p:edge:e:event:0"},
                graph_node_ids={"a"},
            )
        )
        with self.assertRaises(ValueError):
            selected_node_id_from_observation(
                {"parent_id": "p", "event": "terminal", "identity": "q:node:a"},
                graph_node_ids={"a"},
            )

    def test_data_module_imports_from_a_cold_process(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "import mtare_topo.data.gse_exit_token_audit"],
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_world_audit_emits_only_incident_visible_candidates(self) -> None:
        graph, splines, geometry = self._world()
        traversal = next(
            record
            for record in enumerate_directed_traversals(
                parent_id="p", split="train", graph=graph, spline_document=splines
            )
            if record.traversal_id == "p:e0:d0"
        )
        traversal_row = {
            **traversal.to_dict(),
            "global_frame_offset": 0,
            "global_sequence_offset": 0,
            "unique_frame_count": traversal.unique_frame_count,
            "referenced_frame_count": traversal.referenced_frame_count,
        }
        observation = {
            "observation_id": "p:e0:d0:s000000",
            "parent_id": "p",
            "split": "train",
            "traversal_id": "p:e0:d0",
            "edge_id": "e0",
            "from_node_id": "a",
            "to_node_id": "b",
            "sequence_index": 0,
            "frame_index": 4,
            "traversal_arc_m": 4.0,
            "global_sequence_index": 0,
            "global_frame_references": [0, 1, 2, 3, 4],
            "event": "junction",
            "identity": "p:node:a",
            "geometry_valid": True,
            "width_m": 4.0,
            "height_m": 4.0,
            "slope_deg": 0.0,
            "curvature_per_m": 0.0,
        }
        result = world_exit_token_audit(
            parent_id="p",
            split="train",
            observations=[observation],
            traversal_manifest=[traversal_row],
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
            cast_distances=self._open_tunnel_cast,
        )
        self.assertEqual(result["summary"]["candidate_count"], 3)
        self.assertEqual(result["summary"]["visible_token_count"], 3)
        self.assertEqual(result["summary"]["zero_visible_node_event_count"], 0)
        self.assertEqual(result["summary"]["nonincident_candidate_count"], 0)
        self.assertEqual(
            {(row["candidate"]["edge_id"], row["candidate"]["from_node_id"]) for row in result["rows"]},
            {("e0", "a"), ("e1", "a"), ("e2", "a")},
        )
        self.assertTrue(all(row["token"]["opening_width_valid"] for row in result["rows"]))

        adjusted_rows = deepcopy(result["rows"])
        adjusted_rows[0]["token"]["visible"] = False
        adjusted_rows[1]["token"]["opening_width_valid"] = False
        adjusted_rows[1]["token"]["opening_width_m"] = None
        poses = world_unique_frame_poses(
            parent_id="p",
            traversal_manifest=[traversal_row],
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
        )
        exit_map = {
            row["candidate"]["identity"]: index for index, row in enumerate(adjusted_rows)
        }
        packed = pack_world_teacher_targets(
            parent_id="p",
            observations=[observation],
            exit_token_rows=adjusted_rows,
            poses=poses,
            association_identity_to_index={"p:node:a": 0},
            exit_identity_to_index=exit_map,
        )
        self.assertEqual(int(packed.arrays["exit_mask"].sum()), 2)
        self.assertEqual(int(packed.arrays["exit_invisible_count"][0]), 1)
        self.assertEqual(int(packed.arrays["exit_width_valid_mask"].sum()), 1)
        np.testing.assert_allclose(packed.arrays["local_axis_robot"][0], [1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            pack_world_teacher_targets(
                parent_id="p",
                observations=[observation],
                exit_token_rows=adjusted_rows,
                poses=poses,
                association_identity_to_index={"p:node:a": 0},
                exit_identity_to_index=exit_map,
                maximum_exit_tokens=256,
            )

        malformed_profile = deepcopy(adjusted_rows)
        malformed_profile[2]["token"]["vertical_profile_m"] = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            pack_world_teacher_targets(
                parent_id="p",
                observations=[observation],
                exit_token_rows=malformed_profile,
                poses=poses,
                association_identity_to_index={"p:node:a": 0},
                exit_identity_to_index=exit_map,
            )

        mismatched_identity = deepcopy(adjusted_rows)
        mismatched_identity[2]["token"]["identity"] = "p:edge:wrong:d0"
        with self.assertRaises(RuntimeError):
            pack_world_teacher_targets(
                parent_id="p",
                observations=[observation],
                exit_token_rows=mismatched_identity,
                poses=poses,
                association_identity_to_index={"p:node:a": 0},
                exit_identity_to_index=exit_map,
            )

    def test_c10_is_rejected_before_raycast(self) -> None:
        called = False

        def forbidden(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            raise AssertionError("raycast must not run")

        graph, splines, geometry = self._world()
        with self.assertRaises(ValueError):
            world_exit_token_audit(
                parent_id="S01_C10",
                split="validation",
                observations=[],
                traversal_manifest=[],
                graph=graph,
                spline_document=splines,
                geometry_parameters=geometry,
                cast_distances=forbidden,
            )
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
