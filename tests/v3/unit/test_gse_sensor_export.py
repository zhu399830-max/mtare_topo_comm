from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.gse_sequence_inventory import enumerate_directed_traversals
from mtare_topo.data.gse_sensor_export import (
    cast_unique_frame_ranges,
    world_finite_union_qualified_frame_poses,
    world_frame_poses_with_corrections,
    world_unique_frame_poses,
)


class GSESensorExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0]},
                {"id": "n1", "xyz": [10.0, 0.0, 1.0]},
                {"id": "n2", "xyz": [11.5, 0.0, 1.0]},
            ],
            "edges": [
                {"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]},
                {"id": "e1", "node_ids": ["n1", "n2"], "tunnel_ids": [8]},
            ],
        }
        self.splines = {
            "tunnels": [
                {"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]]},
                {"tunnel_id": 8, "points": [[10.0, 0.0, 1.0], [11.5, 0.0, 1.0]]},
            ]
        }

    def _manifest(self) -> list[dict]:
        records = enumerate_directed_traversals(
            parent_id="p",
            split="train",
            graph=self.graph,
            spline_document=self.splines,
        )
        result = []
        frame_offset = 100
        sequence_offset = 200
        for record in records:
            row = record.to_dict()
            row["global_frame_offset"] = frame_offset
            row["global_sequence_offset"] = sequence_offset
            row["unique_frame_count"] = record.unique_frame_count
            result.append(row)
            frame_offset += record.unique_frame_count
            sequence_offset += record.sequence_count
        return result

    def test_unique_poses_retain_short_traversal_but_emit_no_orphan_frames(self) -> None:
        poses = world_unique_frame_poses(
            parent_id="p",
            traversal_manifest=self._manifest(),
            graph=self.graph,
            spline_document=self.splines,
            geometry_parameters={"fta_distance_m": 0.5},
        )
        self.assertEqual(len(poses.global_frame_indices), 22)
        np.testing.assert_array_equal(poses.global_frame_indices, np.arange(100, 122))
        self.assertEqual(set(poses.traversal_ids), {"p:e0:d0", "p:e0:d1"})
        np.testing.assert_allclose(poses.sensor_xyz_m[:, 2] - poses.axis_xyz_m[:, 2], 1.5)
        np.testing.assert_allclose(np.linalg.norm(poses.tangent_world_xyz, axis=1), 1.0)

    def test_full_scan_cast_is_batched_and_masks_invalid_hits(self) -> None:
        poses = world_unique_frame_poses(
            parent_id="p",
            traversal_manifest=self._manifest(),
            graph=self.graph,
            spline_document=self.splines,
            geometry_parameters={"fta_distance_m": 0.5},
        )
        batches: list[int] = []

        def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            batches.append(len(origins))
            result = np.full(directions.shape[:-1], 4.0)
            result[:, 0, 0] = np.inf
            return result

        ranges, valid = cast_unique_frame_ranges(poses, cast_hit_distances=cast, batch_frames=8)
        self.assertEqual(batches, [8, 8, 6])
        self.assertEqual(ranges.shape, (22, 16, 720))
        self.assertEqual(valid.shape, (22, 16, 720))
        self.assertTrue(np.all(ranges[:, 0, 0] == 50.0))
        self.assertTrue(np.all(valid[:, 0, 0] == 0))
        self.assertTrue(np.all(ranges[:, 1:, :] == 4.0))

    def test_union_qualified_poses_move_only_unsafe_endpoint_frames_inward(self) -> None:
        original = world_unique_frame_poses(
            parent_id="p",
            traversal_manifest=self._manifest(),
            graph=self.graph,
            spline_document=self.splines,
            geometry_parameters={"fta_distance_m": 0.5},
        )
        def finite_interval(points: np.ndarray) -> np.ndarray:
            return np.maximum(0.2 - points[:, 0], points[:, 0] - 9.8)

        corrected, corrections = world_finite_union_qualified_frame_poses(
            parent_id="p",
            traversal_manifest=self._manifest(),
            graph=self.graph,
            spline_document=self.splines,
            geometry_parameters={"fta_distance_m": 0.5},
            signed_distance_fields=[finite_interval],
        )

        np.testing.assert_array_equal(
            corrected.global_frame_indices, original.global_frame_indices
        )
        np.testing.assert_array_equal(
            corrected.local_frame_indices, original.local_frame_indices
        )
        self.assertEqual(corrected.traversal_ids, original.traversal_ids)
        self.assertEqual(len(corrections), 4)
        self.assertEqual(
            {(item.traversal_id, item.local_frame_index) for item in corrections},
            {
                ("p:e0:d0", 0), ("p:e0:d0", 10),
                ("p:e0:d1", 0), ("p:e0:d1", 10),
            },
        )
        changed = np.abs(corrected.arc_m - original.arc_m) > 1e-12
        np.testing.assert_array_equal(np.flatnonzero(changed), [0, 10, 11, 21])
        np.testing.assert_array_equal(
            corrected.arc_m[~changed], original.arc_m[~changed]
        )
        np.testing.assert_array_equal(
            corrected.axis_xyz_m[~changed], original.axis_xyz_m[~changed]
        )
        np.testing.assert_array_equal(
            corrected.sensor_xyz_m[~changed], original.sensor_xyz_m[~changed]
        )
        self.assertTrue(np.all(np.diff(corrected.arc_m[:11]) > 0.0))
        self.assertTrue(np.all(np.diff(corrected.arc_m[11:]) > 0.0))
        replayed = world_frame_poses_with_corrections(
            parent_id="p", traversal_manifest=self._manifest(), graph=self.graph,
            spline_document=self.splines, geometry_parameters={"fta_distance_m": 0.5},
            corrections=corrections,
        )
        np.testing.assert_array_equal(replayed.arc_m, corrected.arc_m)
        np.testing.assert_array_equal(replayed.axis_xyz_m, corrected.axis_xyz_m)
        np.testing.assert_array_equal(replayed.sensor_xyz_m, corrected.sensor_xyz_m)
        np.testing.assert_array_equal(replayed.tangent_world_xyz, corrected.tangent_world_xyz)
        np.testing.assert_array_equal(replayed.yaw_deg, corrected.yaw_deg)

    def test_union_qualification_rejects_missing_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "signed-distance"):
            world_finite_union_qualified_frame_poses(
                parent_id="p",
                traversal_manifest=self._manifest(),
                graph=self.graph,
                spline_document=self.splines,
                geometry_parameters={"fta_distance_m": 0.5},
                signed_distance_fields=[],
            )


if __name__ == "__main__":
    unittest.main()
