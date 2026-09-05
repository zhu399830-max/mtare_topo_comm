from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry, world_primitive_relation_sequence_references


class PrimitiveRelationSequencesTest(unittest.TestCase):
    def test_relative_odometry_uses_current_sensor_frame_and_wraps_yaw(self) -> None:
        xyz = np.asarray([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.], [-1., 1., 1.], [0., 0., 1.]])
        yaw = np.asarray([350., 355., 0., 5., 10.])
        value = causal_relative_odometry(xyz, yaw)
        expected_first = np.asarray([np.cos(np.radians(10.)), -np.sin(np.radians(10.)), -1.])
        np.testing.assert_allclose(value.translation_current_sensor_m[0], expected_first, atol=1e-12)
        np.testing.assert_allclose(value.yaw_current_sensor_deg, [-20., -15., -10., -5., 0.], atol=1e-12)
        np.testing.assert_array_equal(value.translation_current_sensor_m[-1], np.zeros(3))

    def test_paired_variant_preserves_source_references(self) -> None:
        manifest = [
            {"parent_id":"p","traversal_id":"p:e0:d0","global_frame_offset":100,"global_sequence_offset":20,"sequence_count":3,"unique_frame_count":7},
            {"parent_id":"p","traversal_id":"p:e1:d0","global_frame_offset":107,"global_sequence_offset":23,"sequence_count":2,"unique_frame_count":6},
        ]
        result = world_primitive_relation_sequence_references(
            parent_id="p", traversal_manifest=manifest,
            shard_global_frame_indices=np.arange(100,113), realization_index=2,
            source_sequence_population=1000,
        )
        np.testing.assert_array_equal(result.source_global_sequence_index, np.arange(20,25))
        np.testing.assert_array_equal(result.variant_global_sequence_index, np.arange(2020,2025))
        np.testing.assert_array_equal(result.frame_row[0], np.arange(5))
        np.testing.assert_array_equal(result.frame_row[-1], np.arange(8,13))
        self.assertEqual(result.traversal_id, ("p:e0:d0",)*3 + ("p:e1:d0",)*2)


if __name__ == "__main__": unittest.main()
