from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.primitive_frame_support import summarize_primitive_frame_support, visible_primitive_targets_from_frame_support
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, SweptSuperellipseProvenanceField


class PrimitiveFrameSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        primitives = tuple(SweptSuperellipsePrimitive(
            primitive_id=f"p{index}", centerline_xyz_m=np.asarray([[0., float(index), 0.], [10., float(index), 0.]]),
            endpoint_half_axes_m=((2., 2.), (2., 2.)), endpoint_shape_exponent=(2., 2.),
        ) for index in range(2))
        self.field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)

    def test_summary_preserves_ambiguous_membership_and_aggregates_crop(self) -> None:
        ranges = np.full((16,720), 50., dtype=np.float32); codes = np.zeros((16,720), dtype=np.uint16)
        ranges[8,0] = 3.; codes[8,0] = 1
        ranges[8,1] = 7.; codes[8,1] = 2
        support = summarize_primitive_frame_support(
            range_m=ranges, primitive_membership_code=codes,
            source_sets=((), (0,), (0,1)), field=self.field,
            sensor_xyz_m=np.asarray([0.,0.,0.]), yaw_deg=0.,
        )
        np.testing.assert_array_equal(support.support_ray_count, [2,1])
        self.assertTrue(support.azimuth_support[0,0]); self.assertTrue(support.azimuth_support[0,1]); self.assertTrue(support.azimuth_support[1,1])
        target = visible_primitive_targets_from_frame_support(
            field=self.field, frame_supports=[support]*5,
            current_sensor_xyz_m=np.asarray([0.,0.,0.]), current_yaw_deg=0., maximum_slots=4,
        )
        np.testing.assert_array_equal(target.primitive_index, [0,1,-1,-1])
        np.testing.assert_array_equal(target.support_ray_count, [10,5,0,0])
        self.assertTrue(np.all(target.temporal_visibility[:, :2] == 1))

    def test_invalid_codebook_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "codebook"):
            summarize_primitive_frame_support(
                range_m=np.full((16,720), 50.), primitive_membership_code=np.zeros((16,720),dtype=np.uint16),
                source_sets=((0,),), field=self.field, sensor_xyz_m=np.zeros(3), yaw_deg=0.,
            )

    def test_four_source_membership_is_preserved_without_truncation(self) -> None:
        primitives = tuple(SweptSuperellipsePrimitive(
            primitive_id=f"p{index}",
            centerline_xyz_m=np.asarray([[0., float(index), 0.], [10., float(index), 0.]]),
            endpoint_half_axes_m=((2., 2.), (2., 2.)), endpoint_shape_exponent=(2., 2.),
        ) for index in range(4))
        field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
        ranges = np.full((16, 720), 50., dtype=np.float32)
        codes = np.zeros((16, 720), dtype=np.uint16)
        ranges[8, 0] = 3.; codes[8, 0] = 1
        support = summarize_primitive_frame_support(
            range_m=ranges, primitive_membership_code=codes,
            source_sets=((), (0, 1, 2, 3)), field=field,
            sensor_xyz_m=np.asarray([0., 0., 0.]), yaw_deg=0.,
        )
        np.testing.assert_array_equal(support.support_ray_count, [1, 1, 1, 1])
        target = visible_primitive_targets_from_frame_support(
            field=field, frame_supports=[support] * 5,
            current_sensor_xyz_m=np.asarray([0., 0., 0.]), current_yaw_deg=0., maximum_slots=4,
        )
        np.testing.assert_array_equal(target.primitive_index, [0, 1, 2, 3])
        np.testing.assert_array_equal(target.support_ray_count, [5, 5, 5, 5])


if __name__ == "__main__": unittest.main()
