from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.data.primitive_relation_sensor_export import render_primitive_sensor_frame
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipsePrimitive,
    SweptSuperellipseProvenanceField,
)


class _FakeRaycaster:
    def ray_exit_hits(self, origins_xyz_m, directions_xyz, initial_inside, *, maximum_m=50.0):
        self.inside = np.asarray(initial_inside)
        count = len(origins_xyz_m)
        result = [None] * count
        result[0] = PrimitiveRayHit(3.0, (3.0, 0.0, 0.0), ("p0",), True)
        result[1] = PrimitiveRayHit(4.0, (4.0, 0.0, 0.0), ("p0", "p1"), False)
        result[2] = PrimitiveRayHit(0.01, (0.01, 0.0, 0.0), ("p0",), True)
        return tuple(result)


class PrimitiveRelationSensorExportTest(unittest.TestCase):
    def test_frame_preserves_complete_membership_and_masks_invalid_hits(self) -> None:
        primitives = tuple(
            SweptSuperellipsePrimitive(
                primitive_id=f"p{index}",
                centerline_xyz_m=np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                endpoint_half_axes_m=((2.0, 2.0), (2.0, 2.0)),
                endpoint_shape_exponent=(2.0, 2.0),
            )
            for index in range(2)
        )
        field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
        codebook = PrimitiveMembershipCodebook(field.primitive_ids)
        raycaster = _FakeRaycaster()
        frame = render_primitive_sensor_frame(
            raycaster=raycaster, field=field, codebook=codebook,
            sensor_xyz_m=np.asarray([1.0, 0.0, 0.0]), yaw_deg=0.0,
        )
        self.assertEqual(frame.range_m[0, 0], 3.0)
        self.assertEqual(frame.range_m[0, 1], 4.0)
        self.assertEqual(frame.range_m[0, 2], 50.0)
        self.assertEqual(frame.ambiguous_ray_count, 1)
        self.assertEqual(codebook.decode(frame.primitive_membership_code[:1, :3]), ((0,), (0, 1), ()))
        self.assertTrue(np.all(raycaster.inside))

    def test_outside_origin_fails_closed(self) -> None:
        primitive = SweptSuperellipsePrimitive(
            primitive_id="p0", centerline_xyz_m=np.asarray([[0., 0., 0.], [1., 0., 0.]]),
            endpoint_half_axes_m=((1., 1.), (1., 1.)), endpoint_shape_exponent=(2., 2.),
        )
        field = SweptSuperellipseProvenanceField((primitive,), spacing_m=.025)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            render_primitive_sensor_frame(
                raycaster=_FakeRaycaster(), field=field,
                codebook=PrimitiveMembershipCodebook(field.primitive_ids),
                sensor_xyz_m=np.asarray([-2., 0., 0.]), yaw_deg=0.,
            )


if __name__ == "__main__":
    unittest.main()
