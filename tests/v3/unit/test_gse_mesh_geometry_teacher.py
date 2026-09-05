from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.teacher.gse_mesh_geometry_teacher import (
    MeshGeometryTeacherConfig,
    apply_mesh_cross_section,
    geometry_transition_mask,
    local_cross_section_rays,
    measure_mesh_geometry,
    measure_mesh_geometry_batch,
    sensor_origin_from_axis,
)
from mtare_topo.teacher.gse_geometry_teacher import local_geometry_target


class GSEMeshGeometryTeacherTest(unittest.TestCase):
    @staticmethod
    def _rectangular_tunnel_cast(origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        del origin
        distances = np.empty(len(directions), dtype=np.float64)
        for index, direction in enumerate(directions):
            if abs(direction[2]) > abs(direction[1]):
                primary_distance = 4.0 if direction[2] > 0.0 else 1.0
                projection = abs(direction[2])
            else:
                primary_distance = 5.0
                projection = abs(direction[1])
            distances[index] = primary_distance / projection
        # One isolated mesh hole in each five-ray fan must not change the median.
        distances[[0, 5, 10, 15]] = np.inf
        return distances

    def test_sensor_origin_reuses_sealed_floor_axis_contract(self) -> None:
        sensor = sensor_origin_from_axis([1.0, 2.0, 3.0], fta_distance_m=-1.7)
        np.testing.assert_allclose(sensor, [1.0, 2.0, 2.3])

    def test_cross_section_measurement_is_projected_and_hole_robust(self) -> None:
        result = measure_mesh_geometry(
            origin_xyz_m=[0.0, 0.0, 1.0],
            tangent_xyz=[1.0, 0.0, 0.0],
            cast_distances=self._rectangular_tunnel_cast,
        )
        self.assertAlmostEqual(result.left_m, 5.0)
        self.assertAlmostEqual(result.right_m, 5.0)
        self.assertAlmostEqual(result.up_m, 4.0)
        self.assertAlmostEqual(result.down_m, 1.0)
        self.assertAlmostEqual(result.width_m, 10.0)
        self.assertAlmostEqual(result.height_m, 5.0)
        self.assertEqual(result.valid_rays, (4, 4, 4, 4))

        reverse = measure_mesh_geometry(
            origin_xyz_m=[0.0, 0.0, 1.0],
            tangent_xyz=[-1.0, 0.0, 0.0],
            cast_distances=self._rectangular_tunnel_cast,
        )
        self.assertAlmostEqual(reverse.width_m, result.width_m)
        self.assertAlmostEqual(reverse.height_m, result.height_m)

        spline = local_geometry_target(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]]),
            5.0,
            tunnel_radius_m=2.0,
        )
        combined = apply_mesh_cross_section(spline, result)
        self.assertAlmostEqual(combined.width_m, 10.0)
        self.assertAlmostEqual(combined.height_m, 5.0)
        self.assertEqual(combined.axis, spline.axis)

    def test_incomplete_side_fails_closed(self) -> None:
        def incomplete(origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
            values = self._rectangular_tunnel_cast(origin, directions)
            values[:5] = np.inf
            return values

        with self.assertRaisesRegex(ValueError, "insufficient"):
            measure_mesh_geometry(
                origin_xyz_m=[0.0, 0.0, 1.0],
                tangent_xyz=[1.0, 0.0, 0.0],
                cast_distances=incomplete,
            )

    def test_batch_measurement_marks_only_incomplete_rows(self) -> None:
        def batch(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
            rows = np.stack(
                [self._rectangular_tunnel_cast(origin, direction) for origin, direction in zip(origins, directions)]
            )
            rows[1, :5] = np.inf
            return rows

        results, complete = measure_mesh_geometry_batch(
            origins_xyz_m=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            tangents_xyz=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cast_distances=batch,
        )
        self.assertEqual(complete.tolist(), [True, False])
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])

    def test_transition_uses_sustained_one_metre_change(self) -> None:
        width = np.asarray([4.0] * 10 + [6.0] * 10)
        height = np.asarray([4.0] * 20)
        labels = geometry_transition_mask(width, height, spacing_m=1.0)
        self.assertTrue(labels[9])
        self.assertTrue(labels[10])
        self.assertFalse(labels[4])
        self.assertFalse(labels[15])

        subthreshold = geometry_transition_mask(
            [4.0] * 10 + [4.9] * 10,
            height,
            spacing_m=1.0,
        )
        self.assertFalse(np.any(subthreshold))

    def test_ray_contract_is_twenty_unit_directions(self) -> None:
        rays, factors, groups = local_cross_section_rays(
            [1.0, 0.0, 0.0],
            fan_offsets_deg=MeshGeometryTeacherConfig().fan_offsets_deg,
        )
        self.assertEqual(rays.shape, (20, 3))
        self.assertEqual(factors.shape, (20,))
        self.assertEqual(tuple(group.stop - group.start for group in groups), (5, 5, 5, 5))
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
