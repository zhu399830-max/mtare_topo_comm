from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.integration.range_image_adapter import registered_points_to_range_image


class MTAReRangeImageAdapterTests(unittest.TestCase):
    def test_exact_synthetic_grid_round_trip_with_full_orientation(self):
        local = lidar_local_directions().astype(np.float64)
        expected = np.linspace(2.0, 30.0, local.shape[0] * local.shape[1]).reshape(local.shape[:2])
        local_points = local * expected[..., None]
        orientation = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
        orientation /= np.linalg.norm(orientation)
        x, y, z, w = orientation
        rotation = np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        origin = np.asarray([4.0, -2.0, 1.5])
        world = local_points.reshape(-1, 3) @ rotation.T + origin
        image, valid, audit = registered_points_to_range_image(
            world[::-1], sensor_origin_world=origin, sensor_orientation_xyzw=orientation
        )
        self.assertTrue(np.all(valid == 1))
        self.assertLess(float(np.max(np.abs(image - expected))), 2e-5)
        self.assertEqual(audit.unique_valid_cells, 16 * 720)
        self.assertEqual(audit.out_of_ring_points, 0)

    def test_duplicate_cell_keeps_first_return(self):
        points = np.asarray([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        image, valid, audit = registered_points_to_range_image(
            points,
            sensor_origin_world=[0, 0, 0],
            sensor_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            elevation_rows_deg=np.asarray([0.0]),
        )
        self.assertEqual(float(image[0, 0]), 2.0)
        self.assertEqual(int(valid[0, 0]), 1)
        self.assertEqual(audit.duplicate_cell_returns, 1)

    def test_invalid_and_off_ring_points_are_audited(self):
        points = np.asarray([
            [np.nan, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 1.0],
        ])
        _, valid, audit = registered_points_to_range_image(
            points,
            sensor_origin_world=[0, 0, 0],
            sensor_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(int(valid.sum()), 0)
        self.assertEqual(audit.input_points, 3)
        self.assertEqual(audit.finite_points, 2)
        self.assertEqual(audit.out_of_range_points, 1)
        self.assertEqual(audit.out_of_ring_points, 1)


if __name__ == "__main__":
    unittest.main()
