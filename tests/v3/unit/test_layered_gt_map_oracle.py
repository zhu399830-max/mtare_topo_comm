from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.oracle.layered_gt_map import LayeredGTMapConfig, LayeredGTMapOracle, load_binary_xyz_ply


def surface(mask: np.ndarray, z: float, resolution: float = 0.2) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    return np.c_[columns * resolution, rows * resolution, np.full(len(rows), z)].astype(np.float32)


class LayeredGTMapOracleTests(unittest.TestCase):
    def config(self) -> LayeredGTMapConfig:
        return LayeredGTMapConfig(local_size_m=12.0, maximum_reach_m=6.0, exit_minimum_reach_m=3.0)

    def test_stacked_surfaces_are_selected_by_pose_height(self):
        mask = np.ones((61, 61), dtype=bool)
        points = np.r_[surface(mask, 0.0), surface(mask, 4.0)]
        oracle = LayeredGTMapOracle(points, self.config())
        low = oracle.local_layer((6.0, 6.0, 0.75), 0.0)
        high = oracle.local_layer((6.0, 6.0, 4.75), 0.0)
        self.assertAlmostEqual(low.selected_center_support_z_m, 0.0)
        self.assertAlmostEqual(high.selected_center_support_z_m, 4.0)
        self.assertTrue(np.allclose(low.support_height_m[low.support_valid], 0.0))
        self.assertTrue(np.allclose(high.support_height_m[high.support_valid], 4.0))

    def test_wall_blocks_layer_without_deleting_other_floor(self):
        mask = np.ones((61, 61), dtype=bool)
        lower = surface(mask, 0.0)
        wall_rows = np.arange(61)
        wall = np.c_[np.full(61 * 4, 6.0), np.repeat(wall_rows * 0.2, 4), np.tile([0.4, 0.8, 1.2, 1.5], 61)]
        upper = surface(mask, 4.0)
        oracle = LayeredGTMapOracle(np.r_[lower, wall, upper], self.config())
        evidence = oracle.local_layer((4.0, 6.0, 0.75), 0.0)
        self.assertTrue(evidence.obstacle.any())
        self.assertFalse(evidence.connected_traversable[:, evidence.connected_traversable.shape[1] // 2 + 11 :].any())

    def test_t_junction_has_three_exit_components_and_junction_role(self):
        mask = np.zeros((101, 101), dtype=bool)
        mask[47:54, 10:91] = True
        mask[10:54, 47:54] = True
        oracle = LayeredGTMapOracle(surface(mask, 0.0), self.config())
        prediction = oracle.predict((10.0, 10.0, 0.75), 0.0)
        self.assertEqual(prediction.exit_count, 3)
        self.assertEqual(int(np.argmax(prediction.count_probabilities)) + 1, 3)
        self.assertEqual(int(np.argmax(prediction.role_probabilities)), 1)

    def test_terminal_has_one_exit_and_is_deterministic(self):
        mask = np.zeros((101, 101), dtype=bool)
        mask[47:54, 50:91] = True
        oracle = LayeredGTMapOracle(surface(mask, 0.0), self.config())
        first = oracle.predict((10.2, 10.0, 0.75), 0.0)
        second = oracle.predict((10.2, 10.0, 0.75), 0.0)
        self.assertEqual(first.exit_count, 1)
        self.assertEqual(int(np.argmax(first.role_probabilities)), 2)
        np.testing.assert_array_equal(first.direction_logits, second.direction_logits)
        np.testing.assert_array_equal(first.z_role, second.z_role)

    def test_no_nearby_support_fails_closed(self):
        mask = np.ones((5, 5), dtype=bool)
        oracle = LayeredGTMapOracle(surface(mask, 0.0), self.config())
        with self.assertRaisesRegex(RuntimeError, "support"):
            oracle.local_layer((100.0, 100.0, 0.75), 0.0)

    def test_binary_ply_contract(self):
        points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.ply"
            with path.open("wb") as stream:
                stream.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                stream.write(points.tobytes())
            np.testing.assert_array_equal(load_binary_xyz_ply(path), points)


if __name__ == "__main__":
    unittest.main()
