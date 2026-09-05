from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.gse_training_targets import (
    circular_roll_training_example,
    heading_deg_to_unit,
    world_vector_to_robot,
)


class GSETrainingTargetsTest(unittest.TestCase):
    def test_world_axis_is_rotated_into_robot_frame_without_pose_input(self) -> None:
        axis = world_vector_to_robot(np.asarray([0.0, 2.0, 2.0]), robot_yaw_deg=90.0)
        np.testing.assert_allclose(axis, [np.sqrt(0.5), 0.0, np.sqrt(0.5)], atol=1e-12)

    def test_heading_unit_matches_model_sine_cosine_contract(self) -> None:
        np.testing.assert_allclose(heading_deg_to_unit(90.0), [1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(heading_deg_to_unit(180.0), [0.0, -1.0], atol=1e-7)

    def test_positive_circular_roll_rotates_axis_and_exit_headings_together(self) -> None:
        student = np.zeros((5, 2, 16, 720), dtype=np.float32)
        student[..., 0] = 1.0
        rolled, targets = circular_roll_training_example(
            student,
            {
                "local_axis": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "exit_heading_unit": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            },
            shift_columns=180,
        )
        self.assertTrue(np.all(rolled[..., 180] == 1.0))
        np.testing.assert_allclose(targets["local_axis"], [0.0, 1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(targets["exit_heading_unit"][0], [1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(targets["exit_heading_unit"][1], [0.0, -1.0], atol=1e-7)

    def test_zero_roll_is_validation_identity(self) -> None:
        student = np.arange(5 * 2 * 16 * 720, dtype=np.float32).reshape(5, 2, 16, 720)
        targets = {
            "local_axis": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "exit_heading_unit": np.asarray([[0.0, 1.0]], dtype=np.float32),
        }
        rolled, changed = circular_roll_training_example(student, targets, shift_columns=0)
        np.testing.assert_array_equal(rolled, student)
        np.testing.assert_array_equal(changed["local_axis"], targets["local_axis"])
        np.testing.assert_array_equal(changed["exit_heading_unit"], targets["exit_heading_unit"])


if __name__ == "__main__":
    unittest.main()
