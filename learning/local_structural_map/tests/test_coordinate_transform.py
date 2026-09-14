import math

import numpy as np

from learning.local_structural_map.geometry import transform_points_world_to_local
from learning.local_structural_map.schema import Pose2D


def test_local_x_is_robot_yaw_forward_and_y_left():
    pose = Pose2D(1.0, 2.0, 0.0, math.pi / 2.0)
    world = np.asarray([[1.0, 3.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    local = transform_points_world_to_local(world, pose)
    np.testing.assert_allclose(local[0, :2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(local[1, :2], [0.0, 1.0], atol=1e-6)
