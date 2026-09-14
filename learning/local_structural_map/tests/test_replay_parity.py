import copy

import numpy as np

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.runtime.mtare_runtime import MTARERuntimeAdapter
from learning.local_structural_map.schema import Pose2D, StandardFrame


class Header:
    def __init__(self):
        self.stamp = type("Stamp", (), {"sec": 0, "nanosec": 1})()
        self.frame_id = "map"


class Scan:
    pass


def test_same_standard_frame_builder_is_deterministic():
    frame = StandardFrame(
        1,
        np.asarray([[2.0, 0.0, 0.0], [2.0, 1.0, 0.5]], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        Pose2D(0.0, 0.0, 0.0, 0.0),
        "M-TARE",
    )
    a = LocalStructuralMapBuilder().update(frame)
    b = LocalStructuralMapBuilder().update(copy.deepcopy(frame))
    np.testing.assert_array_equal(a.tensor, b.tensor)
