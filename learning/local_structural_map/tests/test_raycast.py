import numpy as np

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.raycast import ray_cells
from learning.local_structural_map.schema import Pose2D, StandardFrame


def test_ray_cells_progress_without_endpoint_duplicate():
    cells = ray_cells(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.2, 20.0, 0.1)
    assert len(cells) > 1
    assert np.unique(cells, axis=0).shape[0] == cells.shape[0]


def test_occupied_endpoint_clears_free_conflict():
    cfg = LocalMapConfig()
    builder = LocalStructuralMapBuilder(cfg)
    frame = StandardFrame(
        timestamp_ns=1,
        points_world=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        sensor_origin_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        pose=Pose2D(0.0, 0.0, 0.0, 0.0),
        source="synthetic",
    )
    item = builder.build_single(frame)
    occ = item.occupied > 0.5
    assert int(occ.sum()) == 1
    assert not bool((item.free[occ] > 0.5).any())
