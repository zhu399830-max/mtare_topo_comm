import numpy as np

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.schema import Pose2D, StandardFrame


def test_contract_shape_dtype_channels_are_stable():
    cfg = LocalMapConfig()
    builder = LocalStructuralMapBuilder(cfg)
    points = np.asarray([[2.0, 0.0, 0.0], [3.0, 1.0, 1.0], [np.nan, 0.0, 0.0]], dtype=np.float32)
    frame = StandardFrame(1, points, np.zeros(3), Pose2D(0.0, 0.0, 0.0, 0.0), "synthetic")
    item = builder.update(frame)
    assert item.tensor.shape == (8, 100, 100)
    assert item.tensor.dtype == np.float32
    assert item.channel_names == cfg.channels
    assert np.isfinite(item.tensor).all()
    assert item.metadata["raw_points"] == 3
    assert item.metadata["valid_points_after_filter"] == 2
