from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class StandardFrame:
    timestamp_ns: int
    points_world: np.ndarray
    sensor_origin_world: np.ndarray
    pose: Pose2D
    source: str
    frame_id: str = "map"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points_world = np.asarray(self.points_world, dtype=np.float32).reshape(-1, 3)
        self.sensor_origin_world = np.asarray(self.sensor_origin_world, dtype=np.float32).reshape(3)


@dataclass
class LocalStructuralMap:
    tensor: np.ndarray
    channel_names: List[str]
    timestamp_ns: int
    source: str
    pose: Pose2D
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tensor = np.asarray(self.tensor, dtype=np.float32)

    @property
    def observed(self) -> np.ndarray:
        return self.tensor[self.channel_names.index("observed_mask")]

    @property
    def free(self) -> np.ndarray:
        return self.tensor[self.channel_names.index("free_mask")]

    @property
    def occupied(self) -> np.ndarray:
        return self.tensor[self.channel_names.index("occupied_mask")]
