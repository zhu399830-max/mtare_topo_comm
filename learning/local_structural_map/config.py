from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LocalMapConfig:
    size_m: float = 20.0
    resolution_m: float = 0.2
    min_range_m: float = 0.35
    max_range_m: float = 20.0
    ray_step_m: float = 0.1
    accumulation_window_sec: float = 4.0
    max_frames_in_window: int = 12
    min_z_rel_m: float = -2.5
    max_z_rel_m: float = 4.0
    robot_self_radius_m: float = 0.45
    voxel_size_m: float = 0.2
    occupancy_z_bands_m: Tuple[Tuple[float, float], ...] = ((-2.5, -0.2), (-0.2, 0.6), (0.6, 1.6), (1.6, 4.0))
    endpoint_height_range_norm_max_m: float = 3.0
    channels: List[str] = field(
        default_factory=lambda: [
            "observed_mask",
            "free_mask",
            "occupied_mask",
            "occupancy_z_band_0",
            "occupancy_z_band_1",
            "occupancy_z_band_2",
            "occupancy_z_band_3",
            "endpoint_height_range",
        ]
    )

    @property
    def grid_size(self) -> int:
        return int(round(self.size_m / self.resolution_m))

    @property
    def half_size_m(self) -> float:
        return self.size_m / 2.0

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["grid_size"] = self.grid_size
        data["coordinate_frame"] = {
            "origin": "current robot position",
            "local_x": "current robot yaw forward",
            "local_y": "current robot left",
            "local_z": "up",
            "positive_yaw": "counter-clockwise in world x-y",
        }
        data["unknown_encoding"] = {
            "observed_mask": 0.0,
            "free_mask": 0.0,
            "occupied_mask": 0.0,
            "z_band_occupancy": 0.0,
            "endpoint_height_range_when_invalid": 0.0,
        }
        data["normalization"] = {
            "mask_channels": "0 or 1",
            "endpoint_height_range": "max_z-min_z at occupied endpoint cell clipped to endpoint_height_range_norm_max_m then divided by it",
        }
        return data
