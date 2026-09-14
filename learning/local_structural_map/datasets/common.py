from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from ..geometry import transform_points_local_to_world, yaw_from_quaternion
from ..schema import Pose2D, StandardFrame


POINT_FIELD_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def stamp_to_ns(stamp) -> int:
    sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
    nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsecs", getattr(stamp, "nsec", 0))))
    return sec * 1_000_000_000 + nsec


def pointcloud2_xyz(msg) -> np.ndarray:
    offsets = {field.name: (int(field.offset), int(field.datatype)) for field in msg.fields}
    if not {"x", "y", "z"}.issubset(offsets):
        raise ValueError(f"PointCloud2 lacks x/y/z fields: {sorted(offsets)}")
    count = int(msg.width) * int(msg.height)
    point_step = int(msg.point_step)
    if isinstance(msg.data, (bytes, bytearray)):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
    else:
        raw = np.asarray(msg.data, dtype=np.uint8)
    points = np.empty((count, 3), dtype=np.float32)
    for i, name in enumerate(["x", "y", "z"]):
        offset, datatype = offsets[name]
        dtype = POINT_FIELD_DTYPES[datatype]
        values = np.ndarray(shape=(count,), dtype=dtype, buffer=raw, offset=offset, strides=(point_step,))
        points[:, i] = values.astype(np.float32)
    return points


def pose_from_msg_pose(pose_msg) -> Pose2D:
    p = pose_msg.position
    q = pose_msg.orientation
    return Pose2D(float(p.x), float(p.y), float(p.z), yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w)))


def pose_from_odometry(msg) -> Pose2D:
    return pose_from_msg_pose(msg.pose.pose)


def nearest_by_time(items: Sequence[Tuple[int, object]], target_ns: int, max_dt_ns: int) -> Optional[Tuple[int, object]]:
    if not items:
        return None
    times = np.asarray([t for t, _ in items], dtype=np.int64)
    idx = int(np.argmin(np.abs(times - int(target_ns))))
    dt = abs(int(times[idx]) - int(target_ns))
    if dt > max_dt_ns:
        return None
    return items[idx]


def bag_topic_summary(bag_paths: Iterable[Path]) -> Dict[str, object]:
    from rosbags.highlevel import AnyReader

    out: Dict[str, object] = {"bags": []}
    for path in bag_paths:
        bag_info: Dict[str, object] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            out["bags"].append(bag_info)
            continue
        try:
            with AnyReader([path]) as reader:
                bag_info["start_time_ns"] = int(reader.start_time)
                bag_info["end_time_ns"] = int(reader.end_time)
                bag_info["duration_sec"] = (int(reader.end_time) - int(reader.start_time)) / 1e9
                bag_info["topics"] = [
                    {"topic": c.topic, "msgtype": c.msgtype, "msgcount": int(c.msgcount)}
                    for c in sorted(reader.connections, key=lambda c: c.topic)
                ]
        except Exception as exc:
            bag_info["error"] = f"{type(exc).__name__}: {exc}"
        out["bags"].append(bag_info)
    return out


class StandardFrameStream:
    def __iter__(self) -> Iterator[StandardFrame]:
        raise NotImplementedError
