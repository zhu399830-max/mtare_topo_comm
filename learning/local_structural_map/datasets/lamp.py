from pathlib import Path
from typing import Dict, Iterator, List

from rosbags.highlevel import AnyReader

from .common import pointcloud2_xyz, pose_from_msg_pose, stamp_to_ns
from ..geometry import transform_points_local_to_world
from ..schema import StandardFrame


class LAMPDatasetAdapter:
    def __init__(
        self,
        scan_bag: str | Path,
        robot: str = "husky3",
        max_samples: int | None = None,
        stride: int = 1,
    ) -> None:
        self.scan_bag = Path(scan_bag)
        self.robot = robot
        self.scan_topic = f"/{robot}/lamp/keyed_scans"
        self.pose_graph_topic = f"/{robot}/lamp/pose_graph_incremental"
        self.max_samples = max_samples
        self.stride = max(1, int(stride))

    def __iter__(self) -> Iterator[StandardFrame]:
        poses = self._load_poses_by_key()
        emitted = 0
        seen = 0
        with AnyReader([self.scan_bag]) as reader:
            conns = [c for c in reader.connections if c.topic == self.scan_topic]
            for conn, timestamp_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                if msg.key not in poses:
                    continue
                seen += 1
                if (seen - 1) % self.stride:
                    continue
                pose, pose_stamp_ns = poses[msg.key]
                points_local = pointcloud2_xyz(msg.scan)
                points_world = transform_points_local_to_world(points_local, pose)
                origin = [pose.x, pose.y, pose.z]
                yield StandardFrame(
                    timestamp_ns=pose_stamp_ns or int(timestamp_ns),
                    points_world=points_world,
                    sensor_origin_world=origin,
                    pose=pose,
                    source="LAMP",
                    frame_id="map",
                    metadata={
                        "bag": str(self.scan_bag),
                        "scan_topic": self.scan_topic,
                        "pose_topic": self.pose_graph_topic,
                        "key": int(msg.key),
                        "raw_points": int(len(points_local)),
                    },
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    break

    def _load_poses_by_key(self) -> Dict[int, object]:
        poses: Dict[int, object] = {}
        with AnyReader([self.scan_bag]) as reader:
            conns = [c for c in reader.connections if c.topic == self.pose_graph_topic]
            for conn, timestamp_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                for node in msg.nodes:
                    if node.key not in poses:
                        stamp_ns = stamp_to_ns(node.header.stamp) or int(timestamp_ns)
                        poses[int(node.key)] = (pose_from_msg_pose(node.pose), stamp_ns)
        return poses
