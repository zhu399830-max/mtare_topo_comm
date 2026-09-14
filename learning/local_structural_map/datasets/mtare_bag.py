from pathlib import Path
from typing import Iterator, List, Tuple

from rosbags.highlevel import AnyReader

from .common import nearest_by_time, pointcloud2_xyz, pose_from_odometry, stamp_to_ns
from ..schema import StandardFrame


class MTAREBagAdapter:
    def __init__(
        self,
        bag: str | Path,
        scan_topic: str = "/registered_scan",
        odom_topic: str = "/state_estimation_at_scan",
        max_sync_dt_sec: float = 0.25,
        max_samples: int | None = None,
        stride: int = 1,
        sync_policy: str = "exact_stamp",
    ) -> None:
        self.bag = Path(bag)
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.max_sync_dt_ns = int(max_sync_dt_sec * 1e9)
        self.max_samples = max_samples
        self.stride = max(1, int(stride))
        if sync_policy not in {"exact_stamp", "causal_latest", "nearest"}:
            raise ValueError("sync_policy must be exact_stamp, causal_latest or nearest")
        self.sync_policy = sync_policy

    def __iter__(self) -> Iterator[StandardFrame]:
        if self.sync_policy in {"nearest", "exact_stamp"}:
            yield from self._iter_nearest()
            return
        yield from self._iter_causal_latest()

    def _iter_causal_latest(self) -> Iterator[StandardFrame]:
        latest_odom = None
        latest_odom_stamp = None
        emitted = 0
        seen = 0
        with AnyReader([self.bag]) as reader:
            conns = [c for c in reader.connections if c.topic in {self.scan_topic, self.odom_topic}]
            for conn, timestamp_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == self.odom_topic:
                    latest_odom_stamp = stamp_to_ns(msg.header.stamp) or int(timestamp_ns)
                    latest_odom = msg
                    continue
                if latest_odom is None or latest_odom_stamp is None:
                    continue
                scan_stamp = stamp_to_ns(msg.header.stamp) or int(timestamp_ns)
                if abs(scan_stamp - latest_odom_stamp) > self.max_sync_dt_ns:
                    continue
                seen += 1
                if (seen - 1) % self.stride:
                    continue
                pose = pose_from_odometry(latest_odom)
                points_world = pointcloud2_xyz(msg)
                yield StandardFrame(
                    timestamp_ns=scan_stamp,
                    points_world=points_world,
                    sensor_origin_world=[pose.x, pose.y, pose.z],
                    pose=pose,
                    source="M-TARE",
                    frame_id=msg.header.frame_id or "map",
                    metadata={
                        "bag": str(self.bag),
                        "scan_topic": self.scan_topic,
                        "odom_topic": self.odom_topic,
                        "sync_policy": self.sync_policy,
                        "scan_stamp_ns": int(scan_stamp),
                        "odom_stamp_ns": int(latest_odom_stamp),
                        "sync_dt_ns": int(abs(latest_odom_stamp - scan_stamp)),
                        "raw_points": int(len(points_world)),
                    },
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    break

    def _iter_nearest(self) -> Iterator[StandardFrame]:
        odom = self._load_odom()
        emitted = 0
        seen = 0
        with AnyReader([self.bag]) as reader:
            conns = [c for c in reader.connections if c.topic == self.scan_topic]
            for conn, timestamp_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                scan_stamp = stamp_to_ns(msg.header.stamp) or int(timestamp_ns)
                if self.sync_policy == "exact_stamp":
                    exact = [item for item in odom if item[0] == scan_stamp]
                    match = exact[0] if exact else nearest_by_time(odom, scan_stamp, self.max_sync_dt_ns)
                else:
                    match = nearest_by_time(odom, scan_stamp, self.max_sync_dt_ns)
                if match is None:
                    continue
                odom_stamp, odom_msg = match
                seen += 1
                if (seen - 1) % self.stride:
                    continue
                pose = pose_from_odometry(odom_msg)
                points_world = pointcloud2_xyz(msg)
                yield StandardFrame(
                    timestamp_ns=scan_stamp,
                    points_world=points_world,
                    sensor_origin_world=[pose.x, pose.y, pose.z],
                    pose=pose,
                    source="M-TARE",
                    frame_id=msg.header.frame_id or "map",
                    metadata={
                        "bag": str(self.bag),
                        "scan_topic": self.scan_topic,
                        "odom_topic": self.odom_topic,
                        "sync_policy": self.sync_policy,
                        "scan_stamp_ns": int(scan_stamp),
                        "odom_stamp_ns": int(odom_stamp),
                        "sync_dt_ns": int(abs(odom_stamp - scan_stamp)),
                        "raw_points": int(len(points_world)),
                    },
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    break

    def _load_odom(self) -> List[Tuple[int, object]]:
        out: List[Tuple[int, object]] = []
        with AnyReader([self.bag]) as reader:
            conns = [c for c in reader.connections if c.topic == self.odom_topic]
            for conn, timestamp_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                stamp = stamp_to_ns(msg.header.stamp) or int(timestamp_ns)
                out.append((stamp, msg))
        return out
