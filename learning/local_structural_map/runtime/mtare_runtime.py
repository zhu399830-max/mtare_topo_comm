from typing import Optional

import numpy as np

from ..builder import LocalStructuralMapBuilder
from ..datasets.common import pointcloud2_xyz, pose_from_odometry, stamp_to_ns
from ..schema import LocalStructuralMap, StandardFrame


class MTARERuntimeAdapter:
    def __init__(self, builder: LocalStructuralMapBuilder, max_sync_dt_sec: float = 0.25) -> None:
        self.builder = builder
        self.max_sync_dt_ns = int(max_sync_dt_sec * 1e9)
        self._latest_odom = None
        self._latest_odom_stamp_ns: Optional[int] = None
        self._odom_by_stamp = {}
        self._pending_scans = {}

    def handle_odom(self, odom_msg, receive_time_ns: Optional[int] = None) -> Optional[LocalStructuralMap]:
        stamp = stamp_to_ns(odom_msg.header.stamp) or int(receive_time_ns or 0)
        self._latest_odom = odom_msg
        self._latest_odom_stamp_ns = stamp
        self._odom_by_stamp[stamp] = odom_msg
        self._trim_cache(stamp)
        pending = self._pending_scans.pop(stamp, None)
        if pending is not None:
            return self._build_from_pair(pending, odom_msg, stamp)
        return None

    def handle_scan(self, scan_msg, receive_time_ns: Optional[int] = None) -> Optional[LocalStructuralMap]:
        scan_stamp = stamp_to_ns(scan_msg.header.stamp) or int(receive_time_ns or 0)
        odom_msg = self._odom_by_stamp.get(scan_stamp)
        odom_stamp = scan_stamp if odom_msg is not None else None
        if odom_msg is None and self._odom_by_stamp:
            stamps = np.asarray(list(self._odom_by_stamp.keys()), dtype=np.int64)
            idx = int(np.argmin(np.abs(stamps - scan_stamp)))
            if abs(int(stamps[idx]) - scan_stamp) <= self.max_sync_dt_ns:
                odom_stamp = int(stamps[idx])
                odom_msg = self._odom_by_stamp[odom_stamp]
        if odom_msg is None:
            self._pending_scans[scan_stamp] = scan_msg
            return None
        return self._build_from_pair(scan_msg, odom_msg, int(odom_stamp))

    def _build_from_pair(self, scan_msg, odom_msg, odom_stamp_ns: int) -> LocalStructuralMap:
        scan_stamp = stamp_to_ns(scan_msg.header.stamp)
        if abs(scan_stamp - odom_stamp_ns) > self.max_sync_dt_ns:
            return None
        pose = pose_from_odometry(odom_msg)
        points = pointcloud2_xyz(scan_msg)
        frame = StandardFrame(
            timestamp_ns=scan_stamp,
            points_world=points,
            sensor_origin_world=[pose.x, pose.y, pose.z],
            pose=pose,
            source="M-TARE",
            frame_id=scan_msg.header.frame_id or "map",
            metadata={
                "runtime_adapter": self.__class__.__name__,
                "scan_stamp_ns": int(scan_stamp),
                "odom_stamp_ns": int(odom_stamp_ns),
                "sync_dt_ns": int(abs(scan_stamp - odom_stamp_ns)),
                "raw_points": int(len(points)),
            },
        )
        return self.builder.update(frame)

    def _trim_cache(self, now_stamp_ns: int) -> None:
        min_stamp = now_stamp_ns - max(self.max_sync_dt_ns * 4, int(2e9))
        for key in list(self._odom_by_stamp.keys()):
            if key < min_stamp:
                self._odom_by_stamp.pop(key, None)
        for key in list(self._pending_scans.keys()):
            if key < min_stamp:
                self._pending_scans.pop(key, None)
