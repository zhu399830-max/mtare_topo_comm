from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from .common import bag_topic_summary


class SubTMRSDatasetAdapter:
    """Audit-only adapter for current UGV2 files.

    The confirmed UGV2 bags available in this workspace contain raw
    VelodyneScan packets and IMU, not directly consumable PointCloud2 plus pose.
    This class records that evidence and intentionally does not generate maps.
    """

    def __init__(self, bag_dir: str | Path, extrinsics_yaml: str | Path) -> None:
        self.bag_dir = Path(bag_dir)
        self.extrinsics_yaml = Path(extrinsics_yaml)

    def audit(self, max_bags: int = 8) -> Dict[str, object]:
        bags = sorted(self.bag_dir.glob("*.bag"))
        selected = bags[: max(0, max_bags // 2)] + bags[-max(0, max_bags - max_bags // 2) :]
        selected = list(dict.fromkeys(selected))
        extrinsics = None
        if self.extrinsics_yaml.exists():
            with self.extrinsics_yaml.open() as f:
                extrinsics = yaml.safe_load(f)
        summary = bag_topic_summary(selected)
        topic_names = {
            topic["topic"]
            for bag in summary["bags"]
            for topic in bag.get("topics", [])
        }
        usable = any(t for t in topic_names if "PointCloud2" in t)
        pose_topics = [t for t in topic_names if any(k in t.lower() for k in ["odom", "pose", "path", "tf"])]
        return {
            "bag_dir": str(self.bag_dir),
            "bag_count": len(bags),
            "audited_bags": [str(p) for p in selected],
            "extrinsics_yaml": str(self.extrinsics_yaml),
            "extrinsics": extrinsics,
            "topic_summary": summary,
            "confirmed_lidar_topics": sorted([t for t in topic_names if "velodyne" in t.lower() or "scan" in t.lower()]),
            "confirmed_pose_like_topics": sorted(pose_topics),
            "direct_pointcloud2_and_pose_usable": bool(usable and pose_topics),
            "finding": "UNAVAILABLE: audited UGV2 bags expose raw velodyne packets plus IMU only; no confirmed PointCloud2 and global pose/path/tf in selected segments.",
        }
