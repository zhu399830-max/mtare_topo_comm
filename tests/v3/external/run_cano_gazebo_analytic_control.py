#!/usr/bin/env python3
"""External frozen-image control for the Gazebo organized CPU ray contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mtare_topo.data.cano_gazebo_parity import analytic_box_ranges, gazebo_world_sdf, parity_metrics


IMAGE = "mtare-semantic-runtime:local"
IMAGE_ID = "sha256:9819eea672b2001ed18f12ee09c67f0da53a5ba6ce69192e2f21c128e5e78f9a"
PLUGIN_SHA = "325000c31a6af77114f0d7339781c83eb0c41644adce99f219a0b9ba536ca0d3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    observed_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if observed_id != IMAGE_ID:
        raise RuntimeError(f"image mismatch {observed_id}")
    observed_plugin = subprocess.check_output(
        ["docker", "run", "--rm", "--network", "none", "-v", "/etc/localtime:/etc/localtime:ro", IMAGE, "sha256sum", "/home/docker-user/mtare/autonomous_exploration_development_environment/devel/lib/libgazebo_ros_velodyne_laser.so"],
        text=True,
    ).split()[0]
    if observed_plugin != PLUGIN_SHA:
        raise RuntimeError("plugin hash mismatch")
    temp = Path(tempfile.mkdtemp(prefix="mtare_gazebo_analytic_"))
    try:
        parity = temp / "parity"
        output = temp / "output"
        parity.mkdir()
        output.mkdir()
        pose = {"sensor_xyz_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0}
        (parity / "analytic.world").write_text(gazebo_world_sdf(None, [pose], analytic_box=True), encoding="utf-8")
        subprocess.run(
            [
                "docker", "run", "--rm", "--name", "mtare_gazebo_analytic_contract_v1",
                "--network", "none", "--shm-size", "512m",
                "-v", "/etc/localtime:/etc/localtime:ro",
                "-v", f"{ROOT}:/workspace:ro", "-v", f"{parity}:/parity:ro", "-v", f"{output}:/output:rw",
                IMAGE, "bash", "/workspace/tools/v3/gazebo/run_fixed_lidar_session.sh",
                "/parity/analytic.world", "/output/scans.npz", "1", "/output/logs",
            ],
            check=True,
            timeout=180,
        )
        with np.load(output / "scans.npz", allow_pickle=False) as data:
            ranges = data["range_00"]
            valid = data["valid_00"]
        if ranges.shape != (3, 16, 720) or valid.shape != (3, 16, 720):
            raise RuntimeError(f"unexpected scans {ranges.shape} {valid.shape}")
        expected_range, expected_valid = analytic_box_ranges()
        repeat_max = float(np.max(np.abs(ranges[0][valid[0].astype(bool)] - ranges[1][valid[0].astype(bool)])))
        metrics = parity_metrics(expected_range, expected_valid, ranges[0], valid[0])
        result = {
            "passed": bool(np.array_equal(valid[0], valid[1]) and np.array_equal(valid[0], valid[2]) and repeat_max <= 0.0011 and metrics["passed"]),
            "image_id": observed_id,
            "plugin_sha256": observed_plugin,
            "scan_shape": list(ranges.shape),
            "repeat_max_difference_m": repeat_max,
            "analytic_metrics": metrics,
            "world_sha256": sha256(parity / "analytic.world"),
        }
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    finally:
        shutil.rmtree(temp)


if __name__ == "__main__":
    raise SystemExit(main())
