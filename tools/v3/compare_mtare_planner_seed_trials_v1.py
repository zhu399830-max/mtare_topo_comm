#!/usr/bin/env python3
"""Compare two same-seed trials and one different-seed M-TARE trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "mtare_planner_seed_trial_summary_v1":
        raise ValueError(f"unexpected trial schema: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed11-a", required=True, type=Path)
    parser.add_argument("--seed11-b", required=True, type=Path)
    parser.add_argument("--seed23", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    a, b, different = (load(path) for path in (args.seed11_a, args.seed11_b, args.seed23))
    same_seed_identity = (
        a["seed"] == b["seed"] == 11
        and a["audit_frames"] == b["audit_frames"]
        and a["audit_waypoints_xyz_m"] == b["audit_waypoints_xyz_m"]
    )
    different_seed_valid = (
        different["seed"] == 23
        and different["audit_frame_sequence_sha256"] != a["audit_frame_sequence_sha256"]
    )
    result = {
        "schema_version": "mtare_planner_seed_qualification_comparison_v1",
        "same_seed_full_frame_and_waypoint_identity": same_seed_identity,
        "different_seed_changes_observed_sensor_trajectory_sequence": different_seed_valid,
        "seed11_frame_sha256": a["audit_frame_sequence_sha256"],
        "seed11_waypoint_sha256": a["audit_waypoint_sequence_sha256"],
        "seed23_frame_sha256": different["audit_frame_sequence_sha256"],
        "passed": same_seed_identity and different_seed_valid,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
