#!/usr/bin/env python3
"""Run and seal the approved one-shot C09 route-conditioned geometry qualification."""
from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_cano_c08_hybrid_implicit_qualification_v3 as core


RUN_ID = "gate4_20260817_cano_c09_route_conditioned_geometry_qualification_v1_seed0"
EXECUTOR_NAME = "execute_cano_c09_route_conditioned_geometry_qualification_v1.py"
PASS_STATUS = "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1"
FAIL_STATUS = "FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1"
SCHEMA_VERSION = "cano_c09_route_conditioned_geometry_qualification_runner_v1"


def configure_core() -> None:
    """Apply the frozen C09 resource and evidence profile to the shared sealed runner."""
    core.RUN_ID = RUN_ID
    core.EXECUTOR = PROJECT_ROOT / f"tools/v3/{EXECUTOR_NAME}"
    core.RAM_LIMIT_BYTES = 4 * 1024**3
    core.DISK_LIMIT_BYTES = 8 * 1024**3
    core.TIME_LIMIT_SECONDS = 24 * 60 * 60
    core.PASS_STATUS = PASS_STATUS
    core.FAIL_STATUS = FAIL_STATUS
    core.RUNNER_SCHEMA_VERSION = SCHEMA_VERSION
    core.EXPECTED_SUMMARY = {
        "worlds": 10, "windows": 71, "arc_endpoint_records": 2054,
        "directed_traversals": 2054, "frames": 15833,
        "resolution_frame_audits": 47499, "horizontal_rays": 34199280,
        "collision_vertical_rays": 47499, "support_height_evaluations": 47499,
        "window_frames": 1331, "outside_window_frames": 14502,
        "visual_patch_count": 639, "c09_worlds_read": 10, "preview_count": 10,
    }
    core.CLAIM_BOUNDARY = "C09 route-conditioned geometry qualification only; no LiDAR, inference, graph replay, parameter selection, C10 or M-TARE."


def main() -> int:
    configure_core()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
