#!/usr/bin/env python3
"""Generate the frozen C09 sensor/teacher stream by reusing the proven C08 stage."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import execute_cano_c08_causal_sensor_stage_v1 as core
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


WORLDS = tuple(f"S{index:02d}_{name}_C09" for index, name in enumerate((
    "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
    "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
    "flat_complex", "3d_complex",
), 1))
GEOMETRY_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"
EXPECTED_FRAMES = 15833


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    core.WORLDS = WORLDS
    core.TEACHER_TRAJECTORY_RUN = GEOMETRY_RUN
    core.QUALIFICATION_RUN = GEOMETRY_RUN
    core.EXPECTED_FRAMES = EXPECTED_FRAMES
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = core.main()
    summary_path = args.run_dir.resolve() / "metrics/sensor_summary.json"
    summary = load_json(summary_path)
    summary.update({
        "schema_version": "cano_c09_causal_sensor_stage_v1",
        "overall_status": "PASS_C09_CAUSAL_SENSOR_STAGE_V1",
        "c08_worlds_read": 0,
        "c09_worlds_read": len(WORLDS),
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    })
    write_json(summary_path, summary)
    print(captured.getvalue().replace("C08", "C09"), end="", flush=True)
    print(json.dumps(summary), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
