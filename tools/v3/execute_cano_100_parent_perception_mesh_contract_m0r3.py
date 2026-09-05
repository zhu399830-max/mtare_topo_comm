#!/usr/bin/env python3
"""Execute M0R3 with vertex-to-opposite-triangle-surface replay distance."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
import execute_cano_100_parent_perception_mesh_contract_m0r as m0r
from mtare_topo.governance import write_json


TRIANGLE_COUNT_RELATIVE_LIMIT = 0.0001
POINT_TO_SURFACE_MAXIMUM_M = 0.75


def _geometric_replay(primary, replay):
    return m0r._geometric_replay_with_triangle_limit(
        primary,
        replay,
        triangle_count_relative_limit=TRIANGLE_COUNT_RELATIVE_LIMIT,
        use_point_to_surface=True,
    )


def execute(run_dir: Path):
    m0._materialize = m0r._materialize_with_paths
    m0.replay_pair_audit = _geometric_replay
    summary = m0.execute(run_dir)
    summary["schema_version"] = (
        "cano_100_parent_perception_mesh_contract_summary_m0r3"
    )
    summary["overall_status"] = (
        "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R3"
    )
    summary["replay_contract"] = {
        "exact": [
            "graph identity",
            "spline identity",
            "operation trace",
            "effective geometry parameters",
            "axis array",
            "vertex count",
        ],
        "bounded": {
            "triangle_count_relative_difference_maximum": (
                TRIANGLE_COUNT_RELATIVE_LIMIT
            ),
            "bidirectional_vertex_to_opposite_triangle_surface_maximum_m": (
                POINT_TO_SURFACE_MAXIMUM_M
            ),
            "aabb_endpoint_maximum_coordinate_difference_m": 0.5,
            "surface_area_relative_difference_maximum": 0.01,
        },
        "nearest_vertex_statistics": "recorded_not_required",
        "obj_sha256_equality_required": False,
    }
    summary["claim_boundary"] = (
        "Train-only native perception-surface materialization replay; not formal "
        "data, LiDAR, navigation geometry, training or M-TARE change."
    )
    write_json(run_dir / "metrics/summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
