#!/usr/bin/env python3
"""Run the unchanged single-robot case contract with the V4 topology node."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_mtare_single_robot_case_v1 as base


BASE_METHOD_COMMAND = base.method_command


def method_command(args: argparse.Namespace, planner_output: Path) -> str:
    if args.method_family != "m1d_topology":
        return BASE_METHOD_COMMAND(args, planner_output)
    if not args.checkpoint or not args.checkpoint_sha256:
        raise ValueError("M1D case requires checkpoint identity")
    return (
        "python3 /workspace/tools/v3/semantic_topology_global_node_v4.py "
        f"--checkpoint /workspace/{args.checkpoint} --checkpoint-sha256 {args.checkpoint_sha256} "
        f"--output {planner_output} --publish-period-sec 1.0"
    )


def main() -> int:
    original = base.method_command
    base.method_command = method_command
    try:
        return base.main()
    finally:
        base.method_command = original


if __name__ == "__main__":
    raise SystemExit(main())
