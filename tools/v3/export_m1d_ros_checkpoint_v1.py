#!/usr/bin/env python3
"""One-shot lossless export of a frozen M1D checkpoint for ROS Python 3.8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtare_topo.deployment.m1d_checkpoint import export_m1d_ros_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--expected-mode",
        choices=("M1D", "M1D_AEE_HEAD_ADAPTED_V1", "M1D_AEE_CORRECTIVE_COMPOSITE_V9"),
        default="M1D",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.audit}")
    result = export_m1d_ros_checkpoint(
        source=args.source.resolve(),
        destination=args.output.resolve(),
        expected_source_sha256=args.source_sha256,
        expected_seed=args.seed,
        expected_mode=args.expected_mode,
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
