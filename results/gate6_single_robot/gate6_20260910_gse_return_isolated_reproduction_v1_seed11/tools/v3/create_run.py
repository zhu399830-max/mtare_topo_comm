#!/usr/bin/env python3
"""Create a standard V3 result directory; never execute the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import create_run, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="V3 run spec JSON")
    parser.add_argument(
        "--status",
        type=Path,
        default=PROJECT_ROOT / "results" / "project_status.json",
        help="machine-readable project status",
    )
    args = parser.parse_args()

    spec_path = args.spec if args.spec.is_absolute() else PROJECT_ROOT / args.spec
    status_path = args.status if args.status.is_absolute() else PROJECT_ROOT / args.status
    spec = load_json(spec_path)
    status = load_json(status_path)
    output_dir = create_run(spec, status, PROJECT_ROOT)
    print(output_dir)
    print("CREATED_NOT_EXECUTED: inspect config/command.txt before running anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
