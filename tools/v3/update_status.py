#!/usr/bin/env python3
"""Update operational status fields without changing the active Gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, update_status_fields, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        type=Path,
        default=PROJECT_ROOT / "results" / "project_status.json",
    )
    running = parser.add_mutually_exclusive_group()
    running.add_argument("--set-running-experiment")
    running.add_argument("--clear-running-experiment", action="store_true")
    parser.add_argument("--latest-result")
    parser.add_argument("--blocking-issue")
    parser.add_argument("--next-action")
    parser.add_argument("--updated-at")
    args = parser.parse_args()

    status_path = args.status if args.status.is_absolute() else PROJECT_ROOT / args.status
    status = load_json(status_path)
    running_value: str | None | object = ...
    if args.set_running_experiment:
        running_value = args.set_running_experiment
    elif args.clear_running_experiment:
        running_value = None
    updated = update_status_fields(
        status,
        running_experiment=running_value,
        latest_result=args.latest_result,
        blocking_issue=args.blocking_issue,
        next_action=args.next_action,
        updated_at=args.updated_at,
    )
    write_json(status_path, updated)
    print(status_path)
    print("Gate fields were preserved; this tool cannot authorize or perform Gate transitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
