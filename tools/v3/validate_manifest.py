#!/usr/bin/env python3
"""Validate a proposed V3 data card and strict-test isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-card", required=True, type=Path)
    args = parser.parse_args()
    path = args.data_card if args.data_card.is_absolute() else PROJECT_ROOT / args.data_card
    report = validate_data_card(load_json(path))
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
