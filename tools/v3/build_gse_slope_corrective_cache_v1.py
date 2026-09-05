#!/usr/bin/env python3
"""Build the immutable train-only feature cache for GSE slope correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_slope_corrective_dataset import build_slope_corrective_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_slope_corrective_cache(args.dataset_run, args.output_dir)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
