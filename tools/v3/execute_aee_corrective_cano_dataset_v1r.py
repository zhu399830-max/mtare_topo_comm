#!/usr/bin/env python3
"""Directory-initialization-only replacement entry for corrective Cano V1."""

from __future__ import annotations

import argparse
from pathlib import Path

def initialize_v1r_output_directories(run_dir: Path) -> Path:
    """Create the per-world metric parent missing from the sealed V1 run."""
    metrics_root = run_dir.resolve() / "metrics/cano"
    metrics_root.mkdir(parents=True, exist_ok=True)
    return metrics_root


def main() -> int:
    import execute_aee_corrective_cano_dataset_v1 as implementation

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", required=True, type=Path)
    args, _ = parser.parse_known_args()
    initialize_v1r_output_directories(args.run_dir)
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
