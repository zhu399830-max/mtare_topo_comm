#!/usr/bin/env python3
"""Print a deterministic read-only inventory for GSE causal sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtare_topo.data.gse_sequence_inventory import inventory_from_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    inventory = inventory_from_registry(registry_path=args.registry, mesh_root=args.mesh_root)
    if args.summary_only:
        inventory = {
            key: inventory[key]
            for key in (
                "schema_version",
                "spacing_m",
                "history_frames",
                "history_span_m",
                "strict_test_worlds_read",
                "mtare_worlds_read",
                "split_summaries",
                "totals",
            )
        }
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
