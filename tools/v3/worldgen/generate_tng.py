#!/usr/bin/env python3
"""Generate one topology-parent JSON artifact from an audited configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.data.worldgen import SeedBundle, TNGParameters, generate_tng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one deterministic 3-D tunnel network graph (no mesh)."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "tng_core_config_v1":
        raise ValueError("unsupported or missing schema_version")
    seed_bundle = SeedBundle.from_master(config["master_seed"])
    parameters = TNGParameters(**config["parameters"])
    graph = generate_tng(seed_bundle, parameters)
    payload = graph.to_dict(seed_bundle)
    payload["provenance"] = {
        "generator": "mtare_topo.data.worldgen.tng.generate_tng",
        "config": str(args.config.resolve()),
        "scope": "topology_parent_only_no_mesh_no_dataset_no_training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "topology_parent_id": graph.topology_parent_id,
                "canonical_hash": graph.canonical_hash(),
                "stats": graph.stats(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
