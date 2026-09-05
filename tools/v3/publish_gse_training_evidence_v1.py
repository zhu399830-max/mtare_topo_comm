#!/usr/bin/env python3
"""Publish the sealed GSE training curve bundle with transactional cleanup."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal


EXPECTED_RUN_ID = "gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
EXPECTED_STATUS = "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
FIGURE_ID = "gse_training_curves"
SUFFIXES = (".png", ".pdf", ".svg", ".csv", "_summary.json", "_provenance.json", "_sha256.txt")


def _load_publisher():
    return importlib.import_module("publish_gse_training_figure_v1")


def _files_below(directory: Path) -> set[Path]:
    return {path.resolve() for path in directory.rglob("*") if path.is_file()}


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    if run_dir.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected GSE training evidence source")
    source = verify_complete_run_seal(root, run_dir, EXPECTED_STATUS)
    targets = [destination / f"{FIGURE_ID}{suffix}" for suffix in SUFFIXES]
    if any(path.exists() for path in targets):
        raise RuntimeError("training evidence target exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    initial_files = _files_below(destination)
    initial_directories = {path.resolve() for path in destination.rglob("*") if path.is_dir()}
    try:
        result = _load_publisher().publish(run_dir, destination)
        created = _files_below(destination) - initial_files
        if (
            result.get("figure_id") != FIGURE_ID
            or result.get("published_files") != len(targets)
            or created != {path.resolve() for path in targets}
        ):
            raise RuntimeError("training evidence publisher output set drift")
    except Exception:
        for path in sorted(_files_below(destination) - initial_files, reverse=True):
            path.unlink()
        current_directories = {path.resolve() for path in destination.rglob("*") if path.is_dir()}
        for directory in sorted(current_directories - initial_directories, reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise
    return {
        **result,
        "source_run_seal_sha256": source["seal_sha256"],
        "source_run_seal_entries": source["entries"],
        "atomic_bundle": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
