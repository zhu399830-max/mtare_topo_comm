#!/usr/bin/env python3
"""Atomically publish both sealed GSE offline-topology paper figure bundles."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate4_20260824_gse_offline_topology_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V1"
INDEX_ID = "gse_offline_topology_evidence_index"
FIGURE_SUFFIXES = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
BUNDLE_SPECS = (
    ("publish_gse_offline_topology_figure_v1", "gse_offline_topology", FIGURE_SUFFIXES),
    ("publish_gse_topology_examples_v1", "gse_topology_examples", FIGURE_SUFFIXES),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_below(directory: Path) -> set[Path]:
    return {path.resolve() for path in directory.rglob("*") if path.is_file()}


def _targets(destination: Path) -> list[Path]:
    bundle = [
        destination / f"{artifact_id}{suffix}"
        for _, artifact_id, suffixes in BUNDLE_SPECS
        for suffix in suffixes
    ]
    return bundle + [
        destination / f"{INDEX_ID}.json",
        destination / f"{INDEX_ID}_sha256.txt",
    ]


def _load_publisher(module_name: str):
    return importlib.import_module(module_name)


def publish_all(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    evaluator = load_json(run_dir / "artifacts/offline_topology/summary.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or evaluator.get("overall_status") != EXPECTED_STATUS
        or evaluator.get("scientific_gate", {}).get("passed") is not True
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("offline-topology evidence publication requires the sealed scientific PASS")
    source = verify_complete_run_seal(root, run_dir, EXPECTED_STATUS)

    targets = _targets(destination)
    if any(path.exists() for path in targets):
        raise RuntimeError("offline-topology evidence target exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    initial_files = _files_below(destination)
    initial_directories = {path.resolve() for path in destination.rglob("*") if path.is_dir()}
    results = []
    try:
        for module_name, artifact_id, suffixes in BUNDLE_SPECS:
            before = _files_below(destination)
            result = _load_publisher(module_name).publish(run_dir, destination)
            expected = {(destination / f"{artifact_id}{suffix}").resolve() for suffix in suffixes}
            if result.get("figure_id") != artifact_id or _files_below(destination) - before != expected:
                raise RuntimeError(f"offline-topology publisher output set drift: {artifact_id}")
            manifest = destination / f"{artifact_id}_sha256.txt"
            results.append(
                {
                    "module": module_name,
                    "artifact_id": artifact_id,
                    "published_files": len(expected),
                    "bundle_manifest": str(manifest.relative_to(root)),
                    "bundle_manifest_sha256": _sha256(manifest),
                }
            )
        index = destination / f"{INDEX_ID}.json"
        aggregate_manifest = destination / f"{INDEX_ID}_sha256.txt"
        write_json(
            index,
            {
                "schema_version": "gse_offline_topology_evidence_publication_v1",
                "index_id": INDEX_ID,
                "source_run": str(run_dir.relative_to(root)),
                "source_run_seal_sha256": source["seal_sha256"],
                "source_run_seal_entries": source["entries"],
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "manual_value_entry": False,
                "bundles": results,
                "generator": str(Path(__file__).resolve().relative_to(root)),
                "generator_sha256": _sha256(Path(__file__).resolve()),
            },
        )
        expected_before_manifest = {path.resolve() for path in targets if path != aggregate_manifest}
        if _files_below(destination) - initial_files != expected_before_manifest:
            raise RuntimeError("offline-topology publication produced undeclared files")
        aggregate_manifest.write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(root)}\n"
                for path in sorted(expected_before_manifest)
            ),
            encoding="utf-8",
        )
        if _files_below(destination) - initial_files != {path.resolve() for path in targets}:
            raise RuntimeError("offline-topology aggregate manifest coverage is incomplete")
    except Exception:
        for path in sorted(_files_below(destination) - initial_files, reverse=True):
            path.unlink()
        current_directories = {path.resolve() for path in destination.rglob("*") if path.is_dir()}
        for directory in sorted(current_directories - initial_directories, reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise
    return {
        "index_id": INDEX_ID,
        "bundles": len(results),
        "published_files": len(targets),
        "manifest_sha256": _sha256(destination / f"{INDEX_ID}_sha256.txt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish_all(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
