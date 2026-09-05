#!/usr/bin/env python3
"""Atomically publish every pre-registered GSE C09 paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
INDEX_ID = "gse_perception_evidence_index"
FIGURE_SUFFIXES = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
TABLE_SUFFIXES = (".csv", ".md", ".tex", "_source.json", "_provenance.json", "_sha256.txt")
BUNDLE_SPECS = (
    ("publish_gse_perception_figure_v1", "gse_perception_validation", FIGURE_SUFFIXES),
    ("publish_gse_rejection_analysis_figure_v1", "gse_rejection_analysis", FIGURE_SUFFIXES),
    ("publish_gse_exit_token_validation_figure_v1", "gse_exit_token_validation", FIGURE_SUFFIXES),
    ("publish_gse_uncertainty_analysis_figure_v1", "gse_uncertainty_analysis", FIGURE_SUFFIXES),
    ("publish_gse_per_world_perception_figure_v1", "gse_per_world_perception", FIGURE_SUFFIXES),
    ("publish_gse_event_class_table_v1", "gse_event_class_table", TABLE_SUFFIXES),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_publisher(module_name: str):
    return importlib.import_module(module_name)


def _bundle_targets(destination: Path) -> list[Path]:
    return [
        destination / f"{artifact_id}{suffix}"
        for _, artifact_id, suffixes in BUNDLE_SPECS
        for suffix in suffixes
    ]


def _files_below(directory: Path) -> set[Path]:
    return {path.resolve() for path in directory.rglob("*") if path.is_file()}


def publish_all(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    gate = load_json(run_dir / "metrics/perception_gate.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or gate.get("passed") is not True
        or gate.get("strict_test_worlds_read") != 0
        or gate.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("complete perception evidence publication requires the sealed C09 PASS")
    source_seal = verify_complete_run_seal(root, run_dir, EXPECTED_STATUS)

    index_path = destination / f"{INDEX_ID}.json"
    manifest_path = destination / f"{INDEX_ID}_sha256.txt"
    targets = _bundle_targets(destination) + [index_path, manifest_path]
    if any(path.exists() for path in targets):
        raise RuntimeError("perception evidence publication target exists; refusing overwrite")

    destination.mkdir(parents=True, exist_ok=True)
    initial_files = _files_below(destination)
    initial_directories = {path.resolve() for path in destination.rglob("*") if path.is_dir()}
    results = []
    try:
        for module_name, artifact_id, suffixes in BUNDLE_SPECS:
            before = _files_below(destination)
            module = _load_publisher(module_name)
            result = module.publish(run_dir, destination)
            if result.get("figure_id", result.get("table_id")) != artifact_id:
                raise RuntimeError(f"publisher identity mismatch: {module_name}")
            expected = {(destination / f"{artifact_id}{suffix}").resolve() for suffix in suffixes}
            created = _files_below(destination) - before
            if created != expected:
                raise RuntimeError(f"publisher output set differs from its frozen bundle: {artifact_id}")
            results.append(
                {
                    "module": module_name,
                    "artifact_id": artifact_id,
                    "published_files": len(expected),
                    "bundle_manifest": str((destination / f"{artifact_id}_sha256.txt").relative_to(root)),
                    "bundle_manifest_sha256": _sha256(destination / f"{artifact_id}_sha256.txt"),
                }
            )
        seal = run_dir / "artifacts/evidence_sha256.txt"
        write_json(
            index_path,
            {
                "schema_version": "gse_perception_evidence_publication_v1",
                "index_id": INDEX_ID,
                "source_run": str(run_dir.relative_to(root)),
                "source_run_seal": str(seal.relative_to(root)),
                "source_run_seal_sha256": source_seal["seal_sha256"],
                "source_run_seal_entries": source_seal["entries"],
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "manual_value_entry": False,
                "bundles": results,
                "generator": str(Path(__file__).resolve().relative_to(root)),
                "generator_sha256": _sha256(Path(__file__).resolve()),
            },
        )
        expected_before_manifest = {path.resolve() for path in targets if path != manifest_path}
        actual_before_manifest = _files_below(destination) - initial_files
        if actual_before_manifest != expected_before_manifest:
            raise RuntimeError("complete perception publication produced an undeclared file set")
        evidence = sorted(expected_before_manifest)
        manifest_path.write_text(
            "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in evidence),
            encoding="utf-8",
        )
        if _files_below(destination) - initial_files != {path.resolve() for path in targets}:
            raise RuntimeError("complete perception publication manifest coverage is incomplete")
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
        "manifest_sha256": _sha256(manifest_path),
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
