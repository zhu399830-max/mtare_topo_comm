#!/usr/bin/env python3
"""Publish the sealed GSE teacher-distribution figure into the paper tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate2_20260824_gse_mesh_teacher_distribution_v1r_seed0"
EXPECTED_RUN_STATUS = "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1R"
EXPECTED_EXECUTOR_STATUS = "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1"
SOURCE_FILES = {
    "previews/gse_teacher_distribution.png": "gse_teacher_distribution.png",
    "previews/gse_teacher_distribution.pdf": "gse_teacher_distribution.pdf",
    "artifacts/event_distribution.csv": "gse_teacher_distribution.csv",
    "metrics/summary.json": "gse_teacher_distribution_summary.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_seal(run_dir: Path) -> tuple[Path, dict[str, str]]:
    seal_path = run_dir / "artifacts/evidence_sha256.txt"
    if not seal_path.is_file():
        raise RuntimeError("source evidence seal is missing")
    entries: dict[str, str] = {}
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    return seal_path, entries


def publish(run_dir: Path, destination: Path, *, project_root: Path | None = None) -> dict:
    root = (project_root or PROJECT_ROOT).resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    if run_dir.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected source run identity")
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    runner_summary = load_json(run_dir / "metrics/runner_summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != EXPECTED_RUN_STATUS:
        raise RuntimeError("source run is not a completed PASS")
    if (
        summary.get("overall_status") != EXPECTED_EXECUTOR_STATUS
        or runner_summary.get("overall_status") != EXPECTED_RUN_STATUS
    ):
        raise RuntimeError("source summaries do not agree on PASS")
    if summary.get("train_worlds_read") != 80 or any(
        summary.get(key) != 0
        for key in ("validation_worlds_read", "strict_test_worlds_read", "mtare_worlds_read")
    ):
        raise RuntimeError("source split contract mismatch")

    seal_path, entries = _source_seal(run_dir)
    source_prefix = run_dir.relative_to(root).as_posix() + "/"
    verified: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = run_dir / relative
        sealed_relative = source_prefix + relative
        observed = _sha256(path)
        if entries.get(sealed_relative) != observed:
            raise RuntimeError(f"source file is absent from seal or drifted: {relative}")
        verified[relative] = observed

    targets = [destination / name for name in SOURCE_FILES.values()]
    targets.extend(
        (
            destination / "gse_teacher_distribution_provenance.json",
            destination / "gse_teacher_distribution_sha256.txt",
        )
    )
    if any(path.exists() for path in targets):
        raise RuntimeError("paper figure destination already exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    for relative, name in SOURCE_FILES.items():
        shutil.copyfile(run_dir / relative, destination / name)

    provenance_path = destination / "gse_teacher_distribution_provenance.json"
    provenance = {
        "schema_version": "gse_paper_figure_provenance_v1",
        "figure_id": "gse_teacher_distribution",
        "source_run": str(run_dir.relative_to(root)),
        "source_run_id": EXPECTED_RUN_ID,
        "source_seal": str(seal_path.relative_to(root)),
        "source_seal_sha256": _sha256(seal_path),
        "selection_rule": "all 80 frozen training worlds; no outcome-based world selection",
        "validation_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "source_files": verified,
        "generator": "tools/v3/execute_gse_mesh_teacher_distribution_v1.py",
        "publisher": "tools/v3/publish_gse_teacher_distribution_figure_v1.py",
        "publisher_sha256": _sha256(Path(__file__).resolve()),
    }
    write_json(provenance_path, provenance)
    manifest_path = destination / "gse_teacher_distribution_sha256.txt"
    published = sorted(path for path in targets if path != manifest_path)
    manifest_path.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in published),
        encoding="utf-8",
    )
    return {
        "figure_id": provenance["figure_id"],
        "source_run_id": EXPECTED_RUN_ID,
        "published_files": len(published) + 1,
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/gse_graph",
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
