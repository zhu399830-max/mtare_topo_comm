#!/usr/bin/env python3
"""Independently verify the sealed AEE corrective mesh V1R4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


RUN_ID = "gate2_20260822_aee_corrective_perception_mesh_v1r4_seed20260821"
PASS_STATUS = "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R4"
SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260821_aee_corrective_topology_candidate_audit_v1_seed20260821"
RSS_LIMIT_BYTES = 2 * 1024**3


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    if run_dir.name != RUN_ID:
        errors.append("run ID mismatch")
    state = _load(run_dir / "RUN_STATE.json")
    summary = _load(run_dir / "metrics/summary.json")
    resource = _load(run_dir / "metrics/resource_summary.json")
    manifest = _load(run_dir / "artifacts/mesh_manifest.json")
    parents = _load(SOURCE_RUN / "artifacts/selected_parent_manifest.json")["parents"]
    expected_ids = [item["parent_id"] for item in parents]
    train_ids = sorted(
        item["parent_id"] for item in parents if item["split"] == "corrective_train"
    )
    validation_ids = {
        item["parent_id"] for item in parents if item["split"] == "corrective_validation"
    }

    if state.get("state") != "COMPLETED" or state.get("overall_status") != PASS_STATUS:
        errors.append("RUN_STATE is not the V1R4 PASS state")
    if summary.get("overall_status") != PASS_STATUS:
        errors.append("summary is not V1R4 PASS")
    if summary.get("counts") != {
        "meshes": 20,
        "sanitation": 20,
        "metrics": 20,
        "worker_receipts": 20,
        "train_previews": 10,
    }:
        errors.append("summary counts mismatch")
    if not summary.get("checks") or not all(summary["checks"].values()):
        errors.append("one or more runner checks failed")
    if manifest.get("parent_order") != expected_ids or len(manifest.get("assets", [])) != 20:
        errors.append("mesh manifest order/count mismatch")

    parent_resources = sorted((run_dir / "metrics/parent_resources").glob("*.json"))
    traces = sorted((run_dir / "metrics/resource_traces").glob("*.jsonl"))
    receipts = sorted((run_dir / "metrics/worker_receipts").glob("*.json"))
    if len({len(parent_resources), len(traces), len(receipts)}) != 1:
        errors.append("resource/trace/receipt cardinalities differ")
    if len(parent_resources) != 20 or len(traces) != 20 or len(receipts) != 20:
        errors.append("resource/trace/receipt count is not 20")
    seen_indices: list[int] = []
    for path in parent_resources:
        item = _load(path)
        seen_indices.append(item.get("parent_index"))
        if (
            item.get("within_rss_limit") is not True
            or item.get("stop_reason") is not None
            or item.get("peak_process_tree_rss_bytes", RSS_LIMIT_BYTES + 1)
            > RSS_LIMIT_BYTES
        ):
            errors.append(f"resource contract failed: {path.name}")
    if seen_indices != list(range(20)):
        errors.append("parent resource indices are not exactly 0..19")
    for path in traces:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines or json.loads(lines[-1]).get("final") is not True:
            errors.append(f"resource trace lacks final record: {path.name}")
    for path in receipts:
        item = _load(path)
        memory = item.get("memory_bounded_geometry", {})
        if (
            item.get("schema_version")
            != "aee_corrective_perception_mesh_parent_receipt_v1r4"
            or item.get("source_unchanged") is not True
            or memory.get("implementation") != "exact_query_row_chunking_v1"
            or memory.get("geometry_thresholds_changed") is not False
        ):
            errors.append(f"receipt contract failed: {path.name}")

    preview_ids = sorted(
        path.name.removesuffix("_complete_xy_xz.png")
        for path in (run_dir / "previews/train_complete_maps").glob("*.png")
    )
    if preview_ids != train_ids or set(preview_ids) & validation_ids:
        errors.append("train/validation preview isolation failed")
    exclusion = _load(run_dir / "config/failed_run_exclusion.json")
    if (
        exclusion.get("evidence_metadata_read_count") != 3
        or exclusion.get("asset_read_count") != 0
        or exclusion.get("reused_asset_count") != 0
    ):
        errors.append("failed-run evidence/asset boundary mismatch")
    if (
        resource.get("parent_process_count") != 20
        or resource.get("all_parent_limits_pass") is not True
        or resource.get("batch_stop_reason") is not None
        or resource.get("maximum_parent_peak_process_tree_rss_bytes", RSS_LIMIT_BYTES + 1)
        > RSS_LIMIT_BYTES
    ):
        errors.append("batch resource summary failed")

    seal_path = run_dir / "artifacts/evidence_sha256.txt"
    seal_lines = [line for line in seal_path.read_text(encoding="utf-8").splitlines() if line]
    seal_mismatches: list[str] = []
    for line in seal_lines:
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        if not target.is_file() or _sha256(target) != expected:
            seal_mismatches.append(relative)
    if seal_mismatches:
        errors.append("evidence seal mismatch")
    return {
        "schema_version": "aee_corrective_perception_mesh_independent_verification_v1r4",
        "run_id": run_dir.name,
        "passed": not errors,
        "errors": errors,
        "parent_count": len(expected_ids),
        "seal_entries": len(seal_lines),
        "seal_mismatch_count": len(seal_mismatches),
        "seal_sha256": _sha256(seal_path),
        "maximum_parent_peak_process_tree_rss_bytes": resource.get(
            "maximum_parent_peak_process_tree_rss_bytes"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
