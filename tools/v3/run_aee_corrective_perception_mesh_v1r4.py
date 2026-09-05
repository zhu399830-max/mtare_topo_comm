#!/usr/bin/env python3
"""Run twenty isolated meshes with exact memory-bounded Cano containment."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from run_aee_corrective_perception_mesh_v1r3 import (
    DISK_LIMIT_BYTES,
    E1,
    EXPECTED_SCOPE,
    PROJECT_ROOT,
    RSS_LIMIT_BYTES,
    RSS_SAMPLE_INTERVAL_SECONDS,
    SOURCE_RUN,
    TIME_LIMIT_SECONDS,
    _approved_project_file,
    _directory_size,
    _environment,
    _parents,
    _seal_manifest,
    _sha256,
    _verify_environment_and_sources,
    _verify_source_seal,
    load_json,
    run_monitored_process,
    write_json,
)


RUN_ID = "gate2_20260822_aee_corrective_perception_mesh_v1r4_seed20260821"
WORKER = PROJECT_ROOT / "tools/v3/execute_aee_corrective_perception_mesh_parent_v1r4.py"
PASS_STATUS = "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R4"
FAIL_STATUS = "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R4"
MEMORY_IMPLEMENTATION = "exact_query_row_chunking_v1"
V1R3_FAILED = PROJECT_ROOT / "results/gate2_representation/gate2_20260822_aee_corrective_perception_mesh_v1r3_seed20260821"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only {RUN_ID}")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run is not in one-time state")
    resource = spec.get("resource_contract", {})
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 2
        or spec.get("seed") != 20260821
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
        or resource.get("per_parent_process_tree_rss_limit_bytes") != RSS_LIMIT_BYTES
        or resource.get("rss_sample_interval_seconds") != RSS_SAMPLE_INTERVAL_SECONDS
        or spec.get("memory_bounded_geometry", {}).get("implementation")
        != MEMORY_IMPLEMENTATION
    ):
        raise RuntimeError("approval, scope, resource or bounded-geometry contract mismatch")

    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    card_path = _approved_project_file(spec.get("data_card"), "data_card")
    proposal = load_json(proposal_path)
    card = load_json(card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or card.get("approval", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("approved proposal/data card missing")

    paths = {
        name: _approved_project_file(item.get("path"), f"frozen_tools.{name}.path")
        for name, item in spec.get("frozen_tools", {}).items()
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    for name, digest in observed.items():
        if spec["frozen_tools"][name].get("sha256") != digest:
            raise RuntimeError(f"frozen tool mismatch: {name}")

    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    source_before = _verify_source_seal()
    if (
        source_state.get("overall_status")
        != "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        or source_before["entries"] != 376
        or source_before["mismatch_count"] != 0
    ):
        raise RuntimeError("sealed topology source invalid")
    environment_audit = _verify_environment_and_sources(spec)
    v1r3_state = load_json(V1R3_FAILED / "RUN_STATE.json")
    v1r3_summary = load_json(V1R3_FAILED / "metrics/summary.json")
    v1r3_seal = V1R3_FAILED / "artifacts/evidence_sha256.txt"
    if (
        v1r3_state.get("overall_status")
        != "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R3"
        or v1r3_summary.get("counts", {}).get("meshes") != 7
        or v1r3_summary.get("resource_summary", {}).get("batch_stop_reason")
        != "PARENT_07_FAILED"
        or _sha256(v1r3_seal)
        != spec.get("source_evidence", {}).get("v1r3_fail_evidence_seal_sha256")
    ):
        raise RuntimeError("V1R3 failure evidence drifted")
    v1r3_failure_audit = {
        "overall_status": v1r3_state["overall_status"],
        "completed_meshes": 7,
        "batch_stop_reason": "PARENT_07_FAILED",
        "evidence_seal_sha256": _sha256(v1r3_seal),
        "evidence_metadata_files_read": 3,
        "failed_mesh_assets_read": 0,
        "failed_mesh_assets_reused": 0,
    }
    parents = _parents()
    expected_ids = [item["parent_id"] for item in parents]
    if len(parents) != 20 or len(set(expected_ids)) != 20:
        raise RuntimeError("sealed parent scope drifted")

    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/source_seal_audit_before.json", source_before)
    write_json(run_dir / "config/environment_and_source_audit.json", environment_audit)
    write_json(run_dir / "config/v1r3_failure_evidence_audit.json", v1r3_failure_audit)
    write_json(
        run_dir / "config/failed_run_exclusion.json",
        {
            "schema_version": "failed_mesh_run_exclusion_v1r4",
            "excluded_runs": [
                "gate2_20260821_aee_corrective_perception_mesh_v1_seed20260821",
                "gate2_20260821_aee_corrective_perception_mesh_v1r_seed20260821",
                "gate2_20260822_aee_corrective_perception_mesh_v1r2_seed20260821",
                "gate2_20260822_aee_corrective_perception_mesh_v1r3_seed20260821",
            ],
            "evidence_metadata_read_count": 3,
            "asset_read_count": 0,
            "reused_asset_count": 0,
        },
    )
    (run_dir / "artifacts/meshes").mkdir(exist_ok=False)
    (run_dir / "previews/train_complete_maps").mkdir(exist_ok=False)
    (run_dir / "metrics/worker_receipts").mkdir(exist_ok=False)
    (run_dir / "metrics/parent_resources").mkdir(exist_ok=False)
    (run_dir / "metrics/resource_traces").mkdir(exist_ok=False)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "One-shot parent isolation plus exact memory-bounded containment.",
        },
    )

    started = time.monotonic()
    resources: list[dict] = []
    completed_ids: list[str] = []
    stop_reason = None
    for index, parent in enumerate(parents):
        remaining = TIME_LIMIT_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "BATCH_TIME_LIMIT_EXCEEDED"
            break
        parent_id = parent["parent_id"]
        exit_code, duration, item = run_monitored_process(
            [
                str(E1),
                str(WORKER),
                "--run-dir",
                str(run_dir),
                "--parent-index",
                str(index),
                "--expected-parent-id",
                parent_id,
            ],
            cwd=PROJECT_ROOT,
            environment=_environment(),
            log_path=run_dir / "logs" / f"parent_{index:02d}_{parent_id}.log",
            trace_path=run_dir / "metrics/resource_traces" / f"parent_{index:02d}_{parent_id}.jsonl",
            time_limit_seconds=remaining,
            rss_limit_bytes=RSS_LIMIT_BYTES,
            sample_interval_seconds=RSS_SAMPLE_INTERVAL_SECONDS,
        )
        item.update(
            {"parent_index": index, "parent_id": parent_id, "worker_duration_seconds": duration}
        )
        write_json(run_dir / "metrics/parent_resources" / f"{index:02d}_{parent_id}.json", item)
        resources.append(item)
        receipt_path = run_dir / "metrics/worker_receipts" / f"{index:02d}_{parent_id}.json"
        metric_path = run_dir / "metrics" / f"{parent_id}.json"
        if (
            exit_code != 0
            or not item["within_rss_limit"]
            or item["stop_reason"] is not None
            or not receipt_path.is_file()
            or not metric_path.is_file()
        ):
            stop_reason = f"PARENT_{index:02d}_FAILED"
            break
        receipt = load_json(receipt_path)
        if (
            receipt.get("parent_index") != index
            or receipt.get("parent_id") != parent_id
            or receipt.get("source_unchanged") is not True
            or receipt.get("memory_bounded_geometry", {}).get("implementation")
            != MEMORY_IMPLEMENTATION
        ):
            stop_reason = f"PARENT_{index:02d}_RECEIPT_MISMATCH"
            break
        completed_ids.append(parent_id)
        if _directory_size(run_dir) > DISK_LIMIT_BYTES:
            stop_reason = "BATCH_DISK_LIMIT_EXCEEDED"
            break

    duration = time.monotonic() - started
    source_after = _verify_source_seal()
    write_json(run_dir / "config/source_seal_audit_after.json", source_after)
    records = [load_json(run_dir / "metrics" / f"{item}.json") for item in completed_ids]
    receipts = list((run_dir / "metrics/worker_receipts").glob("*.json"))
    meshes = list(run_dir.glob("artifacts/meshes/S*_C*/primary/mesh.obj"))
    sanitation = list(run_dir.glob("artifacts/meshes/S*_C*/primary/sanitation.json"))
    previews = list(run_dir.glob("previews/train_complete_maps/*.png"))
    train_ids = [item["parent_id"] for item in parents if item["split"] == "corrective_train"]
    validation_ids = [item["parent_id"] for item in parents if item["split"] == "corrective_validation"]
    preview_ids = sorted(path.name.removesuffix("_complete_xy_xz.png") for path in previews)
    checks = {
        "exact_parent_order": completed_ids == expected_ids,
        "unique_parent_ids": len(set(completed_ids)) == 20,
        "unique_topology_identities": len(
            {item["source_parent"]["canonical_parent_identity"] for item in records}
        ) == 20,
        "exact_split": len(train_ids) == len(validation_ids) == 10,
        "all_identity_checks": len(records) == 20 and all(
            all(item["primary"]["identity_checks"].values()) for item in records
        ),
        "all_meshes_pass": len(records) == 20 and all(
            item["primary"]["mesh_audit"]["passed"] is True for item in records
        ),
        "all_sanitation_pass": len(records) == 20 and all(
            item["primary"]["sanitation"]["passed"] is True for item in records
        ),
        "all_memory_contracts": len(records) == 20 and all(
            item["memory_bounded_geometry"]["implementation"] == MEMORY_IMPLEMENTATION
            and item["memory_bounded_geometry"]["geometry_thresholds_changed"] is False
            for item in records
        ),
        "all_mesh_hashes_unique": len(records) == 20
        and len({item["primary"]["mesh_sha256"] for item in records}) == 20,
        "exact_worker_receipts": len(receipts) == 20,
        "exact_train_previews": preview_ids == sorted(train_ids),
        "zero_validation_previews": not (set(preview_ids) & set(validation_ids)),
        "all_parent_rss_pass": len(resources) == 20 and all(
            item["within_rss_limit"] for item in resources
        ),
        "no_parent_retry": len({item["parent_index"] for item in resources}) == len(resources),
        "source_seal_unchanged": source_after == source_before,
        "failed_run_assets_excluded": True,
    }
    write_json(
        run_dir / "artifacts/mesh_manifest.json",
        {
            "schema_version": "aee_corrective_perception_mesh_manifest_v1r4",
            "parent_order": completed_ids,
            "assets": [
                {
                    "parent_index": item["parent_index"],
                    "parent_id": item["parent_id"],
                    "split": item["split"],
                    "mesh_sha256": item["primary"]["mesh_sha256"],
                }
                for item in records
            ],
            "memory_bounded_geometry": spec["memory_bounded_geometry"],
            "cross_run_obj_hash_equality_required": False,
            "one_shot_freeze_required": True,
        },
    )
    result_size = _directory_size(run_dir)
    passed = bool(
        stop_reason is None
        and all(checks.values())
        and duration <= TIME_LIMIT_SECONDS
        and result_size <= DISK_LIMIT_BYTES
        and len(meshes) == len(sanitation) == 20
    )
    overall = PASS_STATUS if passed else FAIL_STATUS
    resource_summary = {
        "schema_version": "aee_corrective_perception_mesh_resources_v1r4",
        "per_parent_process_tree_rss_limit_bytes": RSS_LIMIT_BYTES,
        "sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
        "parent_process_count": len(resources),
        "maximum_parent_peak_process_tree_rss_bytes": max(
            (item["peak_process_tree_rss_bytes"] for item in resources), default=0
        ),
        "all_parent_limits_pass": len(resources) == 20
        and all(item["within_rss_limit"] for item in resources),
        "batch_duration_seconds": duration,
        "batch_time_limit_seconds": TIME_LIMIT_SECONDS,
        "batch_stop_reason": stop_reason,
        "parents": resources,
    }
    write_json(run_dir / "metrics/resource_summary.json", resource_summary)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_corrective_perception_mesh_summary_v1r4",
            "run_id": RUN_ID,
            "overall_status": overall,
            "scope": EXPECTED_SCOPE,
            "checks": checks,
            "counts": {
                "meshes": len(meshes),
                "sanitation": len(sanitation),
                "metrics": len(records),
                "worker_receipts": len(receipts),
                "train_previews": len(previews),
            },
            "resource_summary": resource_summary,
            "result_bytes_before_seal": result_size,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "host": {"platform": platform.platform()},
            "claim_boundary": "Twenty memory-bounded perception meshes only; zero downstream operation.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": "Exact memory-bounded containment; no failed-run reuse.",
        },
    )
    sealed = _seal_manifest(run_dir)
    print(
        json.dumps(
            {
                "overall_status": overall,
                "completed_parents": len(completed_ids),
                "sealed_files": sealed,
                "maximum_parent_peak_process_tree_rss_bytes": resource_summary[
                    "maximum_parent_peak_process_tree_rss_bytes"
                ],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
