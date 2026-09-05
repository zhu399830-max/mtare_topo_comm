#!/usr/bin/env python3
"""Run twenty one-shot parent-isolated mesh workers and seal the batch."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.process_tree_resource import run_monitored_process
from mtare_topo.governance import load_json, write_json
from run_aee_corrective_topology_candidate_audit_v1 import (
    E1,
    _approved_project_file,
    _environment,
    _sha256,
    _verify_environment_and_sources,
)
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest


RUN_ID = "gate2_20260822_aee_corrective_perception_mesh_v1r3_seed20260821"
SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260821_aee_corrective_topology_candidate_audit_v1_seed20260821"
WORKER = PROJECT_ROOT / "tools/v3/execute_aee_corrective_perception_mesh_parent_v1r3.py"
TIME_LIMIT_SECONDS = 7200
DISK_LIMIT_BYTES = 1024**3
RSS_LIMIT_BYTES = 2 * 1024**3
RSS_SAMPLE_INTERVAL_SECONDS = 0.25
EXPECTED_SCOPE = {
    "topology_parents_selected": 20,
    "topology_reconstructions": 20,
    "native_materializations": 20,
    "sanitized_primary_assets": 20,
    "train_complete_previews": 10,
    "validation_complete_previews": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "c09_reads": 0,
    "c10_reads": 0,
    "mtare_changes": 0,
}


def _parents() -> list[dict]:
    parents = load_json(SOURCE_RUN / "artifacts/selected_parent_manifest.json")["parents"]
    if len(parents) != 20:
        raise RuntimeError("sealed selected-parent count drifted")
    return parents


def _verify_source_seal() -> dict:
    manifest = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    mismatch = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatch.append(relative)
    return {
        "entries": len(lines),
        "mismatch_count": len(mismatch),
        "mismatches": mismatch,
        "manifest_sha256": _sha256(manifest),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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
    resource_contract = spec.get("resource_contract", {})
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 2
        or spec.get("seed") != 20260821
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
        or resource_contract.get("per_parent_process_tree_rss_limit_bytes")
        != RSS_LIMIT_BYTES
        or resource_contract.get("rss_sample_interval_seconds")
        != RSS_SAMPLE_INTERVAL_SECONDS
    ):
        raise RuntimeError("approval, mesh scope or per-parent resource contract mismatch")

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
        "runner": Path(__file__).resolve(),
        "worker": WORKER,
        "resource_monitor": PROJECT_ROOT / "src/mtare_topo/evaluation/process_tree_resource.py",
        "batch_v1_executor_helpers": PROJECT_ROOT / "tools/v3/execute_aee_corrective_perception_mesh_v1.py",
        "m0_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
        "m1r_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_m1r.py",
        "mesh_contract": PROJECT_ROOT / "src/mtare_topo/data/cano_perception_mesh_contract.py",
        "topology_executor": PROJECT_ROOT / "tools/v3/execute_aee_corrective_topology_candidate_audit_v1.py",
        "proposal": proposal_path,
        "data_card": card_path,
        "environment": _approved_project_file(
            spec.get("environment", {}).get("path"), "environment.path"
        ),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if set(observed) != set(spec.get("frozen_tools", {})):
        raise RuntimeError("frozen tool set mismatch")
    for name, digest in observed.items():
        if spec["frozen_tools"][name].get("sha256") != digest:
            raise RuntimeError(f"frozen tool mismatch: {name}")

    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    source_seal_before = _verify_source_seal()
    if (
        source_state.get("overall_status")
        != "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        or source_seal_before["entries"] != 376
        or source_seal_before["mismatch_count"] != 0
    ):
        raise RuntimeError("sealed topology source invalid")
    environment_audit = _verify_environment_and_sources(spec)
    parents = _parents()
    expected_ids = [item["parent_id"] for item in parents]
    if len(set(expected_ids)) != 20:
        raise RuntimeError("sealed parent IDs are not unique")

    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/source_seal_audit_before.json", source_seal_before)
    write_json(
        run_dir / "config/environment_and_source_audit.json", environment_audit
    )
    write_json(
        run_dir / "config/failed_run_exclusion.json",
        {
            "schema_version": "failed_mesh_run_exclusion_v1r3",
            "excluded_runs": [
                "gate2_20260821_aee_corrective_perception_mesh_v1_seed20260821",
                "gate2_20260821_aee_corrective_perception_mesh_v1r_seed20260821",
                "gate2_20260822_aee_corrective_perception_mesh_v1r2_seed20260821",
            ],
            "read_count": 0,
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
            "note": "Twenty one-shot fresh-process parent workers; no failed-run reuse.",
        },
    )

    started = time.monotonic()
    resources = []
    completed_ids = []
    batch_stop_reason = None
    for index, parent in enumerate(parents):
        elapsed = time.monotonic() - started
        remaining = TIME_LIMIT_SECONDS - elapsed
        if remaining <= 0:
            batch_stop_reason = "BATCH_TIME_LIMIT_EXCEEDED"
            break
        parent_id = parent["parent_id"]
        argv = [
            str(E1),
            str(WORKER),
            "--run-dir",
            str(run_dir),
            "--parent-index",
            str(index),
            "--expected-parent-id",
            parent_id,
        ]
        exit_code, duration, resource = run_monitored_process(
            argv,
            cwd=PROJECT_ROOT,
            environment=_environment(),
            log_path=run_dir / "logs" / f"parent_{index:02d}_{parent_id}.log",
            trace_path=run_dir
            / "metrics/resource_traces"
            / f"parent_{index:02d}_{parent_id}.jsonl",
            time_limit_seconds=remaining,
            rss_limit_bytes=RSS_LIMIT_BYTES,
            sample_interval_seconds=RSS_SAMPLE_INTERVAL_SECONDS,
        )
        resource.update(
            {"parent_index": index, "parent_id": parent_id, "worker_duration_seconds": duration}
        )
        write_json(
            run_dir / "metrics/parent_resources" / f"{index:02d}_{parent_id}.json",
            resource,
        )
        resources.append(resource)
        receipt_path = (
            run_dir / "metrics/worker_receipts" / f"{index:02d}_{parent_id}.json"
        )
        metric_path = run_dir / "metrics" / f"{parent_id}.json"
        if (
            exit_code != 0
            or not resource["within_rss_limit"]
            or resource["stop_reason"] is not None
            or not receipt_path.is_file()
            or not metric_path.is_file()
        ):
            batch_stop_reason = f"PARENT_{index:02d}_FAILED"
            break
        receipt = load_json(receipt_path)
        if (
            receipt.get("parent_index") != index
            or receipt.get("parent_id") != parent_id
            or receipt.get("source_unchanged") is not True
        ):
            batch_stop_reason = f"PARENT_{index:02d}_RECEIPT_MISMATCH"
            break
        completed_ids.append(parent_id)
        if _directory_size(run_dir) > DISK_LIMIT_BYTES:
            batch_stop_reason = "BATCH_DISK_LIMIT_EXCEEDED"
            break

    total_duration = time.monotonic() - started
    source_seal_after = _verify_source_seal()
    write_json(run_dir / "config/source_seal_audit_after.json", source_seal_after)
    records = [
        load_json(run_dir / "metrics" / f"{parent_id}.json")
        for parent_id in completed_ids
    ]
    receipts = sorted((run_dir / "metrics/worker_receipts").glob("*.json"))
    meshes = list(run_dir.glob("artifacts/meshes/S*_C*/primary/mesh.obj"))
    sanitation = list(run_dir.glob("artifacts/meshes/S*_C*/primary/sanitation.json"))
    previews = list(run_dir.glob("previews/train_complete_maps/*.png"))
    train_ids = [item["parent_id"] for item in parents if item["split"] == "corrective_train"]
    validation_ids = [
        item["parent_id"] for item in parents if item["split"] == "corrective_validation"
    ]
    preview_ids = sorted(path.name.removesuffix("_complete_xy_xz.png") for path in previews)
    checks = {
        "exact_parent_order": completed_ids == expected_ids,
        "unique_parent_ids": len(set(completed_ids)) == 20,
        "unique_topology_identities": len(
            {item["source_parent"]["canonical_parent_identity"] for item in records}
        )
        == 20,
        "exact_split": len(train_ids) == len(validation_ids) == 10,
        "all_identity_checks": len(records) == 20
        and all(all(item["primary"]["identity_checks"].values()) for item in records),
        "all_meshes_pass": len(records) == 20
        and all(item["primary"]["mesh_audit"]["passed"] is True for item in records),
        "all_sanitation_pass": len(records) == 20
        and all(item["primary"]["sanitation"]["passed"] is True for item in records),
        "all_mesh_hashes_unique": len(records) == 20
        and len({item["primary"]["mesh_sha256"] for item in records}) == 20,
        "exact_worker_receipts": len(receipts) == 20,
        "exact_train_previews": preview_ids == sorted(train_ids),
        "zero_validation_previews": not (set(preview_ids) & set(validation_ids)),
        "all_parent_rss_pass": len(resources) == 20
        and all(item["within_rss_limit"] for item in resources),
        "no_parent_retry": len({item["parent_index"] for item in resources})
        == len(resources),
        "source_seal_unchanged": source_seal_after == source_seal_before,
        "failed_run_assets_excluded": True,
    }
    mesh_manifest = {
        "schema_version": "aee_corrective_perception_mesh_manifest_v1r3",
        "parent_order": completed_ids,
        "assets": [
            {
                "parent_index": item["parent_index"],
                "parent_id": item["parent_id"],
                "split": item["split"],
                "mesh_sha256": item["primary"]["mesh_sha256"],
                "metric": str(
                    (run_dir / "metrics" / f"{item['parent_id']}.json").relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
            for item in records
        ],
        "cross_run_obj_hash_equality_required": False,
        "one_shot_freeze_required": True,
    }
    write_json(run_dir / "artifacts/mesh_manifest.json", mesh_manifest)
    result_size = _directory_size(run_dir)
    passed = bool(
        batch_stop_reason is None
        and all(checks.values())
        and total_duration <= TIME_LIMIT_SECONDS
        and result_size <= DISK_LIMIT_BYTES
        and len(meshes) == len(sanitation) == 20
    )
    overall = (
        "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R3"
        if passed
        else "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R3"
    )
    resource_summary = {
        "schema_version": "aee_corrective_perception_mesh_parent_isolation_resources_v1r3",
        "per_parent_process_tree_rss_limit_bytes": RSS_LIMIT_BYTES,
        "sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
        "parent_process_count": len(resources),
        "maximum_parent_peak_process_tree_rss_bytes": max(
            (item["peak_process_tree_rss_bytes"] for item in resources), default=0
        ),
        "all_parent_limits_pass": len(resources) == 20
        and all(item["within_rss_limit"] for item in resources),
        "batch_duration_seconds": total_duration,
        "batch_time_limit_seconds": TIME_LIMIT_SECONDS,
        "batch_stop_reason": batch_stop_reason,
        "parents": resources,
    }
    write_json(run_dir / "metrics/resource_summary.json", resource_summary)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_corrective_perception_mesh_summary_v1r3",
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
            "claim_boundary": "Twenty immutable perception meshes only; zero LiDAR, teacher, formal data, training, C09/C10 or planner change.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": "One-shot parent-isolated native meshes; no failed-run reuse.",
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
