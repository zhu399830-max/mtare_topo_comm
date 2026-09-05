#!/home/zeng-workstation/anaconda3/bin/python
"""Run the corrected V4 method on the frozen 30-case M1D schedule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.governance import load_json, write_json
import run_mtare_single_robot_stochastic_v1 as matrix_base
import run_mtare_single_robot_stochastic_v2 as archive_base
from run_aee_composite_v9_reanchor_probe_v1 import audit_reanchor_mechanism


RUN_ID = "gate6_20260823_aee_composite_v9_reanchor_stochastic_v1_seed20260820"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_REANCHOR_STOCHASTIC_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_REANCHOR_STOCHASTIC_V1"
EXPECTED_READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_REANCHOR_READINESS_V1"
EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
CONTAINER_PREFIX = "mtare-v9-v4"
MINIMUM_FREE_BYTES = 100 * 1024**3
CASE_WRAPPER = "/workspace/tools/v3/run_mtare_single_robot_case_v2.py"
EXPECTED_SNAPSHOT_SCHEMA = "semantic_topology_global_node_v4_snapshot_v1"
CASE_MECHANISM_AUDIT_KEY = "reanchor_mechanism_audit"
MECHANISM_AUDITOR = audit_reanchor_mechanism


def schedule_content_sha256(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_v4_cases(cases: list[Any]) -> list[Any]:
    selected = [case for case in cases if case.method_family == "m1d_topology"]
    if len(selected) != 30 or len({case.case_id for case in selected}) != 30:
        raise RuntimeError("V4 schedule is not exactly 30 unique M1D cases")
    blocks = Counter(case.block_id for case in selected)
    seeds = Counter(case.checkpoint_seed for case in selected)
    if len(blocks) != 10 or set(blocks.values()) != {3}:
        raise RuntimeError("V4 schedule is not ten blocks of three checkpoints")
    if seeds != Counter({0: 10, 1: 10, 2: 10}):
        raise RuntimeError("V4 checkpoint quotas are not 10/10/10")
    return selected


def validate_paired_source_schedule(
    spec: dict[str, Any], cases: list[Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    root = project_root.resolve()
    source_run = (root / spec["paired_source_run"]).resolve()
    source_run.relative_to(root)
    state = load_json(source_run / "RUN_STATE.json")
    summary = load_json(source_run / "metrics/summary.json")
    if state.get("state") != "FAILED" or state.get("overall_status") != spec["paired_source_status"]:
        raise RuntimeError("paired source run is not the expected sealed FAIL")
    if summary.get("completed_case_count") != 90:
        raise RuntimeError("paired source run did not complete all 90 simulations")
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if matrix_base.sha256(seal_path) != spec["paired_source_seal_sha256"]:
        raise RuntimeError("paired source seal identity drift")
    schedule_path = source_run / "config/case_schedule.json"
    schedule_file_sha256 = matrix_base.sha256(schedule_path)
    if schedule_file_sha256 != spec["paired_source_schedule_file_sha256"]:
        raise RuntimeError("paired source schedule file identity drift")
    relative = schedule_path.relative_to(root).as_posix()
    seal_entries = {
        path: digest
        for digest, path in (
            line.split("  ", 1) for line in seal_path.read_text(encoding="utf-8").splitlines()
        )
    }
    required_bound = {
        (source_run / "RUN_STATE.json").relative_to(root).as_posix(): matrix_base.sha256(source_run / "RUN_STATE.json"),
        (source_run / "metrics/summary.json").relative_to(root).as_posix(): matrix_base.sha256(source_run / "metrics/summary.json"),
        relative: schedule_file_sha256,
    }
    if any(seal_entries.get(path) != digest for path, digest in required_bound.items()):
        raise RuntimeError("paired source state, summary or schedule is not bound by its seal")
    source_cases = load_json(schedule_path).get("cases")
    if not isinstance(source_cases, list) or len(source_cases) != 90:
        raise RuntimeError("paired source schedule is not exactly 90 cases")
    schedule_canonical_sha256 = schedule_content_sha256(source_cases)
    if schedule_canonical_sha256 != spec["paired_source_schedule_content_sha256"]:
        raise RuntimeError("paired source schedule canonical-content identity drift")
    source_m1d = [item for item in source_cases if item.get("method_family") == "m1d_topology"]
    selected = [case.to_dict() for case in cases]
    if source_m1d != selected:
        raise RuntimeError("V4 cases are not the exact paired source M1D schedule")
    return {
        "schema_version": "aee_composite_v9_reanchor_paired_source_v1",
        "source_run": spec["paired_source_run"],
        "source_status": spec["paired_source_status"],
        "source_seal_sha256": spec["paired_source_seal_sha256"],
        "source_schedule_path": relative,
        "source_schedule_file_sha256": schedule_file_sha256,
        "source_schedule_content_sha256": schedule_canonical_sha256,
        "matched_m1d_case_count": 30,
        "exact_order_and_identity_match": True,
    }


def case_command(run_dir: Path, case: Any, matrix: dict[str, Any]) -> tuple[list[str], str]:
    archive_base.CONTAINER_PREFIX = CONTAINER_PREFIX
    command, name = archive_base.case_command(run_dir, case, matrix)
    command[-1] = command[-1].replace(
        "/workspace/tools/v3/run_mtare_single_robot_case_v1.py",
        CASE_WRAPPER,
    )
    if CASE_WRAPPER not in command[-1]:
        raise RuntimeError("V4 stochastic runner failed to select the V2 case wrapper")
    return command, name


def summarize_mechanisms(
    summaries: list[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    reanchors = []
    for item in summaries:
        case_dir = run_dir / "artifacts/cases" / item["case"]["case_id"]
        snapshot = load_json(case_dir / item["planner_evidence"]["topology_snapshot"])
        reanchors.append(int(snapshot["runtime"]["verified_reanchor_count"]))
    return {
        "verified_reanchor_count_total": sum(reanchors),
        "verified_reanchor_case_count": sum(value > 0 for value in reanchors),
    }


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "aee_composite_v9_reanchor_stochastic_summary_v1",
        "overall_status": STATUS_FAIL,
        "failure_reason": message,
        "completed_case_count": completed,
        "planned_case_count": 30,
        "training_steps": 0,
        "optimizer_steps": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "duration_seconds": time.monotonic() - started,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "FAILED", "overall_status": STATUS_FAIL,
    })
    matrix_base.seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("V4 stochastic run identity/state mismatch")
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "closed_loop_single":
            raise RuntimeError("Gate-6 closed-loop scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved autonomous execution scope required")
        for name, item in spec["frozen_tools"].items():
            if matrix_base.sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen V4 stochastic tool drift: {name}")

        matrix, all_cases, source_audit = matrix_base.validate_frozen_inputs(spec)
        cases = select_v4_cases(all_cases)
        paired_source = validate_paired_source_schedule(spec, cases)
        archive_base.EXPECTED_READINESS_STATUS = EXPECTED_READINESS_STATUS
        archive_base.EXPECTED_CHECKPOINT_MODE = EXPECTED_CHECKPOINT_MODE
        archive_base.TOPIC_CONTRACT = str(spec["readiness_topic_contract"])
        readiness = archive_base.verify_readiness_source(spec)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", archive_base.IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("V4 stochastic image identity drift")
        planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
        if matrix_base.image_file_hash(planner_path) != spec["planner_binary_sha256"]:
            raise RuntimeError("V4 stochastic planner binary identity drift")

        schedule = [case.to_dict() for case in cases]
        schedule_audit = {
            "schema_version": "aee_composite_v9_reanchor_schedule_audit_v1",
            "source_matrix_schedule_content_sha256": source_audit["schedule_sha256"],
            "paired_source_schedule_file_sha256": paired_source[
                "source_schedule_file_sha256"
            ],
            "paired_source_schedule_content_sha256": paired_source[
                "source_schedule_content_sha256"
            ],
            "selected_case_count": 30,
            "block_count": 10,
            "cases_per_block": 3,
            "checkpoint_case_counts": {str(seed): 10 for seed in (0, 1, 2)},
            "selection_rule": "method_family == m1d_topology; preserve frozen source order and identities",
        }
        write_json(run_dir / "config/readiness_source.json", readiness)
        write_json(run_dir / "config/source_matrix_audit.json", source_audit)
        write_json(run_dir / "config/paired_source.json", paired_source)
        write_json(run_dir / "config/schedule_audit.json", schedule_audit)
        write_json(run_dir / "config/case_schedule.json", {
            "schema_version": "aee_composite_v9_reanchor_case_schedule_v1", "cases": schedule,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        for case in cases:
            if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
                raise RuntimeError("free disk below 100 GiB before V4 stochastic case")
            command, name = case_command(run_dir, case, matrix)
            archive_base.run_case(command, name, run_dir / f"logs/cases/{case.case_id}.log")
            summary = archive_base.finalize_case(
                run_dir / f"artifacts/cases/{case.case_id}", case.case_id
            )
            snapshot = load_json(
                run_dir / f"artifacts/cases/{case.case_id}" /
                summary["planner_evidence"]["topology_snapshot"]
            )
            if snapshot.get("schema_version") != EXPECTED_SNAPSHOT_SCHEMA:
                raise RuntimeError(f"case did not use V4 node: {case.case_id}")
            summary[CASE_MECHANISM_AUDIT_KEY] = MECHANISM_AUDITOR(
                run_dir / f"artifacts/cases/{case.case_id}", require_reanchor=False
            )
            write_json(run_dir / f"artifacts/cases/{case.case_id}/summary.json", summary)
            summaries.append(summary)
            write_json(run_dir / "metrics/progress.json", {
                "schema_version": "aee_composite_v9_reanchor_stochastic_progress_v1",
                "completed_case_count": len(summaries),
                "planned_case_count": 30,
                "last_case_id": case.case_id,
                "elapsed_wall_sec": time.monotonic() - started,
                "free_disk_bytes": shutil.disk_usage(run_dir).free,
            })

        fallback_rates = [
            item["method_identity"]["post_warmup_fallback_rate"] for item in summaries
        ]
        summary = {
            "schema_version": "aee_composite_v9_reanchor_stochastic_summary_v1",
            "overall_status": STATUS_PASS,
            "completed_case_count": 30,
            "planned_case_count": 30,
            "block_count": 10,
            "checkpoint_case_counts": {"0": 10, "1": 10, "2": 10},
            "simulated_runtime_sec": 18000,
            "archive_bytes": sum(int(item["storage"]["archive_bytes"]) for item in summaries),
            "post_warmup_fallback_rate_mean": sum(fallback_rates) / len(fallback_rates),
            "post_warmup_fallback_rate_maximum": max(fallback_rates),
            **summarize_mechanisms(summaries, run_dir),
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed = matrix_base.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
