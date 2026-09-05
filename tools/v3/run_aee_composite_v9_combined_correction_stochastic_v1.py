#!/home/zeng-workstation/anaconda3/bin/python
"""Run V5 on the exact 30-case M1D slice of the composed Gate-6 matrix."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401 - initializes repository src path
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.governance import load_json
import run_aee_composite_v9_reanchor_stochastic_v1 as base
import run_mtare_single_robot_stochastic_v2 as archive_base
from run_aee_composite_v9_combined_correction_probe_v1 import (
    audit_combined_correction,
)


RUN_ID = "gate6_20260823_aee_composite_v9_combined_correction_stochastic_v1_seed20260820"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1"
EXPECTED_READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_READINESS_V1R"
EXPECTED_AUDIT_STATUS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"
CONTAINER_PREFIX = "mtare-v9-v5"
CASE_WRAPPER = "/workspace/tools/v3/run_mtare_single_robot_case_v3.py"
EXPECTED_SNAPSHOT_SCHEMA = "semantic_topology_global_node_v5_snapshot_v1"


def audit_matrix_case(case_dir: Path, *, require_reanchor: bool = False) -> dict[str, Any]:
    del require_reanchor
    return audit_combined_correction(case_dir, require_correction=False)


def validate_composed_paired_source_schedule(
    spec: dict[str, Any], cases: list[Any], *, project_root: Path = base.PROJECT_ROOT
) -> dict[str, Any]:
    root = project_root.resolve()
    source = (root / spec["paired_source_audit_run"]).resolve()
    source.relative_to(root)
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    manifest_path = source / "config/source_manifest.json"
    provenance_path = source / "config/source_provenance.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state, summary, manifest, provenance = (
        load_json(state_path), load_json(summary_path), load_json(manifest_path),
        load_json(provenance_path),
    )
    expected_status = spec.get("paired_source_audit_status")
    if expected_status != EXPECTED_AUDIT_STATUS:
        raise RuntimeError("V5 paired source audit status contract drift")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != expected_status
        or summary.get("overall_status") != expected_status
        or summary.get("source_case_count") != 90
        or summary.get("v9_case_count") != 30
        or manifest.get("case_count") != 90
        or provenance.get("source_mutation_permitted") is not False
    ):
        raise RuntimeError("V5 paired composed source is not the exact audit PASS")
    if base.matrix_base.sha256(seal_path) != spec["paired_source_audit_seal_sha256"]:
        raise RuntimeError("V5 paired source audit seal identity drift")
    entries = {
        relative: digest
        for digest, relative in (
            line.split("  ", 1) for line in seal_path.read_text(encoding="utf-8").splitlines()
        )
    }
    bound = {}
    for path in (state_path, summary_path, manifest_path, provenance_path):
        relative = path.relative_to(root).as_posix()
        observed = base.matrix_base.sha256(path)
        if entries.get(relative) != observed:
            raise RuntimeError(f"V5 paired source evidence drift: {relative}")
        bound[relative] = observed
    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list) or len(manifest_cases) != 90:
        raise RuntimeError("V5 paired source manifest is not exactly 90 cases")
    source_m1d = [row for row in manifest_cases if row.get("method_family") == "m1d_topology"]
    expected = [
        {
            "index": case.index, "case_id": case.case_id,
            "block_id": case.block_id, "method_family": case.method_family,
        }
        for case in cases
    ]
    observed = [
        {key: row.get(key) for key in ("index", "case_id", "block_id", "method_family")}
        for row in source_m1d
    ]
    if observed != expected:
        raise RuntimeError("V5 cases are not the exact composed-source M1D schedule")
    sources = Counter(row.get("source") for row in source_m1d)
    expected_sources = Counter({"failed_source_run": 22, "recovery_run": 8})
    if sources != expected_sources:
        raise RuntimeError("V5 paired M1D source composition drift")
    return {
        "schema_version": "aee_composite_v9_combined_correction_paired_source_v1",
        "source_audit_run": spec["paired_source_audit_run"],
        "source_audit_status": expected_status,
        "source_audit_seal_sha256": spec["paired_source_audit_seal_sha256"],
        "source_schedule_path": manifest_path.relative_to(root).as_posix(),
        "source_schedule_file_sha256": base.matrix_base.sha256(manifest_path),
        "source_schedule_content_sha256": base.schedule_content_sha256(manifest_cases),
        "matched_m1d_case_count": 30,
        "exact_order_and_identity_match": True,
        "m1d_source_counts": dict(sorted(sources.items())),
        "verified_bound_files": bound,
    }


def case_command(run_dir: Path, case: Any, matrix: dict[str, Any]) -> tuple[list[str], str]:
    archive_base.CONTAINER_PREFIX = CONTAINER_PREFIX
    command, name = archive_base.case_command(run_dir, case, matrix)
    command[-1] = command[-1].replace(
        "/workspace/tools/v3/run_mtare_single_robot_case_v1.py", CASE_WRAPPER
    )
    if CASE_WRAPPER not in command[-1]:
        raise RuntimeError("V5 stochastic runner failed to select the V3 case wrapper")
    return command, name


def summarize_mechanisms(
    summaries: list[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    reanchors, rejections = [], []
    outcomes: Counter[str] = Counter()
    for item in summaries:
        case_dir = run_dir / "artifacts/cases" / item["case"]["case_id"]
        snapshot = load_json(case_dir / item["planner_evidence"]["topology_snapshot"])
        runtime = snapshot["runtime"]
        reanchors.append(int(runtime["verified_reanchor_count"]))
        rejections.append(int(runtime["frontier_execution_rejection_count"]))
        outcomes.update({
            str(key): int(value)
            for key, value in runtime["frontier_execution_outcomes"].items()
        })
    return {
        "verified_reanchor_count_total": sum(reanchors),
        "verified_reanchor_case_count": sum(value > 0 for value in reanchors),
        "frontier_execution_rejection_count_total": sum(rejections),
        "frontier_execution_rejection_case_count": sum(value > 0 for value in rejections),
        "case_count_with_both_correction_types": sum(
            first > 0 and second > 0 for first, second in zip(reanchors, rejections)
        ),
        "frontier_execution_outcomes": dict(sorted(outcomes.items())),
    }


def main() -> int:
    archive_had_run_case = hasattr(archive_base, "run_case")
    archive_run_case = getattr(archive_base, "run_case", None)
    original = {
        "RUN_ID": base.RUN_ID,
        "STATUS_PASS": base.STATUS_PASS,
        "STATUS_FAIL": base.STATUS_FAIL,
        "EXPECTED_READINESS_STATUS": base.EXPECTED_READINESS_STATUS,
        "EXPECTED_CHECKPOINT_MODE": base.EXPECTED_CHECKPOINT_MODE,
        "CONTAINER_PREFIX": base.CONTAINER_PREFIX,
        "CASE_WRAPPER": base.CASE_WRAPPER,
        "EXPECTED_SNAPSHOT_SCHEMA": base.EXPECTED_SNAPSHOT_SCHEMA,
        "CASE_MECHANISM_AUDIT_KEY": base.CASE_MECHANISM_AUDIT_KEY,
        "MECHANISM_AUDITOR": base.MECHANISM_AUDITOR,
        "validate_paired_source_schedule": base.validate_paired_source_schedule,
        "case_command": base.case_command,
        "summarize_mechanisms": base.summarize_mechanisms,
    }
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.EXPECTED_READINESS_STATUS = EXPECTED_READINESS_STATUS
    base.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
    base.CONTAINER_PREFIX = CONTAINER_PREFIX
    base.CASE_WRAPPER = CASE_WRAPPER
    base.EXPECTED_SNAPSHOT_SCHEMA = EXPECTED_SNAPSHOT_SCHEMA
    base.CASE_MECHANISM_AUDIT_KEY = "combined_correction_mechanism_audit"
    base.MECHANISM_AUDITOR = audit_matrix_case
    base.validate_paired_source_schedule = validate_composed_paired_source_schedule
    base.case_command = case_command
    base.summarize_mechanisms = summarize_mechanisms
    # The V2 archive adapter delegates execution to its V1 base and does not
    # re-export ``run_case``.  The re-anchor orchestration expects that symbol
    # on the adapter, so bind the already-frozen V1 implementation explicitly.
    archive_base.run_case = base.matrix_base.run_case
    try:
        return base.main()
    finally:
        if archive_had_run_case:
            archive_base.run_case = archive_run_case
        else:
            delattr(archive_base, "run_case")
        for name, value in original.items():
            setattr(base, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
