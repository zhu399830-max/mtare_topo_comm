#!/usr/bin/env python3
"""Finalize complete TF32-off C07 evidence without repeating model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_EVIDENCE_CORRECTIVE_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_EVIDENCE_CORRECTIVE_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_EVIDENCE_CORRECTIVE_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_corrective_v1r_seed0"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_corrective_v1r.json"
SOURCE_RUNNER = PROJECT_ROOT / "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1.py"
EXPECTED_SOURCE_SEAL_FILES = 28
EXPECTED_ROWS = 64_644
EXPECTED_FORWARD_ROWS = 387_864
EXPECTED_CONTRACT = {
    "deterministic_algorithms": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_seal(source: Path, project_root: Path, expected_count: int) -> dict[str, str]:
    """Verify every immutable source entry and reject path/population drift."""
    seal = source / "artifacts/evidence_sha256.txt"
    if not seal.is_file():
        raise RuntimeError("source evidence seal is missing")
    source_prefix = source.resolve()
    entries: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in entries:
            raise RuntimeError(f"duplicate source seal path: {relative}")
        path = (project_root / relative).resolve()
        try:
            path.relative_to(source_prefix)
        except ValueError as exc:
            raise RuntimeError(f"source seal escaped source run: {relative}") from exc
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        entries[relative] = expected
    if len(entries) != expected_count:
        raise RuntimeError(
            f"source seal population drift: expected {expected_count}, got {len(entries)}"
        )
    return entries


def source_bug_signature(path: Path) -> bool:
    """Recognize the exact post-seal integer/len wrapper defect being corrected."""
    text = path.read_text(encoding="utf-8")
    return (
        "entries = _seal(run)" in text
        and '"entries": len(entries)' in text
        and '"evidence_entries": len(entries)' in text
    )


def seal_run(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    checks: dict[str, bool] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    source_entries: dict[str, str] = {}
    tests_returncode: int | None = None
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("TF32 evidence corrective executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"TF32 evidence corrective Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            if not path.is_file():
                raise RuntimeError(f"TF32 evidence corrective input missing: {relative}")
            before[relative] = sha256(path)
            if before[relative] != expected:
                raise RuntimeError(f"TF32 evidence corrective input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"TF32 evidence corrective tool drift: {record['path']}")

        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        write_json(run / "config/environment.json", {
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "operation": "zero_inference_evidence_audit",
        })
        write_json(run / "config/source_integrity_before.json", before)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        tests = [
            "tests/v3/unit/test_primitive_relation_observable_c07_tf32_evidence_corrective_v1.py",
            "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
            "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
            )
        tests_returncode = result.returncode
        if tests_returncode:
            raise RuntimeError("TF32 evidence corrective tests failed")

        state = load_json(SOURCE / "RUN_STATE.json")
        outer = load_json(SOURCE / "metrics/summary.json")
        evaluation = load_json(SOURCE / "metrics/evaluation/summary.json")
        comparison = load_json(SOURCE / "metrics/old_vs_corrected.json")
        resources = load_json(SOURCE / "metrics/resource_monitor.json")
        source_entries = verify_source_seal(SOURCE, PROJECT_ROOT, EXPECTED_SOURCE_SEAL_FILES)
        source_spec = load_json(SOURCE_SPEC)
        upstream_inputs = {
            relative: sha256(PROJECT_ROOT / relative)
            for relative in source_spec["frozen_inputs"]
        }
        upstream_tools = {
            record["path"]: sha256(PROJECT_ROOT / record["path"])
            for record in source_spec["frozen_tools"].values()
        }
        seeds = evaluation.get("c07", {}).get("seeds", [])
        checks = {
            "source_compute_completed_before_wrapper_defect": (
                state.get("state") == "COMPLETED"
                and state.get("error") is None
                and outer.get("overall_status") == "PASS_SYSTEM_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1"
                and outer.get("error") is None
            ),
            "source_post_seal_typeerror_signature_exact": source_bug_signature(SOURCE_RUNNER),
            "source_full_inference_completed": (
                outer.get("model_forward_rows") == EXPECTED_FORWARD_ROWS
                and outer.get("c07_rows_per_seed") == EXPECTED_ROWS
                and outer.get("frozen_seeds") == 3
            ),
            "source_numerical_contract_exact": evaluation.get("numerical_contract") == EXPECTED_CONTRACT,
            "source_three_seed_population_complete": (
                len(seeds) == 3
                and [int(item["seed"]) for item in seeds] == [0, 1, 2]
                and all(int(item["metrics"]["rows"]) == EXPECTED_ROWS for item in seeds)
            ),
            "source_scientific_fail_exact": (
                evaluation.get("overall_status") == "FAIL_PRIMITIVE_RELATION_OBSERVABLE_THREE_SEED_C07_V1"
                and evaluation.get("scientific_pass") is False
                and evaluation.get("decision") == "STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH"
                and evaluation.get("c07", {}).get("passing_seeds") == 0
                and all(item["comparison"]["pass"] is False for item in seeds)
            ),
            "safe_attachment_nonzero_gate_failed_all_seeds": all(
                int(item["metrics"]["attachment_safe_selection"]["true_positive"]) == 0
                for item in seeds
            ),
            "old_and_corrected_decision_consistent": (
                comparison.get("old_passing_seeds") == 0
                and comparison.get("corrected_passing_seeds") == 0
                and comparison.get("scientific_decision_changed") is False
            ),
            "source_seal_28_files_exact": len(source_entries) == EXPECTED_SOURCE_SEAL_FILES,
            "source_upstream_inputs_unchanged": upstream_inputs == source_spec["frozen_inputs"],
            "source_upstream_tools_unchanged": all(
                upstream_tools[record["path"]] == record["sha256"]
                for record in source_spec["frozen_tools"].values()
            ),
            "resource_contract_pass": (
                resources.get("returncode") == 2
                and resources.get("killed_for_host_limit") is False
                and int(outer.get("peak_host_rss_bytes", 2**63)) <= 16 * 1024**3
                and int(outer.get("peak_cuda_reserved_bytes", 2**63)) <= 16 * 1024**3
            ),
            "isolation_pass": (
                outer.get("optimizer_steps") == 0
                and outer.get("c08_rows_read") == 0
                and outer.get("c09_c10_worlds_read") == 0
                and outer.get("graph_replays") == 0
                and outer.get("mtare_worlds_read") == 0
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"TF32 evidence corrective checks failed: {checks}")

        destination = run / "metrics/source_evaluation"
        destination.mkdir(parents=True, exist_ok=False)
        copies = [
            (SOURCE / "metrics/evaluation/summary.json", destination / "summary.json"),
            (SOURCE / "metrics/evaluation/c07_seed0.json", destination / "c07_seed0.json"),
            (SOURCE / "metrics/evaluation/c07_seed1.json", destination / "c07_seed1.json"),
            (SOURCE / "metrics/evaluation/c07_seed2.json", destination / "c07_seed2.json"),
            (SOURCE / "metrics/old_vs_corrected.json", run / "metrics/old_vs_corrected.json"),
            (SOURCE / "metrics/resource_monitor.json", run / "metrics/source_resource_monitor.json"),
            (SOURCE / "logs/00_unit_tests.log", run / "logs/source_00_unit_tests.log"),
            (SOURCE / "logs/01_c07_tf32_corrective.log", run / "logs/source_01_c07_tf32_corrective.log"),
        ]
        copied: dict[str, dict[str, str]] = {}
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError(f"TF32 evidence copy mismatch: {source}")
            copied[str(target.relative_to(run))] = {
                "source": str(source.relative_to(PROJECT_ROOT)), "sha256": sha256(target),
            }
        for suffix in ("png", "pdf", "svg"):
            source = SOURCE / f"metrics/evaluation/primitive_relation_observable_comparison.{suffix}"
            metric_target = destination / f"primitive_relation_observable_comparison.{suffix}"
            preview_target = run / "previews" / f"primitive_relation_observable_tf32_corrected.{suffix}"
            shutil.copy2(source, metric_target)
            shutil.copy2(source, preview_target)
            if sha256(source) != sha256(metric_target) or sha256(source) != sha256(preview_target):
                raise RuntimeError(f"TF32 figure copy mismatch: {suffix}")
        write_json(run / "artifacts/source_corrective_provenance.json", {
            "source_run": str(SOURCE.relative_to(PROJECT_ROOT)),
            "source_run_state": state,
            "source_evidence_sha256_list_sha256": sha256(SOURCE / "artifacts/evidence_sha256.txt"),
            "source_seal_files": len(source_entries),
            "source_model_forward_rows": EXPECTED_FORWARD_ROWS,
            "new_model_forward_rows": 0,
            "new_optimizer_steps": 0,
            "copied_files": copied,
            "wrapper_defect": "_seal_returns_int_but_runner_called_len_on_that_int_after_sealing",
        })

        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("TF32 evidence corrective changed frozen inputs")
        write_json(run / "config/source_integrity_after.json", after)
        if time.monotonic() - started > 600:
            raise RuntimeError("TF32 evidence corrective wall time exceeded")
        if sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > 20 * 1024**2:
            raise RuntimeError("TF32 evidence corrective output exceeded")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    duration = time.monotonic() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(run / "metrics/summary.json", {
        "schema_version": "primitive_relation_observable_c07_tf32_evidence_corrective_v1",
        "overall_status": overall,
        "evidence_scientific_pass": overall == PASS,
        "observable_relation_model_scientific_pass": False,
        "scientific_decision": "STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH" if overall == PASS else None,
        "checks": checks, "error": error, "tests_returncode": tests_returncode,
        "source_seal_files": len(source_entries),
        "source_model_forward_rows": EXPECTED_FORWARD_ROWS,
        "new_model_forward_rows": 0, "optimizer_steps": 0,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "peak_host_rss_kib": peak_rss_kib, "duration_seconds": duration,
        "source_unchanged": bool(before and before == after),
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": duration,
    })
    expected_entries = len([path for path in run.rglob("*") if path.is_file()]) + 1
    write_json(run / "artifacts/seal_summary.json", {
        "schema_version": "primitive_relation_observable_c07_tf32_evidence_seal_v1",
        "expected_evidence_entries": expected_entries,
        "overall_status": overall, "error": error,
    })
    evidence_entries = seal_run(run)
    print(json.dumps({
        "overall_status": overall, "error": error,
        "observable_relation_model_scientific_pass": False,
        "scientific_decision": "STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH",
        "evidence_entries": evidence_entries,
        "expected_evidence_entries": expected_entries,
    }, indent=2, sort_keys=True))
    return 0 if overall == PASS and error is None and evidence_entries == expected_entries else 2


if __name__ == "__main__":
    raise SystemExit(main())
