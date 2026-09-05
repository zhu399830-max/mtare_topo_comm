#!/usr/bin/env python3
"""Seal completed V1R attribution evidence after its post-compute NameError."""

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


RUN_ID = "gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_EVIDENCE_CORRECTIVE_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_EVIDENCE_CORRECTIVE_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_EVIDENCE_CORRECTIVE_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1r_seed0"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_v1r.json"
EXPECTED_DIAGNOSIS = "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE"
EXPECTED_DECISION = "STOP_CURRENT_RELATION_HEAD_AND_REASSESS_METHOD"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def _verify_source_seal() -> int:
    seal = SOURCE / "artifacts/evidence_sha256.txt"
    count = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"source attribution seal mismatch: {relative}")
        count += 1
    if count != 23:
        raise RuntimeError("source attribution seal population drift")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; checks = {}; diagnosis = None; decision = None
    before = {}; after = {}; tests_returncode = None; source_seal_files = 0
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("attribution evidence corrective executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"attribution evidence corrective Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"attribution evidence corrective input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"attribution evidence corrective tool drift: {record['path']}")

        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        write_json(run / "config/environment.json", {
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "operation": "zero_inference_evidence_audit",
        })
        write_json(run / "config/source_integrity_before.json", before)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        tests = [
            "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
            "tests/v3/unit/test_primitive_relation_failure_attribution.py",
            "tests/v3/unit/test_primitive_relation_metrics.py",
            "tests/v3/unit/test_evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
            )
        tests_returncode = result.returncode
        if tests_returncode:
            raise RuntimeError("attribution evidence corrective tests failed")

        state = load_json(SOURCE / "RUN_STATE.json")
        outer = load_json(SOURCE / "metrics/summary.json")
        attribution = load_json(SOURCE / "metrics/attribution/summary.json")
        source_seal_files = _verify_source_seal()
        source_spec = load_json(SOURCE_SPEC)
        upstream = {
            relative: _sha(PROJECT_ROOT / relative) for relative in source_spec["frozen_inputs"]
        }
        checks = {
            "source_failed_only_post_compute_nameerror": (
                state.get("state") == "FAILED"
                and state.get("error") == "NameError: name 'SEEDS' is not defined"
                and outer.get("error") == state.get("error")
            ),
            "source_full_inference_completed": (
                outer.get("model_inference_rows") == 193_932
                and attribution.get("model_inference_rows") == 193_932
            ),
            "source_attribution_scientific_evidence_complete": (
                attribution.get("overall_status") == "PASS_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_V1"
                and attribution.get("scientific_pass") is True
                and all(attribution.get("checks", {}).values())
            ),
            "source_diagnosis_exact": attribution.get("diagnosis") == EXPECTED_DIAGNOSIS,
            "source_decision_exact": attribution.get("decision") == EXPECTED_DECISION,
            "all_diagnostic_conditions_zero_passing_seeds": (
                len(attribution.get("condition_passing_seed_counts", {})) == 7
                and set(attribution["condition_passing_seed_counts"].values()) == {0}
            ),
            "three_seed_outputs_complete": all(
                (SOURCE / f"metrics/attribution/seed{seed}.json").is_file() for seed in range(3)
            ),
            "paper_figures_complete": all(
                (SOURCE / f"metrics/attribution/primitive_relation_sparse_port_failure_attribution.{suffix}").is_file()
                for suffix in ("png", "pdf", "svg")
            ),
            "source_seal_23_files_exact": source_seal_files == 23,
            "source_upstream_inputs_unchanged": upstream == source_spec["frozen_inputs"],
            "resource_contract_pass": (
                int(outer.get("peak_host_rss_kib", 2**63)) <= 16 * 1024 * 1024
                and int(attribution.get("peak_cuda_reserved_bytes", 2**63)) <= 16 * 1024**3
            ),
            "isolation_pass": (
                attribution.get("optimizer_steps") == 0
                and attribution.get("c08_rows_read") == 0
                and attribution.get("c09_c10_worlds_read") == 0
                and attribution.get("graph_replays") == 0
                and attribution.get("mtare_worlds_read") == 0
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"attribution evidence corrective checks failed: {checks}")

        destination = run / "metrics/attribution"
        destination.mkdir(parents=True, exist_ok=False)
        names = [
            "summary.json", "per_task.csv", "figure_source.json",
            *[f"seed{seed}.json" for seed in range(3)],
            *[f"primitive_relation_sparse_port_failure_attribution.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        copied = {}
        for name in names:
            source = SOURCE / "metrics/attribution" / name
            target = destination / name
            shutil.copy2(source, target)
            copied[name] = {"source_sha256": _sha(source), "copied_sha256": _sha(target)}
            if copied[name]["source_sha256"] != copied[name]["copied_sha256"]:
                raise RuntimeError(f"attribution evidence copy mismatch: {name}")
        for suffix in ("png", "pdf", "svg"):
            shutil.copy2(
                destination / f"primitive_relation_sparse_port_failure_attribution.{suffix}",
                run / "previews",
            )
        diagnosis = attribution["diagnosis"]; decision = attribution["decision"]
        write_json(run / "artifacts/source_attribution_provenance.json", {
            "source_run": str(SOURCE.relative_to(PROJECT_ROOT)),
            "source_run_state": state,
            "source_evidence_sha256_list_sha256": _sha(SOURCE / "artifacts/evidence_sha256.txt"),
            "source_seal_files": source_seal_files,
            "copied_files": copied,
            "new_model_inference_rows": 0,
            "new_optimizer_steps": 0,
        })

        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("attribution evidence corrective changed frozen inputs")
        write_json(run / "config/source_integrity_after.json", after)
        if time.monotonic() - started > 600:
            raise RuntimeError("attribution evidence corrective wall time exceeded")
        if sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > 20 * 1024**2:
            raise RuntimeError("attribution evidence corrective output exceeded")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    duration = time.monotonic() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(run / "metrics/summary.json", {
        "schema_version": "primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "sparse_port_model_scientific_pass": False,
        "diagnosis": diagnosis, "decision": decision, "checks": checks,
        "error": error, "tests_returncode": tests_returncode,
        "source_seal_files": source_seal_files,
        "new_model_inference_rows": 0, "source_model_inference_rows": 193_932,
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "peak_host_rss_kib": peak_rss_kib, "duration_seconds": duration,
        "source_unchanged": bool(before and before == after),
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": duration,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "error": error, "diagnosis": diagnosis,
        "decision": decision, "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
