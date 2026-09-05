#!/usr/bin/env python3
"""Re-evaluate sealed composition-anchor checkpoints after a symmetry-only fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0"
PASS = "PASS_SYSTEM_PRIMITIVE_COMPOSITION_ANCHOR_C07_EVALUATION_CORRECTIVE_V1"
FAIL = "FAIL_SYSTEM_PRIMITIVE_COMPOSITION_ANCHOR_C07_EVALUATION_CORRECTIVE_V1"
CARD_STATUS = (
    "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_"
    "C07_EVALUATION_CORRECTIVE_V1"
)
PYTHON = (
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/"
    "phase3_torch290_cu129_zarr2187_v1/bin/python"
)
SOURCE = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0"
)
SOURCE_SPEC = PROJECT_ROOT / (
    "configs/v3/gate3/primitive_composition_anchor_three_seed_training_v1.json"
)
P1A = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
)
P1B = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
)
OBSERVABILITY = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
)
ANCHORS = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
)
BASELINE = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
)
DIAGNOSTIC = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
)

EXPECTED_SOURCE_SEAL_FILES = 49
EXPECTED_ROWS = 64_644
EXPECTED_STEPS = 30_699
EXPECTED_SOURCE_ERROR = (
    "RuntimeError: composition-anchor C07 evaluator produced no scientific result"
)
EXPECTED_EVALUATOR_ERROR = (
    "ValueError: composition-anchor safe score must be finite and symmetric"
)
MAXIMUM_HOST_RSS_BYTES = 4 * 1024**3
MAXIMUM_GPU_BYTES = 16 * 1024**3
MAXIMUM_RESULT_BYTES = 512 * 1024**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def verify_source_seal(source: Path, expected_count: int) -> dict[str, str]:
    seal = source / "artifacts/evidence_sha256.txt"
    if not seal.is_file():
        raise RuntimeError("composition-anchor source seal is missing")
    prefix = source.resolve()
    entries: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in entries:
            raise RuntimeError(f"duplicate source seal path: {relative}")
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(prefix)
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


def source_failure_signature(state: dict, summary: dict, evaluation_log: str) -> bool:
    subprocesses = {row.get("stage"): row for row in summary.get("subprocesses", [])}
    return (
        state.get("state") == "FAILED"
        and state.get("error") == EXPECTED_SOURCE_ERROR
        and summary.get("error") == EXPECTED_SOURCE_ERROR
        and summary.get("optimizer_steps") == EXPECTED_STEPS
        and summary.get("c08_rows_read") == 0
        and summary.get("graph_replays") == 0
        and summary.get("mtare_worlds_read") == 0
        and all(
            subprocesses.get(f"seed{seed}_training", {}).get("returncode") == 0
            for seed in range(3)
        )
        and subprocesses.get("c07_evaluation", {}).get("returncode") == 1
        and EXPECTED_EVALUATOR_ERROR in evaluation_log
    )


def parse_nvidia_process_memory(text: str, pid: int) -> int:
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            return int(fields[1].split()[0]) * 1024**2
    return 0


def _gpu_process_memory(pid: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return parse_nvidia_process_memory(output, pid)


def _host_peak(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise RuntimeError("unexpected VmHWM format")
                return int(fields[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_evaluator_monitored(
    command: list[str], *, log: Path, environment: dict[str, str], timeout_seconds: int,
) -> dict[str, int | bool]:
    peak_host = peak_gpu = samples = 0
    killed_for_limit = timed_out = False
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
        try:
            while process.poll() is None:
                samples += 1
                peak_host = max(peak_host, _host_peak(process.pid))
                if samples % 5 == 1:
                    peak_gpu = max(peak_gpu, _gpu_process_memory(process.pid))
                if peak_host > MAXIMUM_HOST_RSS_BYTES or peak_gpu > MAXIMUM_GPU_BYTES:
                    killed_for_limit = True
                    _terminate(process)
                    break
                if time.monotonic() - started > timeout_seconds:
                    timed_out = True
                    _terminate(process)
                    break
                time.sleep(1.0)
        finally:
            _terminate(process)
        returncode = int(process.returncode)
    peak_host = max(
        peak_host, int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024),
    )
    return {
        "returncode": returncode,
        "monitor_samples": samples,
        "peak_host_rss_bytes": peak_host,
        "peak_gpu_process_memory_bytes": peak_gpu,
        "killed_for_resource_limit": killed_for_limit,
        "timed_out": timed_out,
    }


def seal_run(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
        ),
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
    model_scientific_pass = False
    decision = None
    checks: dict[str, bool] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    source_entries: dict[str, str] = {}
    evaluation: dict = {}
    monitor: dict = {}
    tests_returncode: int | None = None
    try:
        if (
            run.name != RUN_ID
            or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED"
        ):
            raise RuntimeError("composition-anchor evaluation corrective executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(
                f"composition-anchor corrective Data Card invalid: {validation.errors}"
            )
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            if not path.is_file():
                raise RuntimeError(f"composition-anchor corrective input missing: {relative}")
            before[relative] = sha256(path)
            if before[relative] != expected:
                raise RuntimeError(f"composition-anchor corrective input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(
                    f"composition-anchor corrective tool drift: {record['path']}"
                )

        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "zarr": zarr.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_versions = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if versions != expected_versions:
            raise RuntimeError(f"composition-anchor corrective environment drift: {versions}")
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        write_json(run / "config/environment.json", {
            "versions": versions, "platform": platform.platform(),
        })
        write_json(run / "config/source_integrity_before.json", before)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        environment["PYTHONHASHSEED"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_composition_anchor_model.py",
            "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
            "tests/v3/unit/test_primitive_composition_anchor_c07_evaluation_corrective_v1.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests],
                cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
                timeout=600, check=False,
            )
        tests_returncode = int(completed.returncode)
        if tests_returncode:
            raise RuntimeError("composition-anchor corrective tests failed")

        source_state = load_json(SOURCE / "RUN_STATE.json")
        source_summary = load_json(SOURCE / "metrics/summary.json")
        source_log = (SOURCE / "logs/04_c07_evaluation.log").read_text(encoding="utf-8")
        source_entries = verify_source_seal(SOURCE, EXPECTED_SOURCE_SEAL_FILES)
        source_spec = load_json(SOURCE_SPEC)
        seed_contracts = []
        for seed in range(3):
            seed_root = SOURCE / f"artifacts/models/seed{seed}"
            summary = load_json(seed_root / "summary.json")
            contract = load_json(SOURCE / f"metrics/seed{seed}_contract_checks.json")
            best_epoch = int(summary["best_epoch"])
            seed_contracts.append(
                summary.get("optimizer_steps") == 10_233
                and summary.get("epochs") == 3
                and summary.get("fit_rows_per_epoch") == 426_552
                and summary.get("c07_rows_per_epoch") == EXPECTED_ROWS
                and summary.get("frozen_state_sha256_before")
                == summary.get("frozen_state_sha256_after")
                and all(contract.values())
                and sha256(seed_root / "selected.pt")
                == sha256(seed_root / f"epoch_{best_epoch:02d}.pt")
            )
        checks.update({
            "source_failure_is_exact_post_training_symmetry_error": source_failure_signature(
                source_state, source_summary, source_log,
            ),
            "source_seal_49_files_exact": len(source_entries) == EXPECTED_SOURCE_SEAL_FILES,
            "source_three_seed_training_contracts_complete": all(seed_contracts),
            "source_frozen_inputs_still_exact": all(
                sha256(PROJECT_ROOT / relative) == expected
                for relative, expected in source_spec["frozen_inputs"].items()
            ),
        })
        if not all(checks.values()):
            raise RuntimeError(f"composition-anchor source evidence invalid: {checks}")

        output = run / "metrics/evaluation"
        command = [
            PYTHON,
            str(PROJECT_ROOT / "tools/v3/evaluate_primitive_composition_anchor_three_seed_v1.py"),
            "--models-root", str(SOURCE / "artifacts/models"),
            "--sensor-root", str(P1A / "artifacts/dataset"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--observability-root", str(OBSERVABILITY / "artifacts/endpoint_observability"),
            "--anchor-root", str(ANCHORS / "artifacts/materialized/anchor_targets"),
            "--baseline-c07-summary", str(BASELINE / "metrics/summary.json"),
            "--existence-threshold-source", str(DIAGNOSTIC / "metrics/diagnostic/summary.json"),
            "--output-dir", str(output),
        ]
        (run / "config/evaluation_command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8",
        )
        monitor = run_evaluator_monitored(
            command, log=run / "logs/01_c07_evaluation_corrective.log",
            environment=environment, timeout_seconds=14_400,
        )
        write_json(run / "metrics/evaluation_resource_monitor.json", monitor)
        if (output / "summary.json").is_file():
            evaluation = load_json(output / "summary.json")
        recognized = {
            "PASS_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1",
            "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1",
        }
        seeds = evaluation.get("c07", {}).get("seeds", [])
        numerical = [
            row.get("metrics", {}).get("safe_score_numerical_contract", {}) for row in seeds
        ]
        model_scientific_pass = bool(evaluation.get("scientific_pass"))
        decision = evaluation.get("decision")
        checks.update({
            "corrective_evaluator_return_recognized": monitor.get("returncode") in (0, 2),
            "corrective_evaluator_not_killed": not monitor.get("killed_for_resource_limit")
                and not monitor.get("timed_out"),
            "complete_three_seed_c07_population": (
                len(seeds) == 3
                and [row.get("seed") for row in seeds] == [0, 1, 2]
                and all(row.get("metrics", {}).get("rows") == EXPECTED_ROWS for row in seeds)
            ),
            "scientific_result_recognized": evaluation.get("overall_status") in recognized,
            "corrected_score_finite_and_exact_symmetric": (
                len(numerical) == 3
                and all(row.get("corrected_all_finite") is True for row in numerical)
                and all(row.get("corrected_exact_symmetric") is True for row in numerical)
                and all(row.get("corrected_max_asymmetry") == 0.0 for row in numerical)
            ),
            "legacy_left_associated_asymmetry_reproduced": (
                len(numerical) == 3
                and all(row.get("legacy_left_associated_max_asymmetry", 0.0) > 0.0 for row in numerical)
            ),
            "resource_contract_pass": (
                int(monitor.get("peak_host_rss_bytes", 2**63)) <= MAXIMUM_HOST_RSS_BYTES
                and int(monitor.get("peak_gpu_process_memory_bytes", 2**63)) <= MAXIMUM_GPU_BYTES
            ),
            "zero_new_training_and_strict_isolation": (
                evaluation.get("c08_rows_read") == 0
                and evaluation.get("c09_c10_worlds_read") == 0
                and evaluation.get("graph_replays") == 0
                and evaluation.get("mtare_worlds_read") == 0
            ),
        })
        if not all(checks.values()):
            raise RuntimeError(f"composition-anchor corrective evidence failed: {checks}")
        for suffix in ("png", "pdf", "svg"):
            source = output / f"primitive_composition_anchor_c07_comparison.{suffix}"
            if not source.is_file():
                raise RuntimeError(f"composition-anchor corrective figure missing: {suffix}")
            shutil.copy2(source, run / "previews" / source.name)
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("composition-anchor corrective changed frozen inputs")
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "artifacts/corrective_provenance.json", {
            "source_run": str(SOURCE.relative_to(PROJECT_ROOT)),
            "source_run_state": source_state,
            "source_evidence_sha256_list_sha256": sha256(
                SOURCE / "artifacts/evidence_sha256.txt"
            ),
            "source_seal_files": len(source_entries),
            "new_optimizer_steps": 0,
            "new_model_forward_rows": 3 * EXPECTED_ROWS,
            "corrective": (
                "Group the two endpoint-evidence factors into one exactly symmetric "
                "outer product before multiplying by the already symmetric compatibility."
            ),
            "changed_scientific_inputs_or_thresholds": False,
        })
        if directory_size(run) > MAXIMUM_RESULT_BYTES:
            raise RuntimeError("composition-anchor corrective result exceeds 512 MiB")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8",
        )

    duration = time.monotonic() - started
    write_json(run / "metrics/summary.json", {
        "schema_version": "primitive_composition_anchor_c07_evaluation_corrective_v1",
        "overall_status": overall,
        "evidence_corrective_pass": overall == PASS and error is None,
        "model_scientific_pass": model_scientific_pass if overall == PASS else None,
        "scientific_decision": decision if overall == PASS else None,
        "evaluation": evaluation,
        "checks": checks,
        "source_seal_files": len(source_entries),
        "tests_returncode": tests_returncode,
        "optimizer_steps": 0,
        "model_forward_rows": 3 * EXPECTED_ROWS if evaluation else 0,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "resource_monitor": monitor,
        "source_unchanged": bool(before and before == after),
        "duration_seconds": duration,
        "error": error,
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": duration,
    })
    predicted_entries = len([
        path for path in run.rglob("*")
        if path.is_file() and path.name != "evidence_sha256.txt"
    ]) + 1
    write_json(run / "artifacts/seal_summary.json", {
        "schema_version": "primitive_composition_anchor_c07_evaluation_corrective_seal_v1",
        "expected_evidence_entries": predicted_entries,
        "overall_status": overall, "error": error,
    })
    evidence_entries = seal_run(run)
    print(json.dumps({
        "overall_status": overall,
        "model_scientific_pass": model_scientific_pass if overall == PASS else None,
        "scientific_decision": decision if overall == PASS else None,
        "error": error, "evidence_entries": evidence_entries,
    }, indent=2, sort_keys=True))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
