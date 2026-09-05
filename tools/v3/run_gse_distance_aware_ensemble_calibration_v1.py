#!/usr/bin/env python3
"""Execute and seal the immutable distance-aware V2 ensemble calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260826_gse_distance_aware_ensemble_calibration_v1_seed0"
PASS_STATUS = "PASS_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
FAIL_STATUS = "FAIL_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
SOURCE_V2 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
SOURCE_V1R = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_open_set_association_corrective_v1r_seed0"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_distance_aware_ensemble_v1.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_failed_source(run: Path, status: str, entries: int) -> dict:
    state = load_json(run / "RUN_STATE.json")
    summary = load_json(run / "metrics/summary.json")
    if state.get("state") != "FAILED" or state.get("overall_status") != status:
        raise RuntimeError(f"source state drift: {run.name}")
    if summary.get("overall_status") != status:
        raise RuntimeError(f"source summary drift: {run.name}")
    seal = run / "artifacts/evidence_sha256.txt"
    sealed = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        if not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        sealed.add(target.resolve())
    actual = {path.resolve() for path in run.rglob("*") if path.is_file() and path != seal}
    if len(sealed) != entries or sealed != actual:
        raise RuntimeError(f"source seal coverage drift: {run.name}")
    return {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "status": status,
        "seal_entries": len(sealed),
        "seal_sha256": _sha256(seal),
    }


def _peak_rss(log_path: Path) -> int | None:
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        log_path.read_text(encoding="utf-8"),
    )
    return int(match.group(1)) if match else None


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time ensemble calibration run state mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    source_before = None
    source_after = None
    result = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "threshold_calibration":
            raise RuntimeError("ensemble calibration scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("ensemble calibration is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS:
            raise RuntimeError("ensemble calibration Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool mismatch: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen input mismatch: {relative}")
        observed_environment = json.loads(subprocess.check_output(
            [str(PYTHON), "-c", "import json,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__},sort_keys=True))"],
            text=True,
        ))
        if observed_environment != {"numpy": "2.1.3", "python": "3.13.5"}:
            raise RuntimeError(f"ensemble calibration environment drift: {observed_environment}")
        source_before = {
            "v2": _verify_failed_source(SOURCE_V2, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2", 40),
            "v1r": _verify_failed_source(SOURCE_V1R, "FAIL_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R", 37),
        }
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": observed_environment,
            "source_before": source_before, "gpu_used": False,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Read-only C07-C08 score calibration; C09/C10/M-TARE forbidden.",
        })
        output = run_dir / "artifacts/calibration"
        log_path = run_dir / "logs/00_calibration.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EVALUATOR),
            "--source-run", str(SOURCE_V2), "--output-dir", str(output),
        ]
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False,
            )
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        peak_rss = _peak_rss(log_path)
        source_after = {
            "v2": _verify_failed_source(SOURCE_V2, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2", 40),
            "v1r": _verify_failed_source(SOURCE_V1R, "FAIL_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R", 37),
        }
        if source_before != source_after:
            raise RuntimeError("source changed during ensemble calibration")
        selection = result.get("selection") or {}
        if (
            completed.returncode != 0
            or peak_rss is None
            or peak_rss > 2 * 1024**2
            or result.get("overall_status") != PASS_STATUS
            or result.get("domain_counts", {}).get("eligible_pairs") != 31469
            or selection.get("precision", 0.0) < 0.98
            or selection.get("false_accept_rate", 1.0) > 0.01
            or selection.get("recall", 0.0) < 0.25
            or result.get("optimizer_steps") != 0
            or result.get("backbone_optimizer_steps") != 0
            or result.get("c09_worlds_read") != 0
            or result.get("strict_test_worlds_read") != 0
            or result.get("mtare_worlds_read") != 0
        ):
            raise RuntimeError("distance-aware ensemble calibration did not pass its frozen contract")
        write_json(run_dir / "metrics/calibration_summary.json", {
            **result, "peak_host_rss_kib": peak_rss,
            "summary_sha256": _sha256(output / "summary.json"),
            "outputs_sha256": _sha256(output / "ensemble_selection_outputs.npz"),
        })
        if time.monotonic() - started > 600:
            raise RuntimeError("ensemble calibration duration exceeded")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        duration = time.monotonic() - started
        write_json(run_dir / "metrics/summary.json", {
            "schema_version": "gse_distance_aware_ensemble_calibration_outer_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS_STATUS,
            "error": error,
            "duration_seconds": duration,
            "calibration": result,
            "source_before": source_before,
            "source_after": source_after,
            "optimizer_steps": 0,
            "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
            "overall_status": overall,
        })
        _seal(run_dir)
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
