#!/usr/bin/env python3
"""Run immutable C09 validation for the C07--C08 maximin slope residual."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_gse_corrected_perception_validation_v1 import (
    CORRECTIVE,
    DATASET,
    ORIGINAL_C09,
    PYTHON,
    _run,
    _seal,
    _sha256,
    _sources,
)


RUN_ID = "gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
PASS_STATUS = "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"
FAIL_STATUS = "FAIL_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"
RSS_LIMIT_KIB = 8 * 1024**2
DISK_LIMIT_BYTES = 2 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time risk-calibrated perception run identity mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    stages: dict[str, object] = {}
    sources_before = None
    sources_after = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "threshold_calibration" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("risk-calibrated perception scope is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2":
            raise RuntimeError("risk-calibrated perception Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"risk-calibrated perception frozen tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"risk-calibrated perception frozen input drift: {relative}")
        if shutil.disk_usage(PROJECT_ROOT).free < 4 * 1024**3:
            raise RuntimeError("less than 4 GiB free before risk-calibrated C09 validation")
        sources_before = _sources()
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        write_json(
            run_dir / "config/environment.json",
            json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"], text=True)),
        )
        raw_dir = run_dir / "artifacts/raw_full_residual_c09"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_slope_corrective_c09_v1.py"), "--dataset-run", str(DATASET), "--corrective-run", str(CORRECTIVE), "--output-dir", str(raw_dir)],
            run_dir / "logs/01_raw_full_residual_c09.log", env, 3600,
        )
        stages["raw_full_residual"] = load_json(raw_dir / "summary.json") if (raw_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["raw_full_residual"].get("overall_status") != "PASS_GSE_SLOPE_CORRECTIVE_C09_EVALUATION_V1":
            raise RuntimeError("raw full-residual C09 evaluation failed technical evidence")
        calibrated_dir = run_dir / "artifacts/risk_calibrated_slope_c09"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/apply_gse_slope_risk_calibration_v2.py"), "--corrective-run", str(CORRECTIVE), "--raw-c09-dir", str(raw_dir), "--output-dir", str(calibrated_dir)],
            run_dir / "logs/02_risk_calibration.log", env, 600,
        )
        stages["risk_calibrated_slope"] = load_json(calibrated_dir / "summary.json") if (calibrated_dir / "summary.json").is_file() else {"exit_code": code}
        if code != 0 or stages["risk_calibrated_slope"].get("overall_status") != "PASS_GSE_SLOPE_RISK_CALIBRATED_C09_EVALUATION_V2":
            raise RuntimeError("risk-calibrated C09 slope evaluation failed technical evidence")
        gate_path = run_dir / "metrics/corrected_perception_gate.json"
        code = _run(
            [str(PYTHON), str(PROJECT_ROOT / "tools/v3/summarize_gse_corrected_perception_gate_v1.py"), "--original-gate", str(ORIGINAL_C09 / "metrics/perception_gate.json"), "--corrective-summary", str(calibrated_dir / "summary.json"), "--output", str(gate_path)],
            run_dir / "logs/03_complete_perception_gate.log", env, 600,
        )
        stages["complete_perception_gate"] = load_json(gate_path) if gate_path.is_file() else {"exit_code": code}
        if code != 0 or stages["complete_perception_gate"].get("passed") is not True:
            raise RuntimeError("risk-calibrated GSE perception did not pass the unchanged gates")
        sources_after = _sources()
        if sources_before != sources_after:
            raise RuntimeError("risk-calibrated perception source evidence changed during validation")
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if result_bytes > DISK_LIMIT_BYTES or rss > RSS_LIMIT_KIB:
            raise RuntimeError("risk-calibrated perception resource contract failed")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_risk_calibrated_perception_validation_run_v2",
            "overall_status": overall,
            "error": error,
            "stages": stages,
            "sources_before": sources_before,
            "sources_after": sources_after,
            "selection_worlds_read": 20,
            "selection_sequences": 45_942,
            "validation_worlds_read": 10,
            "validation_sequences": 24_462,
            "c10_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "optimizer_steps": 0,
            "model_updates": 0,
            "result_bytes_before_seal": result_bytes,
            "maximum_child_rss_kib": rss,
            "duration_seconds": time.monotonic() - started,
        },
    )
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    sealed = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "sealed_files": sealed}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
