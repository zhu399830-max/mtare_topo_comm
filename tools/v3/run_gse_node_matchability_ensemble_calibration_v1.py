#!/usr/bin/env python3
"""Execute and seal one immutable node-matchability ensemble calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_failed_component_run_seal
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0"
PASS_STATUS = "PASS_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"
FAIL_STATUS = "FAIL_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_node_matchability_ensemble_v1.py"


def _source() -> dict:
    return verify_failed_component_run_seal(
        PROJECT_ROOT, SOURCE, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time node calibration identity mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    result = {}
    source_before = source_after = None
    try:
        if spec.get("gate") != 4 or spec.get("operation") != "threshold_calibration":
            raise RuntimeError("node calibration scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("node calibration is not authorized")
        if load_json(PROJECT_ROOT / spec["data_card"]).get("status") != CARD_STATUS:
            raise RuntimeError("node calibration Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen node-calibration tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen node-calibration input drift: {relative}")
        source_before = _source()
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"
        })
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__},sort_keys=True))",
        ], text=True))
        write_json(run_dir / "config/environment.json", {"versions": environment, "gpu_used": False})
        output = run_dir / "artifacts/calibration"
        log = run_dir / "logs/00_calibration.log"
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EVALUATOR),
            "--source-run", str(SOURCE), "--output-dir", str(output),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False,
            )
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        peak_rss = _peak_rss(log)
        source_after = _source()
        selection = result.get("selection") or {}
        if (
            completed.returncode != 0
            or source_before != source_after
            or peak_rss is None or peak_rss > 2 * 1024**2
            or result.get("overall_status") != PASS_STATUS
            or result.get("selection_observations") != 45942
            or selection.get("precision", 0.0) < 0.98
            or selection.get("false_accept_rate", 1.0) > 0.01
            or selection.get("recall", 0.0) < 0.25
            or len(selection.get("per_family", {})) != 10
            or any(result.get(name) != 0 for name in (
                "optimizer_steps", "backbone_optimizer_steps", "c09_worlds_read",
                "strict_test_worlds_read", "mtare_worlds_read",
            ))
        ):
            raise RuntimeError("node-matchability calibration failed its frozen contract")
        write_json(run_dir / "metrics/calibration_summary.json", {
            **result,
            "peak_host_rss_kib": peak_rss,
            "summary_sha256": _sha256(output / "summary.json"),
            "outputs_sha256": _sha256(output / "node_matchability_selection_outputs.npz"),
        })
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_node_matchability_ensemble_calibration_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS_STATUS,
        "error": error,
        "duration_seconds": time.monotonic() - started,
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
        "overall_status": overall, "error": error,
    })
    _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
