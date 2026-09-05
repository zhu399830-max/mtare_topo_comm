#!/usr/bin/env python3
"""Execute and seal one immutable three-seed rare-event corrective training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260826_gse_rare_event_corrective_training_v1_seed0"
PASS_STATUS = "PASS_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
VERIFIER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_rare_event_corrective_v1.py"


def _sources() -> dict:
    return {
        "verifier": verify_failed_component_run_seal(
            PROJECT_ROOT, VERIFIER, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
        ),
        "dataset": verify_complete_run_seal(
            PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time rare-event corrective run identity mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    result = {}
    before = after = None
    peak_rss = None
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training":
            raise RuntimeError("rare-event corrective scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("rare-event corrective is not authorized")
        if load_json(PROJECT_ROOT / spec["data_card"]).get("status") != CARD_STATUS:
            raise RuntimeError("rare-event corrective Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen corrective tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen corrective input drift: {relative}")
        before = _sources()
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"
        })
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()},sort_keys=True))",
        ], text=True))
        if environment.get("cuda_available") is not True:
            raise RuntimeError("rare-event corrective requires the frozen CUDA sidecar")
        write_json(run_dir / "config/environment.json", {"versions": environment, "device": "cuda:0"})
        output = run_dir / "artifacts/training"
        log = run_dir / "logs/00_training.log"
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(TRAINER),
            "--verifier-run", str(VERIFIER),
            "--dataset-run", str(DATASET),
            "--output-dir", str(output),
            "--device", "cuda:0",
            "--epochs", "40",
            "--batch-size", "512",
            "--learning-rate", "0.001",
            "--weight-decay", "0.0001",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT, timeout=7200, check=False,
            )
        returncode = int(completed.returncode)
        result = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        peak_rss = _peak_rss(log)
        after = _sources()
        if (
            before != after
            or peak_rss is None or peak_rss > 4 * 1024**2
            or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or result.get("fit_worlds") != 60
            or result.get("fit_observations") != 142184
            or result.get("selection_worlds") != 20
            or result.get("selection_observations") != 45942
            or set(result.get("seeds", {})) != {"0", "1", "2"}
            or int(result.get("optimizer_steps", 0)) <= 0
            or any(result.get(name) != 0 for name in (
                "backbone_optimizer_steps", "c09_worlds_read",
                "strict_test_worlds_read", "mtare_worlds_read",
            ))
            or (returncode == 0) != (result.get("overall_status") == PASS_STATUS)
        ):
            raise RuntimeError("rare-event corrective violated its execution/evidence contract")
        write_json(run_dir / "metrics/training_summary.json", {
            **result,
            "peak_host_rss_kib": peak_rss,
            "summary_sha256": _sha256(output / "summary.json"),
            "ensemble_outputs_sha256": _sha256(output / "ensemble_selection_outputs.npz"),
        })
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_rare_event_corrective_training_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS_STATUS,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "peak_host_rss_kib": peak_rss,
        "training": result,
        "source_before": before,
        "source_after": after,
        "optimizer_steps": int(result.get("optimizer_steps", 0)),
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
