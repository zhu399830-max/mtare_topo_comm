#!/usr/bin/env python3
"""Execute and seal one immutable causal episode class-mass audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_causal_episode_class_mass_audit_v1_seed0"
PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_CLASS_MASS_AUDIT_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EPISODE_CLASS_MASS_AUDIT_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_CLASS_MASS_AUDIT_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_causal_episode_class_mass_audit_v1.py"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_episode_training_v1r_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
RISK = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("class-mass audit may execute only once")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    returncode = None
    peak_rss = None
    result = {}
    before = {}
    after = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("class-mass audit scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("class-mass audit Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen audit tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen audit input drift: {relative}")
            before[relative] = actual
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "cpu_only": True,
            "versions": json.loads(subprocess.check_output([
                str(PYTHON), "-c", "import json,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__},sort_keys=True))"
            ], text=True)),
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Read-only analytic class-loss and logit-gradient mass audit; zero inference/optimization.",
        })
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR),
            "--run-dir", str(run_dir), "--training-run", str(TRAINING),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--pair-cache", str(VERIFIER / "artifacts/pair_cache/pairs.npz"),
            "--risk-audit-summary", str(RISK / "metrics/risk_conflict_summary.json"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        log = run_dir / "logs/00_class_mass_audit.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result = load_json(run_dir / "metrics/class_mass_audit.json")
        for relative, expected in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"source changed during audit: {relative}")
            after[relative] = actual
        if (
            returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (returncode == 0) != bool(result.get("scientific_pass"))
            or result.get("optimizer_steps") != 0
            or result.get("model_inference_frames") != 0
            or any(result.get(key) != 0 for key in ("c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"))
            or peak_rss is None or peak_rss > 4 * 1024**2
        ):
            raise RuntimeError("class-mass audit evidence contract drift")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_causal_episode_class_mass_audit_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "returncode": returncode, "peak_host_rss_kib": peak_rss,
        "source_unchanged": bool(before and before == after), "audit": result,
        "optimizer_steps": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
