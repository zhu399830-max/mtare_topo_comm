#!/usr/bin/env python3
"""Execute and seal one immutable action-conditioned state feasibility proof."""

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


RUN_ID = "gate3_20260827_gse_action_conditioned_state_feasibility_v1_seed0"
PASS_STATUS = "PASS_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1"
FAIL_STATUS = "FAIL_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/"
    "envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_action_conditioned_state_feasibility_v1.py"
TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
VERIFIER = PROJECT_ROOT / (
    "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
)
TIME_LIMIT_SECONDS = 900
DISK_LIMIT_BYTES = 256 * 1024 * 1024


def _sources() -> dict:
    return {
        "corrected_teacher": verify_complete_run_seal(
            PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"
        ),
        "frozen_observation_components": verify_failed_component_run_seal(
            PROJECT_ROOT, VERIFIER, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("action-conditioned state proof may execute only once")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    returncode = None
    peak_rss = None
    result = {}
    before = {}
    after = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("action-conditioned state proof scope drift")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing user authorization is not bound to this proof")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("action-conditioned state Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen action-conditioned tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen action-conditioned input drift: {relative}")
            before[relative] = actual
        source_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "matplotlib": "3.10.0"}:
            raise RuntimeError(f"action-conditioned sidecar drift: {environment}")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment, "cpu_only": True,
        })
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "C01-C08-only frozen-feature state-segmentation feasibility proof; zero inference/optimization.",
        })
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR),
            "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--pair-cache", str(VERIFIER / "artifacts/pair_cache/pairs.npz"),
        ]
        for seed in (0, 1, 2):
            command.extend([
                "--feature", str(VERIFIER / f"artifacts/models/seed{seed}/frozen_observation_features.npy")
            ])
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        log = run_dir / "logs/00_action_conditioned_state.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True,
                stdout=stream, stderr=subprocess.STDOUT,
                timeout=TIME_LIMIT_SECONDS, check=False,
            )
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result_path = run_dir / "metrics/action_conditioned_state_feasibility.json"
        result = load_json(result_path) if result_path.is_file() else {}
        source_after = _sources()
        for relative, expected in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"source changed during action-conditioned proof: {relative}")
            after[relative] = actual
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        required = [
            run_dir / "artifacts/method_metrics.csv",
            run_dir / "artifacts/triggers.csv",
            run_dir / "previews/gse_action_conditioned_state_feasibility.png",
            run_dir / "previews/gse_action_conditioned_state_feasibility.pdf",
            run_dir / "previews/gse_action_conditioned_state_feasibility.svg",
            run_dir / "previews/gse_action_conditioned_state_feasibility_source.json",
        ]
        if (
            source_before != source_after
            or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (returncode == 0) != bool(result.get("scientific_pass"))
            or result.get("fit_worlds") != 60
            or result.get("fit_observations") != 142184
            or result.get("selection_worlds") != 20
            or result.get("selection_observations") != 45942
            or result.get("fit_episodes") != 3956
            or result.get("selection_episodes") != 1350
            or any(result.get(name) != 0 for name in (
                "optimizer_steps", "model_inference_frames", "c09_worlds_read",
                "strict_test_worlds_read", "mtare_worlds_read",
            ))
            or peak_rss is None
            or peak_rss > 4 * 1024**2
            or output_bytes > DISK_LIMIT_BYTES
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("action-conditioned state evidence contract drift")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_action_conditioned_state_feasibility_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS_STATUS,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "peak_host_rss_kib": peak_rss,
        "source_unchanged": bool(before and before == after),
        "proof": result,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
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
