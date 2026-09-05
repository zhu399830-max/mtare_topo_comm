#!/usr/bin/env python3
"""Execute and seal one immutable twelve-scan reference proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_causal_episode_reference_proof_v1_seed0"
RUN_ID_V1R = "gate3_20260827_gse_causal_episode_reference_proof_v1r_seed0"
PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
PASS_STATUS_V1R = "PASS_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1R"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
FAIL_STATUS_V1R = "FAIL_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
CARD_STATUS_V1R = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1R"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_causal_episode_reference_proof_v1.py"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
AUDIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
CHANGE_PROOF = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"


def _sources() -> dict:
    return {
        "dataset": verify_complete_run_seal(PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"),
        "teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"),
        "supervision_audit": verify_complete_run_seal(PROJECT_ROOT, AUDIT, "PASS_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1"),
        "change_point_proof": verify_complete_run_seal(PROJECT_ROOT, CHANGE_PROOF, "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    revised = run_dir.name == RUN_ID_V1R
    expected_run_id = RUN_ID_V1R if revised else RUN_ID
    expected_pass = PASS_STATUS_V1R if revised else PASS_STATUS
    expected_fail = FAIL_STATUS_V1R if revised else FAIL_STATUS
    expected_card = CARD_STATUS_V1R if revised else CARD_STATUS
    if run_dir.name != expected_run_id or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("causal episode reference proof identity/state mismatch")
    started = time.monotonic()
    overall = expected_fail
    error = None
    returncode = None
    peak_rss = None
    result: dict = {}
    before: dict = {}
    after: dict = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("causal episode proof scope mismatch")
        card = load_json(PROJECT_ROOT / spec["audit_card"])
        if card.get("status") != expected_card or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("causal episode proof card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen episode proof tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen episode proof input drift: {relative}")
        before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,sys,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'zarr':zarr.__version__},sort_keys=True))",
        ], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "zarr": "2.18.7"}:
            raise RuntimeError(f"causal episode proof sidecar drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "cpu_only": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": expected_run_id, "state": "RUNNING", "note": "Read-only C01-C08 twelve-scan reference proof."})
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR),
            "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--dataset-run", str(DATASET),
            "--change-proof-summary", str(CHANGE_PROOF / "metrics/summary.json"),
            "--expected-status", expected_pass,
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        log = run_dir / "logs/00_reference_proof.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result_path = run_dir / "metrics/reference_proof.json"
        result = load_json(result_path) if result_path.is_file() else {}
        after = _sources()
        required = [result_path, run_dir / "artifacts/world_reference_summary.csv"]
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            before != after or returncode != 0 or result.get("overall_status") != expected_pass
            or result.get("worlds") != 80 or result.get("inventory_directed_traversals") != 16078
            or result.get("observed_directed_traversals") != 16076 or result.get("zero_observation_traversals") != 2
            or result.get("unique_frames") != 252430 or result.get("causal_observations") != 188126
            or result.get("structural_episodes") != 5306 or result.get("sensor_payload_bytes_written") != 0
            or result.get("new_rays") != 0
            or any(result.get(name) != 0 for name in ("optimizer_steps", "model_inference_frames", "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"))
            or peak_rss is None or peak_rss > 4 * 1024**2 or output_bytes > 64 * 1024**2
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("causal episode reference proof violated its contract")
        overall = expected_pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_causal_episode_reference_proof_outer_v1", "overall_status": overall,
        "scientific_qualification": False, "error": error, "duration_seconds": time.monotonic() - started,
        "returncode": returncode, "peak_host_rss_kib": peak_rss, "reference_proof": result,
        "source_before": before, "source_after": after, "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": expected_run_id,
        "state": "COMPLETED" if overall == expected_pass else "FAILED", "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == expected_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
