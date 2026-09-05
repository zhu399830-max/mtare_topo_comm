#!/usr/bin/env python3
"""Execute and seal the immutable C01-C08 supervision-unit audit."""

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


RUN_ID = "gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
PASS_STATUS = "PASS_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_causal_event_supervision_audit_v1.py"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
PROOF = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
RISK = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_geometry_risk_conflict_audit_v1_seed0"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_geometry_conditioned_identity_risk_capacity_v1_seed0"


def _sources() -> dict:
    return {
        "corrected_teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"),
        "change_point_proof": verify_complete_run_seal(PROJECT_ROOT, PROOF, "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1"),
        "risk_conflict": verify_complete_run_seal(PROJECT_ROOT, RISK, "PASS_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1"),
        "identity_risk_capacity": verify_failed_component_run_seal(PROJECT_ROOT, CAPACITY, "FAIL_GSE_GEOMETRY_CONDITIONED_IDENTITY_RISK_CAPACITY_V1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID:
        raise RuntimeError("supervision audit run identity mismatch")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("supervision audit may execute only once")

    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    returncode = None
    peak_rss = None
    result: dict = {}
    before: dict = {}
    after: dict = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("seed") != 0:
            raise RuntimeError("supervision audit scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing authorization is not bound")
        card = load_json(PROJECT_ROOT / spec["audit_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("supervision audit Data Card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen supervision tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen supervision input drift: {relative}")
        before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))",
        ], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "matplotlib": "3.10.0"}:
            raise RuntimeError(f"supervision sidecar drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "cpu_only": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "C01-C08-only frame-versus-episode supervision audit.",
        })
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR),
            "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--proof-points", str(PROOF / "artifacts/bidirectional_change_points.jsonl"),
            "--proof-labels", str(PROOF / "artifacts/causal_change_point_labels.jsonl"),
            "--risk-summary", str(RISK / "metrics/risk_conflict_summary.json"),
            "--capacity-summary", str(CAPACITY / "metrics/capacity_summary.json"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"] = "2"
        log = run_dir / "logs/00_supervision_audit.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        returncode = int(completed.returncode)
        peak_rss = _peak_rss(log)
        result_path = run_dir / "metrics/supervision_audit.json"
        result = load_json(result_path) if result_path.is_file() else {}
        after = _sources()
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        required = [
            run_dir / "artifacts/event_episode_inventory.csv",
            run_dir / "artifacts/transition_timing.csv",
            result_path,
            run_dir / "previews/gse_causal_event_supervision.png",
            run_dir / "previews/gse_causal_event_supervision.pdf",
            run_dir / "previews/gse_causal_event_supervision.svg",
            run_dir / "previews/gse_causal_event_supervision_source.json",
            run_dir / "previews/gse_causal_event_supervision_provenance.json",
        ]
        if (
            before != after or returncode != 0 or result.get("overall_status") != PASS_STATUS
            or result.get("worlds") != 80 or result.get("causal_observations") != 188126
            or result.get("structural_identities") != 1534 or result.get("structural_episodes") != 5306
            or result.get("transition_alignment", {}).get("final_transition_labels") != 1031
            or result.get("transition_alignment", {}).get("directional_change_episodes") != 152
            or any(result.get(name) != 0 for name in ("optimizer_steps", "model_inference_frames", "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"))
            or peak_rss is None or peak_rss > 4 * 1024**2 or output_bytes > 128 * 1024**2
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("supervision audit violated its execution/evidence contract")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_causal_event_supervision_audit_outer_v1",
        "overall_status": overall,
        "scientific_qualification": False,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "peak_host_rss_kib": peak_rss,
        "supervision_audit": result,
        "source_before": before,
        "source_after": after,
        "source_unchanged": bool(before and before == after),
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
