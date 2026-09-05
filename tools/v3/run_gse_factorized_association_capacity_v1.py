#!/usr/bin/env python3
"""Run and seal one immutable Factorized GSE association capacity proof."""

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


RUN_ID = "gate3_20260827_gse_factorized_association_capacity_v1_seed0"
PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_factorized_association_capacity_v1.py"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
MANIFEST = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0"
FEATURES = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"


def _sources() -> dict:
    return {
        "teacher": verify_complete_run_seal(
            PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"
        ),
        "association_teacher": verify_complete_run_seal(
            PROJECT_ROOT, MANIFEST, "PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1"
        ),
        "frozen_perception_components": verify_failed_component_run_seal(
            PROJECT_ROOT, FEATURES, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("factorized association capacity may execute only once")
    started = time.monotonic()
    overall, error, returncode, peak_rss = FAIL_STATUS, None, None, None
    result, before, after = {}, {}, {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("factorized association capacity scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("factorized association capacity Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen capacity tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen capacity input drift: {relative}")
            before[relative] = actual
        sources_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,torch,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "matplotlib": "3.10.0", "torch": "2.9.0+cu129"}
        if environment != expected:
            raise RuntimeError(f"factorized association capacity environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment, "cpu_only_training": True,
        })
        write_json(run_dir / "config/source_integrity_before.json", sources_before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "C01-C08 frozen-perception three-seed factorized association capacity proof.",
        })
        command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR), "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--manifest", str(MANIFEST / "artifacts/identity_balanced_association_manifest.jsonl"),
        ]
        for seed in range(3):
            base = FEATURES / f"artifacts/models/seed{seed}"
            command.extend([f"--observation{seed}", str(base / "frozen_observation_features.npy")])
            command.extend([f"--tokens{seed}", str(base / "frozen_exit_token_outputs.npz")])
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"] = "2", "2"
        log = run_dir / "logs/00_factorized_association_capacity.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream,
                stderr=subprocess.STDOUT, timeout=3600, check=False,
            )
        returncode = int(completed.returncode); peak_rss = _peak_rss(log)
        metric_path = run_dir / "metrics/factorized_association_capacity.json"
        result = load_json(metric_path) if metric_path.is_file() else {}
        sources_after = _sources()
        for relative, expected_sha in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected_sha:
                raise RuntimeError(f"source changed during capacity proof: {relative}")
            after[relative] = actual
        required = [
            run_dir / "previews/gse_factorized_association_capacity.png",
            run_dir / "previews/gse_factorized_association_capacity.pdf",
            run_dir / "previews/gse_factorized_association_capacity.svg",
            run_dir / "previews/gse_factorized_association_capacity_source.json",
        ]
        for seed in range(3):
            required.extend([
                run_dir / f"artifacts/models/seed{seed}/summary.json",
                run_dir / f"artifacts/models/seed{seed}/descriptor_only_selection_outputs.npz",
            ])
            for variant in ("full_route_conditioned", "no_route_geometry"):
                required.extend([
                    run_dir / f"artifacts/models/seed{seed}/{variant}/best.pt",
                    run_dir / f"artifacts/models/seed{seed}/{variant}/summary.json",
                    run_dir / f"artifacts/models/seed{seed}/{variant}/selection_outputs.npz",
                    run_dir / f"artifacts/models/seed{seed}/{variant}/selection_curve.jsonl",
                ])
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            sources_before != sources_after or returncode not in (0, 2)
            or result.get("overall_status") not in (PASS_STATUS, FAIL_STATUS)
            or (returncode == 0) != bool(result.get("scientific_pass"))
            or result.get("optimizer_steps", 0) <= 0 or result.get("backbone_optimizer_steps") != 0
            or any(result.get(name) != 0 for name in (
                "model_inference_frames", "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"
            ))
            or peak_rss is None or peak_rss > 4 * 1024**2 or output_bytes > 512 * 1024**2
            or not all(path.is_file() for path in required)
        ):
            raise RuntimeError("factorized association capacity evidence contract drift")
        overall = str(result["overall_status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_factorized_association_capacity_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "returncode": returncode, "peak_host_rss_kib": peak_rss,
        "source_unchanged": bool(before and before == after), "capacity": result,
        "optimizer_steps": int(result.get("optimizer_steps", 0)), "backbone_optimizer_steps": 0,
        "model_inference_frames": 0, "c09_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
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
