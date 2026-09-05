#!/usr/bin/env python3
"""Build unified observations, rerun capacity proof once, and seal the result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R"
INNER_PASS = "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
BUILDER = PROJECT_ROOT / "tools/v3/build_gse_unified_observation_features_v1.py"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_factorized_association_capacity_v1.py"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
MANIFEST = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0"
FEATURES = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
RISK = PROJECT_ROOT / "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
OLD_CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1_seed0"


def _sources() -> dict[str, object]:
    return {
        "teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"),
        "association_teacher": verify_complete_run_seal(PROJECT_ROOT, MANIFEST, "PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1"),
        "frozen_perception_components": verify_failed_component_run_seal(PROJECT_ROOT, FEATURES, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"),
        "slope_corrective": verify_complete_run_seal(PROJECT_ROOT, CORRECTIVE, "PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R"),
        "risk_calibration": verify_complete_run_seal(PROJECT_ROOT, RISK, "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"),
        "old_capacity_component_proof": verify_complete_run_seal(PROJECT_ROOT, OLD_CAPACITY, INNER_PASS),
    }


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream,
            stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    return int(completed.returncode), _peak_rss(log)


def _unchanged_ablations(run_dir: Path) -> bool:
    for seed in range(3):
        current = run_dir / f"artifacts/models/seed{seed}"
        old = OLD_CAPACITY / f"artifacts/models/seed{seed}"
        for relative in (
            "descriptor_only_selection_outputs.npz",
            "no_route_geometry/selection_outputs.npz",
        ):
            with np.load(current / relative, allow_pickle=False) as left, np.load(old / relative, allow_pickle=False) as right:
                if left.files != right.files or any(not np.array_equal(left[key], right[key]) for key in left.files):
                    return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("unified association capacity may execute only once")
    started = time.monotonic()
    overall, error = FAIL_STATUS, None
    builder_code = capacity_code = None
    builder_rss = capacity_rss = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    exact_ablations = False
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("unified association capacity scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != CARD_STATUS or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("unified association capacity Data Card drift")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen unified capacity tool drift: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"frozen unified capacity input drift: {relative}")
            before[relative] = actual
        sources_before = _sources()
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,matplotlib,numpy,torch,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "matplotlib": "3.10.0", "torch": "2.9.0+cu129", "cuda": "12.9", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected:
            raise RuntimeError(f"unified association capacity environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "exact_cuda_slope_replay": True, "cpu_only_association_training": True})
        write_json(run_dir / "config/source_integrity_before.json", sources_before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING", "note": "C01-C08 final-slope unified association capacity corrective."})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
        unified = run_dir / "artifacts/unified_observation"
        builder_command = [
            "/usr/bin/time", "-v", str(PYTHON), str(BUILDER),
            "--cache-dir", str(CORRECTIVE / "artifacts/slope_corrective_cache"),
            "--corrective-run", str(CORRECTIVE),
            "--risk-calibration", str(RISK / "artifacts/risk_calibrated_slope_c09/risk_calibration.json"),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--output-dir", str(unified),
        ]
        for seed in range(3):
            builder_command.extend([f"--observation{seed}", str(FEATURES / f"artifacts/models/seed{seed}/frozen_observation_features.npy")])
        capacity_command = [
            "/usr/bin/time", "-v", str(PYTHON), str(EXECUTOR), "--run-dir", str(run_dir),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--manifest", str(MANIFEST / "artifacts/identity_balanced_association_manifest.jsonl"),
        ]
        for seed in range(3):
            capacity_command.extend([f"--observation{seed}", str(unified / f"seed{seed}_unified_observation_features.npy")])
            capacity_command.extend([f"--tokens{seed}", str(FEATURES / f"artifacts/models/seed{seed}/frozen_exit_token_outputs.npz")])
        (run_dir / "config/command.txt").write_text(" ".join(builder_command) + "\n" + " ".join(capacity_command) + "\n", encoding="utf-8")
        builder_code, builder_rss = _run(builder_command, run_dir / "logs/00_unified_observation.log", env, 1800)
        integration = load_json(unified / "summary.json") if (unified / "summary.json").is_file() else {}
        if (
            builder_code != 0 or builder_rss is None or builder_rss > 4 * 1024**2
            or integration.get("overall_status") != "PASS_GSE_UNIFIED_OBSERVATION_FEATURES_V1"
            or integration.get("observations_per_seed") != 188126
            or integration.get("slope_corrective_inference_sequences") != 564378
            or integration.get("gse_backbone_inference_frames") != 0
            or integration.get("peak_gpu_memory_bytes", 9 * 1024**3) > 8 * 1024**3
            or not all(row.get("selection_gpu_replay_byte_exact") and row.get("audit", {}).get("changed_columns") == [10] and row.get("audit", {}).get("unchanged_columns_byte_exact") for row in integration.get("seeds", []))
        ):
            raise RuntimeError("unified observation integration evidence drift")
        capacity_code, capacity_rss = _run(capacity_command, run_dir / "logs/01_factorized_association_capacity.log", env, 3600)
        metric_path = run_dir / "metrics/factorized_association_capacity.json"
        result = load_json(metric_path) if metric_path.is_file() else {}
        exact_ablations = _unchanged_ablations(run_dir)
        sources_after = _sources()
        for relative, expected_sha in before.items():
            actual = _sha256(PROJECT_ROOT / relative)
            if actual != expected_sha:
                raise RuntimeError(f"source changed during unified capacity proof: {relative}")
            after[relative] = actual
        required = [
            unified / "summary.json",
            run_dir / "previews/gse_factorized_association_capacity.png",
            run_dir / "previews/gse_factorized_association_capacity.pdf",
            run_dir / "previews/gse_factorized_association_capacity.svg",
            run_dir / "previews/gse_factorized_association_capacity_source.json",
        ]
        for seed in range(3):
            required.extend([unified / f"seed{seed}_unified_observation_features.npy", unified / f"seed{seed}_slope_outputs.npz", run_dir / f"artifacts/models/seed{seed}/summary.json"])
            for variant in ("full_route_conditioned", "no_route_geometry"):
                required.extend([run_dir / f"artifacts/models/seed{seed}/{variant}/best.pt", run_dir / f"artifacts/models/seed{seed}/{variant}/summary.json", run_dir / f"artifacts/models/seed{seed}/{variant}/selection_outputs.npz"])
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            sources_before != sources_after or builder_code != 0 or capacity_code != 0
            or result.get("overall_status") != INNER_PASS or result.get("scientific_pass") is not True
            or not exact_ablations or capacity_rss is None or capacity_rss > 4 * 1024**2
            or output_bytes > 1024**3 or not all(path.is_file() for path in required)
            or any(result.get(name) != 0 for name in ("backbone_optimizer_steps", "model_inference_frames", "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("unified factorized association capacity evidence contract drift")
        overall = PASS_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_factorized_association_capacity_outer_v1r",
        "overall_status": overall, "scientific_pass": overall == PASS_STATUS,
        "error": error, "duration_seconds": time.monotonic() - started,
        "builder_returncode": builder_code, "capacity_returncode": capacity_code,
        "builder_peak_host_rss_kib": builder_rss, "capacity_peak_host_rss_kib": capacity_rss,
        "source_unchanged": bool(before and before == after),
        "non_route_ablations_byte_exact_to_v1": exact_ablations,
        "capacity": result,
        "slope_corrective_inference_sequences": 564378 if builder_code == 0 else 0,
        "optimizer_steps": int(result.get("optimizer_steps", 0)), "backbone_optimizer_steps": 0,
        "gse_backbone_inference_frames": 0, "c09_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
