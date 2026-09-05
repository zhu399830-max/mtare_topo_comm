#!/usr/bin/env python3
"""Run, qualify and seal the one immutable three-seed causal episode training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_gse_distance_aware_ensemble_calibration_v1 import _peak_rss, _seal, _sha256


RUN_ID = "gate3_20260827_gse_causal_episode_training_v1_seed0"
RUN_ID_V1R = "gate3_20260827_gse_causal_episode_training_v1r_seed0"
PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_TRAINING_V1"
PASS_STATUS_V1R = "PASS_GSE_CAUSAL_EPISODE_TRAINING_V1R"
FAIL_STATUS = "FAIL_GSE_CAUSAL_EPISODE_TRAINING_V1"
FAIL_STATUS_V1R = "FAIL_GSE_CAUSAL_EPISODE_TRAINING_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_TRAINING_V1"
CARD_STATUS_V1R = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_TRAINING_V1R"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/"
    "envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
SUPERVISION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
OLD_DIRECTIONAL = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
BUILDER = PROJECT_ROOT / "tools/v3/build_gse_causal_episode_spatial_cache_v1.py"
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_causal_episode_detector_v1.py"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_causal_episode_ensemble_v1.py"
PER_COMMAND_TIMEOUT_SECONDS = 14_400
TOTAL_TIMEOUT_SECONDS = 43_200
FINAL_DISK_LIMIT_BYTES = 2 * 1024**3


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, int | None]:
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
            text=True, stdout=stream, stderr=subprocess.STDOUT,
            timeout=PER_COMMAND_TIMEOUT_SECONDS, check=False,
        )
    return int(completed.returncode), _peak_rss(log_path)


def _verify_frozen(spec: dict) -> dict:
    observed = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha256(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
        observed[relative] = actual
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name not in (RUN_ID, RUN_ID_V1R) or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("causal episode training may execute only once")
    revised = run_dir.name == RUN_ID_V1R
    run_id = RUN_ID_V1R if revised else RUN_ID
    pass_status = PASS_STATUS_V1R if revised else PASS_STATUS
    fail_status = FAIL_STATUS_V1R if revised else FAIL_STATUS
    card_status = CARD_STATUS_V1R if revised else CARD_STATUS
    started = time.monotonic()
    overall = fail_status
    error = None
    subprocess_records = []
    source_before = {}
    source_after = {}
    ensemble = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("causal episode formal scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing user authorization is not bound to this operation")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != card_status or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("causal episode training Data Card mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen causal episode tool drift: {name}")
        source_before = _verify_frozen(spec)
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected_environment = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7", "numcodecs": "0.15.1",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected_environment:
            raise RuntimeError(f"causal episode environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free before sequential cache training")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "deterministic_algorithms": True, "sequential_temporary_cache": True,
        })
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING",
            "note": "Three seeds sequentially cache C01-C08 frozen spatial embeddings and train a past-only episode detector.",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        scratch_root = run_dir / "scratch"
        scratch_root.mkdir()
        models_root = run_dir / "artifacts/models"
        manifests_root = run_dir / "artifacts/cache_manifests"
        models_root.mkdir(parents=True)
        manifests_root.mkdir(parents=True)
        deletion_records = []
        for seed in (0, 1, 2):
            if time.monotonic() - started > TOTAL_TIMEOUT_SECONDS:
                raise RuntimeError("causal episode total wall-time limit exceeded")
            cache = scratch_root / f"seed{seed}_cache"
            checkpoint = TRAINING / f"artifacts/models/seed{seed}/best.pt"
            baseline = VERIFIER / f"artifacts/models/seed{seed}/frozen_observation_features.npy"
            build_command = [
                str(PYTHON), str(BUILDER),
                "--dataset-run", str(DATASET),
                "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
                "--transition-timing", str(SUPERVISION / "artifacts/transition_timing.csv"),
                "--checkpoint", str(checkpoint),
                "--baseline-features", str(baseline),
                "--population-index", str(VERIFIER / "artifacts/pair_cache/pairs.npz"),
                "--output-dir", str(cache), "--seed", str(seed),
                "--batch-size", str(spec["hyperparameters"]["cache_batch_size"]),
            ]
            code, rss = _run(build_command, run_dir / f"logs/{seed * 2:02d}_seed{seed}_cache.log", env)
            subprocess_records.append({"stage": "cache", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} spatial cache build failed")
            cache_manifest = load_json(cache / "manifest.json")
            if (
                cache_manifest.get("unique_frames") != 252430
                or cache_manifest.get("causal_observations") != 188126
                or cache_manifest.get("structural_episodes") != 5306
                or cache_manifest.get("boundary_supervision") != {
                    "timing_rows": 1031, "valid_backprojection_rows": 1030,
                    "pre_boundary_event_rows": 1,
                }
                or cache_manifest.get("optimizer_steps") != 0
                or cache_manifest.get("c09_worlds_read") != 0
                or cache_manifest.get("strict_test_worlds_read") != 0
                or cache_manifest.get("mtare_worlds_read") != 0
                or int(cache_manifest.get("cache_bytes", 0)) > 3 * 1024**3
            ):
                raise RuntimeError(f"seed{seed} cache evidence contract drift")
            shutil.copy2(cache / "manifest.json", manifests_root / f"seed{seed}_manifest.json")
            train_command = [
                str(PYTHON), str(TRAINER),
                "--cache-dir", str(cache), "--baseline-features", str(baseline),
                "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
                "--output-dir", str(models_root / f"seed{seed}"), "--seed", str(seed),
                "--epochs", str(spec["hyperparameters"]["epochs"]),
                "--batch-size", str(spec["hyperparameters"]["training_batch_size"]),
                "--evaluation-batch-size", str(spec["hyperparameters"]["evaluation_batch_size"]),
                "--learning-rate", str(spec["hyperparameters"]["learning_rate"]),
                "--weight-decay", str(spec["hyperparameters"]["weight_decay"]),
            ]
            code, rss = _run(train_command, run_dir / f"logs/{seed * 2 + 1:02d}_seed{seed}_training.log", env)
            subprocess_records.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} detector training failed")
            seed_summary = load_json(models_root / f"seed{seed}/summary.json")
            if (
                seed_summary.get("optimizer_steps") != 13332
                or seed_summary.get("backbone_optimizer_steps") != 0
                or seed_summary.get("fit_observations") != 142184
                or seed_summary.get("selection_observations") != 45942
                or seed_summary.get("c09_worlds_read") != 0
                or seed_summary.get("strict_test_worlds_read") != 0
                or seed_summary.get("mtare_worlds_read") != 0
            ):
                raise RuntimeError(f"seed{seed} training evidence contract drift")
            cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            shutil.rmtree(cache)
            deletion_records.append({
                "seed": seed, "deleted_regenerable_cache_bytes": cache_bytes,
                "retained_manifest": str((manifests_root / f"seed{seed}_manifest.json").relative_to(run_dir)),
            })
        scratch_root.rmdir()
        write_json(run_dir / "artifacts/cache_deletion_audit.json", {
            "schema_version": "gse_causal_episode_cache_deletion_audit_v1",
            "records": deletion_records,
            "reason": "Per-seed spatial caches are deterministic regenerable scratch; digests and provenance are retained.",
        })
        evaluation_dir = run_dir / "metrics/ensemble"
        evaluate_command = [
            str(PYTHON), str(EVALUATOR), "--training-root", str(models_root),
            "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"),
            "--old-directional-root", str(OLD_DIRECTIONAL / "artifacts/training"),
            "--output-dir", str(evaluation_dir),
        ]
        code, rss = _run(evaluate_command, run_dir / "logs/06_ensemble_evaluation.log", env)
        subprocess_records.append({"stage": "ensemble", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("ensemble evaluator failed as a program")
        ensemble = load_json(evaluation_dir / "summary.json")
        if (
            ensemble.get("overall_status") not in (
                "PASS_GSE_CAUSAL_EPISODE_ENSEMBLE_V1", "FAIL_GSE_CAUSAL_EPISODE_ENSEMBLE_V1"
            )
            or (code == 0) != bool(ensemble.get("scientific_pass"))
            or ensemble.get("selection_observations") != 45942
            or any(ensemble.get(key) != 0 for key in (
                "c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read"
            ))
        ):
            raise RuntimeError("ensemble evidence contract drift")
        source_after = _verify_frozen(spec)
        if source_before != source_after:
            raise RuntimeError("frozen source changed during causal episode training")
        output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if output_bytes > FINAL_DISK_LIMIT_BYTES:
            raise RuntimeError("retained causal episode evidence exceeds 2 GiB")
        overall = pass_status if ensemble["scientific_pass"] else fail_status
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    summary = {
        "schema_version": "gse_causal_episode_training_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == pass_status,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "subprocesses": subprocess_records,
        "ensemble": ensemble,
        "source_unchanged": bool(source_before and source_before == source_after),
        "optimizer_steps": 39996 if len([r for r in subprocess_records if r["stage"] == "training" and r["returncode"] == 0]) == 3 else None,
        "backbone_optimizer_steps": 0,
        "model_inference_frames": 757290 if len([r for r in subprocess_records if r["stage"] == "cache" and r["returncode"] == 0]) == 3 else None,
        "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if overall == pass_status else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == pass_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
