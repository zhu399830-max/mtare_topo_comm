#!/usr/bin/env python3
"""Build the train-only cache, train three corrective seeds once, and seal it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
PASS_STATUS = "PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R"
FAIL_STATUS = "FAIL_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
CACHE_BUILDER = PROJECT_ROOT / "tools/v3/build_gse_slope_corrective_cache_v1.py"
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_slope_corrective_v1.py"
DATASET = (
    PROJECT_ROOT
    / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
)
DATASET_STATUS = "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
HOST_RSS_LIMIT_KIB = 8 * 1024**2
GPU_MEMORY_LIMIT_BYTES = 2 * 1024**3
DISK_LIMIT_BYTES = 2 * 1024**3
TOTAL_TIME_LIMIT_SECONDS = 7_200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_dataset() -> dict:
    state = load_json(DATASET / "RUN_STATE.json")
    summary = load_json(DATASET / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != DATASET_STATUS
        or summary.get("overall_status") != DATASET_STATUS
        or summary.get("split_totals", {}).get("train", {}).get("sequences") != 188126
        or summary.get("split_totals", {}).get("validation", {}).get("sequences") != 24462
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("frozen GSE dataset is not the required sealed PASS")
    seal = DATASET / "artifacts/evidence_sha256.txt"
    checked = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source dataset seal mismatch: {relative}")
        checked += 1
    if checked != 33083:
        raise RuntimeError("source dataset seal entry count drift")
    return {
        "run": str(DATASET.relative_to(PROJECT_ROOT)),
        "seal_sha256": _sha256(seal),
        "verified_seal_entries": checked,
        "summary_sha256": _sha256(DATASET / "metrics/summary.json"),
    }


def _environment() -> dict:
    raw = subprocess.check_output(
        [
            str(PYTHON),
            "-c",
            "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory},sort_keys=True))",
        ],
        text=True,
    )
    observed = json.loads(raw)
    expected = {
        "python": "3.13.5",
        "numpy": "2.1.3",
        "torch": "2.9.0+cu129",
        "cuda": "12.9",
        "cudnn": 91002,
        "zarr": "2.18.7",
        "numcodecs": "0.15.1",
        "gpu": "NVIDIA GeForce RTX 5090 D",
        "gpu_total_bytes": 33659879424,
    }
    if observed != expected:
        raise RuntimeError(f"frozen slope corrective environment drift: {observed}")
    return observed


def _stream(argv: list[str], log_path: Path, env: dict[str, str], timeout: int) -> int:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if time.monotonic() - started > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return 124
            return process.wait()
        finally:
            log.write(f"\nrunner_observed_seconds={time.monotonic() - started:.6f}\n")


def _peak_rss(log_path: Path) -> int | None:
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        log_path.read_text(encoding="utf-8"),
    )
    return int(match.group(1)) if match else None


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time slope corrective run state mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error: str | None = None
    source_before = None
    source_after = None
    seed_summaries: list[dict] = []
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("formal slope corrective scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("formal slope corrective run is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_SLOPE_CORRECTIVE_TRAINING_V1R":
            raise RuntimeError("slope corrective Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen slope corrective tool mismatch: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen slope corrective input mismatch: {relative}")
        environment = _environment()
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free before slope corrective training")
        source_before = _verify_dataset()
        write_json(
            run_dir / "config/environment.json",
            {
                "executable": str(PYTHON),
                "versions": environment,
                "deterministic_algorithms": True,
                "source_dataset_before": source_before,
            },
        )
        write_json(
            run_dir / "RUN_STATE.json",
            {
                "schema_version": "v3_run_state_v1",
                "run_id": RUN_ID,
                "state": "RUNNING",
                "note": "C01-C06 fit, C07-C08 selection; C09/C10/M-TARE forbidden.",
            },
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        cache_dir = run_dir / "artifacts/slope_corrective_cache"
        cache_log = run_dir / "logs/00_train_only_feature_cache.log"
        cache_code = _stream(
            [
                "/usr/bin/time",
                "-v",
                str(PYTHON),
                str(CACHE_BUILDER),
                "--dataset-run",
                str(DATASET),
                "--output-dir",
                str(cache_dir),
            ],
            cache_log,
            env,
            1_800,
        )
        cache_manifest = load_json(cache_dir / "manifest.json") if (cache_dir / "manifest.json").is_file() else {}
        cache_rss = _peak_rss(cache_log)
        if (
            cache_code != 0
            or cache_rss is None
            or cache_rss > HOST_RSS_LIMIT_KIB
            or cache_manifest.get("counts")
            != {
                "fit_worlds": 60,
                "selection_worlds": 20,
                "fit_sequences": 142184,
                "selection_sequences": 45942,
                "unique_frames": 252430,
                "strict_test_worlds_read": 0,
                "c09_worlds_read": 0,
                "mtare_worlds_read": 0,
            }
        ):
            raise RuntimeError("train-only slope corrective feature cache failed")
        write_json(
            run_dir / "metrics/cache_summary.json",
            {
                "counts": cache_manifest["counts"],
                "peak_host_rss_kib": cache_rss,
                "manifest_sha256": _sha256(cache_dir / "manifest.json"),
                "normalization_sha256": _sha256(cache_dir / "normalization.json"),
            },
        )
        for seed in (0, 1, 2):
            output_dir = run_dir / f"artifacts/models/seed{seed}"
            log_path = run_dir / f"logs/{seed + 1:02d}_seed{seed}_training.log"
            code = _stream(
                [
                    "/usr/bin/time",
                    "-v",
                    str(PYTHON),
                    str(TRAINER),
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(seed),
                    "--epochs",
                    "50",
                    "--batch-size",
                    "1024",
                    "--learning-rate",
                    "0.001",
                    "--weight-decay",
                    "0.0001",
                    "--patience",
                    "8",
                ],
                log_path,
                env,
                1_200,
            )
            child = load_json(output_dir / "summary.json") if (output_dir / "summary.json").is_file() else {}
            peak_rss = _peak_rss(log_path)
            if (
                code != 0
                or peak_rss is None
                or peak_rss > HOST_RSS_LIMIT_KIB
                or child.get("overall_status") != "PASS_GSE_SLOPE_CORRECTIVE_TRAINING_SEED_V1"
                or child.get("seed") != seed
                or child.get("fit_sequences_per_epoch") != 142184
                or child.get("selection_sequences_per_evaluation") != 45942
                or child.get("c09_worlds_read") != 0
                or child.get("strict_test_worlds_read") != 0
                or child.get("mtare_worlds_read") != 0
                or child.get("peak_gpu_memory_bytes", GPU_MEMORY_LIMIT_BYTES + 1) > GPU_MEMORY_LIMIT_BYTES
                or not all(
                    (output_dir / name).is_file()
                    for name in (
                        "best.pt",
                        "last.pt",
                        "selection_outputs.npz",
                        "best_selection_metrics.json",
                        "epoch_metrics.jsonl",
                    )
                )
            ):
                raise RuntimeError(f"formal slope corrective seed {seed} failed technical evidence")
            metrics = child["best_selection"]
            seed_summaries.append(
                {
                    "seed": seed,
                    "epochs_completed": child["epochs_completed"],
                    "best_epoch": child["best_epoch"],
                    "optimizer_steps": child["optimizer_steps"],
                    "parameters": child["parameters"],
                    "duration_seconds": child["duration_seconds"],
                    "peak_host_rss_kib": peak_rss,
                    "peak_gpu_memory_bytes": child["peak_gpu_memory_bytes"],
                    "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
                    "selection_outputs_sha256": _sha256(output_dir / "selection_outputs.npz"),
                    "best_selection": metrics,
                }
            )
            write_json(run_dir / "metrics/seed_progress.json", {"completed": seed_summaries})
        improvements = [
            float(item["best_selection"]["relative_improvement_over_five_frame_prior"])
            for item in seed_summaries
        ]
        nonregression = all(
            float(item["best_selection"]["corrected_mae_deg"])
            <= float(item["best_selection"]["five_frame_prior_mae_deg"]) + 1e-8
            for item in seed_summaries
        )
        family_safe = all(
            float(family["relative_improvement"]) >= -0.05
            for item in seed_summaries
            for family in item["best_selection"]["per_topology_family"]
        )
        scientific_pass = (
            nonregression
            and sum(value >= 0.05 for value in improvements) >= 2
            and sum(improvements) / len(improvements) >= 0.05
            and family_safe
        )
        if not scientific_pass:
            raise RuntimeError(
                "train-only corrective gate failed: requires mean >=5%, at least two seeds >=5%, all seeds nonregressing, and every family >=-5%"
            )
        source_after = _verify_dataset()
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            source_before != source_after
            or result_bytes > DISK_LIMIT_BYTES
            or time.monotonic() - started > TOTAL_TIME_LIMIT_SECONDS
        ):
            raise RuntimeError("slope corrective aggregate integrity or resource contract failed")
        overall = PASS_STATUS
        write_json(
            run_dir / "metrics/summary.json",
            {
                "schema_version": "gse_slope_corrective_three_seed_training_v1r",
                "overall_status": overall,
                "scientific_status": "TRAIN_ONLY_CORRECTIVE_PASS_AWAITING_ONE_NEW_COMPLETE_C09_VALIDATION",
                "seeds": seed_summaries,
                "mean_relative_improvement_over_five_frame_prior": sum(improvements) / len(improvements),
                "seeds_at_least_five_percent_improvement": sum(value >= 0.05 for value in improvements),
                "all_seed_nonregression": nonregression,
                "all_topology_family_regression_within_five_percent": family_safe,
                "source_dataset_before": source_before,
                "source_dataset_after": source_after,
                "source_unchanged": True,
                "fit_worlds": 60,
                "selection_worlds": 20,
                "fit_sequences": 142184,
                "selection_sequences": 45942,
                "c09_worlds_read": 0,
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "optimizer_steps": sum(item["optimizer_steps"] for item in seed_summaries),
                "duration_seconds": time.monotonic() - started,
                "result_bytes_before_seal": result_bytes,
            },
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            run_dir / "metrics/summary.json",
            {
                "schema_version": "gse_slope_corrective_three_seed_training_v1r",
                "overall_status": overall,
                "error": error,
                "completed_seeds": seed_summaries,
                "source_dataset_before": source_before,
                "source_dataset_after": source_after,
                "c09_worlds_read": 0,
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "duration_seconds": time.monotonic() - started,
            },
        )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    sealed_files = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files, "error": error}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
