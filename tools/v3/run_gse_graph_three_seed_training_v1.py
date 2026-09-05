#!/usr/bin/env python3
"""Run seeds 0/1/2 once, validate evidence, and seal GSE training V1."""

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


RUN_ID = "gate2_20260824_gse_graph_three_seed_training_v1_seed0"
PASS_STATUS = "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_graph_v1.py"
SMOKE = PROJECT_ROOT / "tools/v3/smoke_gse_real_batch_v1.py"
DATASET = (
    PROJECT_ROOT
    / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
)
DATASET_STATUS = "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
SEED_TIMEOUT_SECONDS = 40_000
TOTAL_TIME_LIMIT_SECONDS = 129_600
HOST_RSS_LIMIT_KIB = 16 * 1024**2
GPU_MEMORY_LIMIT_BYTES = 28 * 1024**3
DISK_LIMIT_BYTES = 8 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset() -> dict:
    state = load_json(DATASET / "RUN_STATE.json")
    summary = load_json(DATASET / "metrics/summary.json")
    runner = load_json(DATASET / "metrics/runner_summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != DATASET_STATUS
        or summary.get("overall_status") != DATASET_STATUS
        or runner.get("overall_status") != DATASET_STATUS
        or summary.get("split_totals", {}).get("train", {}).get("sequences") != 188126
        or summary.get("split_totals", {}).get("validation", {}).get("sequences") != 24462
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("the frozen GSE dataset is not the required sealed PASS")
    seal = DATASET / "artifacts/evidence_sha256.txt"
    checked = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"dataset seal mismatch: {relative}")
        checked += 1
    if checked != 33083:
        raise RuntimeError("dataset seal entry count drift")
    return {
        "run": str(DATASET.relative_to(PROJECT_ROOT)),
        "seal_sha256": _sha256(seal),
        "verified_seal_entries": checked,
        "summary_sha256": _sha256(DATASET / "metrics/summary.json"),
    }


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _stream_command(argv: list[str], *, log_path: Path, env: dict[str, str], timeout: int) -> int:
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
        raise RuntimeError(f"frozen GSE training environment drift: {observed}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time GSE training run state mismatch")
    started = time.monotonic()
    overall = "FAIL_GSE_GRAPH_THREE_SEED_TRAINING_V1"
    error: str | None = None
    summaries = []
    source_before = None
    source_after = None
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("formal GSE training scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("formal GSE training is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_FORMAL_THREE_SEED_TRAINING":
            raise RuntimeError("operation-bound GSE training Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen training tool mismatch: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen training input mismatch: {relative}")
        environment = _environment()
        if shutil.disk_usage(PROJECT_ROOT).free < 16 * 1024**3:
            raise RuntimeError("less than 16 GiB free before formal GSE training")
        source_before = _verify_dataset()
        write_json(
            run_dir / "config/environment.json",
            {
                "executable": str(PYTHON),
                "versions": environment,
                "cublas_workspace_config": ":4096:8",
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
                "note": "Three frozen GSE seeds; validation-only checkpoint selection; C10/M-TARE forbidden.",
            },
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        smoke_log = run_dir / "logs/00_real_batch_no_update_smoke.log"
        smoke_code = _stream_command(
            [str(PYTHON), str(SMOKE), "--dataset-run", str(DATASET), "--batch-size", "64"],
            log_path=smoke_log,
            env=env,
            timeout=600,
        )
        smoke_text = smoke_log.read_text(encoding="utf-8").split("\nrunner_observed_seconds=", 1)[0]
        smoke = json.loads(smoke_text)
        write_json(run_dir / "metrics/real_batch_smoke.json", smoke)
        if (
            smoke_code != 0
            or smoke.get("overall_status") != "PASS_GSE_REAL_BATCH_NO_UPDATE_SMOKE_V1"
            or smoke.get("weights_unchanged") is not True
            or smoke.get("optimizer_steps") != 0
            or smoke.get("batch_positive_association_identities", 0) < 1
        ):
            raise RuntimeError("real GSE batch no-update smoke failed")

        for seed in (0, 1, 2):
            output_dir = run_dir / f"artifacts/models/seed{seed}"
            log_path = run_dir / f"logs/{seed + 1:02d}_seed{seed}_training.log"
            argv = [
                "/usr/bin/time",
                "-v",
                str(PYTHON),
                str(TRAINER),
                "--dataset-run",
                str(DATASET),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--epochs",
                "30",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0003",
                "--weight-decay",
                "0.0001",
                "--patience",
                "6",
            ]
            code = _stream_command(argv, log_path=log_path, env=env, timeout=SEED_TIMEOUT_SECONDS)
            text = log_path.read_text(encoding="utf-8")
            rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
            peak_rss_kib = int(rss_match.group(1)) if rss_match else None
            child = load_json(output_dir / "summary.json") if (output_dir / "summary.json").is_file() else {}
            if (
                code != 0
                or peak_rss_kib is None
                or peak_rss_kib > HOST_RSS_LIMIT_KIB
                or child.get("overall_status") != "PASS_GSE_GRAPH_TRAINING_SEED_V1"
                or child.get("seed") != seed
                or child.get("train_sequences_per_epoch") != 188126
                or child.get("validation_sequences_per_evaluation") != 24462
                or child.get("strict_test_worlds_read") != 0
                or child.get("mtare_worlds_read") != 0
                or child.get("peak_gpu_memory_bytes", GPU_MEMORY_LIMIT_BYTES + 1) > GPU_MEMORY_LIMIT_BYTES
                or not all((output_dir / name).is_file() for name in ("best.pt", "last.pt", "validation_outputs.npz", "best_validation_metrics.json", "epoch_metrics.jsonl"))
            ):
                raise RuntimeError(f"formal GSE seed {seed} failed evidence or resource checks")
            summaries.append(
                {
                    "seed": seed,
                    "epochs_completed": child["epochs_completed"],
                    "best_epoch": child["best_epoch"],
                    "optimizer_steps": child["optimizer_steps"],
                    "parameters": child["parameters"],
                    "duration_seconds": child["duration_seconds"],
                    "peak_host_rss_kib": peak_rss_kib,
                    "peak_gpu_memory_bytes": child["peak_gpu_memory_bytes"],
                    "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
                    "last_checkpoint_sha256": _sha256(output_dir / "last.pt"),
                    "validation_outputs_sha256": _sha256(output_dir / "validation_outputs.npz"),
                    "best_validation": child["best_validation"],
                }
            )
            write_json(run_dir / "metrics/seed_progress.json", {"completed": summaries})
        source_after = _verify_dataset()
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if (
            source_before != source_after
            or len(summaries) != 3
            or {record["seed"] for record in summaries} != {0, 1, 2}
            or result_bytes > DISK_LIMIT_BYTES
            or time.monotonic() - started > TOTAL_TIME_LIMIT_SECONDS
        ):
            raise RuntimeError("formal GSE aggregate integrity/resource contract failed")
        overall = PASS_STATUS
        write_json(
            run_dir / "metrics/summary.json",
            {
                "schema_version": "gse_graph_three_seed_training_v1",
                "overall_status": overall,
                "scientific_status": "TRAINING_COMPLETE_AWAITING_OFFLINE_BASELINE_AND_TOPOLOGY_GATES",
                "seeds": summaries,
                "source_dataset_before": source_before,
                "source_dataset_after": source_after,
                "source_unchanged": True,
                "selection_rule": "minimum validation total multitask loss independently per seed",
                "strict_test_worlds_read": 0,
                "mtare_worlds_read": 0,
                "threshold_calibration_runs": 0,
                "optimizer_steps": sum(record["optimizer_steps"] for record in summaries),
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
                "schema_version": "gse_graph_three_seed_training_v1",
                "overall_status": overall,
                "error": error,
                "completed_seeds": summaries,
                "source_dataset_before": source_before,
                "source_dataset_after": source_after,
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
