#!/usr/bin/env python3
"""Execute and seal the one immutable GSE open-set association corrective."""

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


RUN_ID = "gate3_20260826_gse_open_set_association_corrective_v1_seed0"
PASS_STATUS = "PASS_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1"
FAIL_STATUS = "FAIL_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1"
DATA_CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1"
SUMMARY_SCHEMA_VERSION = "gse_open_set_association_corrective_v1"
MODEL_EXPECTED_PARAMETERS = 51266
MODEL_REQUIRED_FILES = (
    "best.pt", "normalization.npz", "selection_outputs.npz", "epoch_metrics.jsonl",
    "frozen_observation_features.npy", "summary.json",
)
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
PAIR_BUILDER = PROJECT_ROOT / "tools/v3/build_gse_open_set_pair_cache_v1.py"
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_open_set_association_v1.py"
RSS_LIMIT_KIB = 8 * 1024**2
GPU_LIMIT_BYTES = 4 * 1024**3
DISK_LIMIT_BYTES = 2 * 1024**3
TIME_LIMIT_SECONDS = 6 * 3600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_seal(run: Path, expected_status: str, expected_entries: int) -> dict:
    state = load_json(run / "RUN_STATE.json")
    summary = load_json(run / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != expected_status
        or summary.get("overall_status") != expected_status
    ):
        raise RuntimeError(f"source run is not the required completed PASS: {run.name}")
    seal = run / "artifacts/evidence_sha256.txt"
    checked = 0
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        if not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        checked += 1
    if checked != expected_entries:
        raise RuntimeError(f"source seal entry count drift: {run.name}: {checked}")
    return {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "status": expected_status,
        "seal_entries": checked,
        "seal_sha256": _sha256(seal),
    }


def _environment() -> dict:
    raw = subprocess.check_output(
        [
            str(PYTHON), "-c",
            "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory},sort_keys=True))",
        ],
        text=True,
    )
    observed = json.loads(raw)
    expected = {
        "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
        "cuda": "12.9", "cudnn": 91002, "zarr": "2.18.7", "numcodecs": "0.15.1",
        "gpu": "NVIDIA GeForce RTX 5090 D", "gpu_total_bytes": 33659879424,
    }
    if observed != expected:
        raise RuntimeError(f"frozen open-set environment drift: {observed}")
    return observed


def _stream(argv: list[str], log_path: Path, env: dict[str, str], timeout: int) -> int:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
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


def _peak_rss(path: Path) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(encoding="utf-8"))
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
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time open-set corrective run state mismatch")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    source_before = None
    source_after = None
    seed_summaries = []
    scientific_pass = False
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("formal open-set corrective scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("formal open-set corrective is not authorized")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != DATA_CARD_STATUS:
            raise RuntimeError("open-set corrective Data Card status mismatch")
        for name, record in spec["frozen_tools"].items():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool mismatch: {name}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha256(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen input mismatch: {relative}")
        if shutil.disk_usage(PROJECT_ROOT).free < 6 * 1024**3:
            raise RuntimeError("less than 6 GiB free before open-set corrective")
        environment = _environment()
        source_before = {
            "dataset": _verify_seal(DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1", 33083),
            "training": _verify_seal(TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R", 33),
        }
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "deterministic_algorithms": True,
            "source_before": source_before,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "C01-C06 fit, C07-C08 selection; C09/C10/M-TARE forbidden.",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        pair_cache = run_dir / "artifacts/pair_cache"
        pair_log = run_dir / "logs/00_pair_cache.log"
        code = _stream([
            "/usr/bin/time", "-v", str(PYTHON), str(PAIR_BUILDER),
            "--dataset-run", str(DATASET), "--output-dir", str(pair_cache),
        ], pair_log, env, 1800)
        pair_summary = load_json(pair_cache / "summary.json") if (pair_cache / "summary.json").is_file() else {}
        pair_rss = _peak_rss(pair_log)
        if (
            code != 0 or pair_rss is None or pair_rss > RSS_LIMIT_KIB
            or pair_summary.get("overall_status") != "PASS_GSE_OPEN_SET_PAIR_CACHE_V1"
            or pair_summary.get("counts", {}).get("fit_total") != 135232
            or pair_summary.get("counts", {}).get("selection_total") != 45372
        ):
            raise RuntimeError("formal open-set pair cache failed")
        write_json(run_dir / "metrics/pair_cache_summary.json", {
            **pair_summary, "peak_host_rss_kib": pair_rss,
            "summary_sha256": _sha256(pair_cache / "summary.json"),
        })
        for seed in (0, 1, 2):
            checkpoint = TRAINING / f"artifacts/models/seed{seed}/best.pt"
            output = run_dir / f"artifacts/models/seed{seed}"
            log = run_dir / f"logs/{seed + 1:02d}_seed{seed}.log"
            code = _stream([
                "/usr/bin/time", "-v", str(PYTHON), str(TRAINER),
                "--dataset-run", str(DATASET), "--checkpoint", str(checkpoint),
                "--pair-cache", str(pair_cache), "--output-dir", str(output), "--seed", str(seed),
            ], log, env, 5400)
            summary = load_json(output / "summary.json") if (output / "summary.json").is_file() else {}
            peak_rss = _peak_rss(log)
            if (
                code not in (0, 2) or peak_rss is None or peak_rss > RSS_LIMIT_KIB
                or summary.get("seed") != seed or summary.get("fit_pairs") != 135232
                or summary.get("selection_pairs") != 45372
                or summary.get("c09_worlds_read") != 0 or summary.get("strict_test_worlds_read") != 0
                or summary.get("mtare_worlds_read") != 0
                or summary.get("peak_gpu_memory_bytes", GPU_LIMIT_BYTES + 1) > GPU_LIMIT_BYTES
                or summary.get("parameters") != MODEL_EXPECTED_PARAMETERS
                or not all((output / name).is_file() for name in MODEL_REQUIRED_FILES)
            ):
                raise RuntimeError(f"formal verifier seed {seed} technical evidence failed")
            seed_summaries.append({
                "seed": seed, "overall_status": summary["overall_status"],
                "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
                "optimizer_steps": summary["optimizer_steps"], "parameters": summary["parameters"],
                "best_selection": summary["best_selection"], "duration_seconds": summary["duration_seconds"],
                "peak_host_rss_kib": peak_rss, "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
                "checkpoint_sha256": _sha256(output / "best.pt"),
                "normalization_sha256": _sha256(output / "normalization.npz"),
                "selection_outputs_sha256": _sha256(output / "selection_outputs.npz"),
            })
            write_json(run_dir / "metrics/seed_progress.json", {"completed": seed_summaries})
        scientific_pass = all(row["overall_status"].startswith("PASS_") for row in seed_summaries)
        source_after = {
            "dataset": _verify_seal(DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1", 33083),
            "training": _verify_seal(TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R", 33),
        }
        if source_before != source_after:
            raise RuntimeError("source evidence changed during corrective")
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if result_bytes > DISK_LIMIT_BYTES or time.monotonic() - started > TIME_LIMIT_SECONDS:
            raise RuntimeError("open-set corrective resource contract exceeded")
        overall = PASS_STATUS if scientific_pass else FAIL_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        duration = time.monotonic() - started
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "overall_status": overall,
            "scientific_pass": scientific_pass,
            "error": error,
            "duration_seconds": duration,
            "pair_definition": "same_parent_strictly_past_3d_euclidean_le_16m",
            "pair_counts": {"fit": 135232, "selection": 45372},
            "seeds": seed_summaries,
            "source_before": source_before,
            "source_after": source_after,
            "optimizer_steps": sum(int(row.get("optimizer_steps", 0)) for row in seed_summaries),
            "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "metrics/runner_summary.json", {
            "overall_status": overall, "duration_seconds": duration,
            "result_bytes_before_seal": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
            "error": error,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "COMPLETED" if overall == PASS_STATUS else "FAILED",
            "overall_status": overall,
        })
        _seal(run_dir)
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
