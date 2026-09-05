#!/usr/bin/env python3
"""Execute, qualify, retain paper evidence and seal one spatial-set capacity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_EVENT_SET_CAPACITY_V1"
FAIL = "FAIL_GSE_SPATIAL_EVENT_SET_CAPACITY_V1"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0/artifacts/export/teacher"
ENCODERS = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
PROJECTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts/projection"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    observed = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha256(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"spatial event capacity frozen input drift: {relative}")
        observed[relative] = actual
    return observed


def _peak_rss(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
            text=True, stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    return int(completed.returncode), _peak_rss(log)


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
        raise RuntimeError("spatial event capacity formal run may execute only once")
    started = time.monotonic()
    overall = FAIL
    error = None
    source_before = {}
    source_after = {}
    stages = []
    evaluation = {}
    deletion_records = []
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if (
            spec.get("gate") != 3
            or spec.get("operation") != "training"
            or spec.get("user_authorization", {}).get("status") != "APPROVED"
            or card.get("approval", {}).get("status") != "APPROVED"
            or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_CAPACITY_V1"
        ):
            raise RuntimeError("spatial event capacity authorization/scope drift")
        for record in spec["frozen_tools"].values():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"spatial event capacity tool drift: {record['path']}")
        source_before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {
            "python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7", "numcodecs": "0.15.1",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected:
            raise RuntimeError(f"spatial event capacity environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 20 * 1024**3:
            raise RuntimeError("less than 20 GiB free before one temporary causal feature cache")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "deterministic_algorithms": True, "sequential_cache": True,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Three seed-matched frozen encoders; only 93,638-parameter set decoders update.",
        })
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        scratch = run_dir / "scratch"
        models = run_dir / "artifacts/models"
        manifests = run_dir / "artifacts/cache_manifests"
        legacy = run_dir / "artifacts/exclusive_baseline"
        scratch.mkdir(); models.mkdir(parents=True); manifests.mkdir(parents=True); legacy.mkdir(parents=True)
        for seed in (0, 1, 2):
            cache = scratch / f"seed{seed}_causal_cache"
            build = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/build_gse_spatial_event_set_cache_v1.py"),
                "--dataset-run", str(DATASET), "--teacher-root", str(TEACHER),
                "--checkpoint", str(ENCODERS / f"artifacts/models/seed{seed}/best.pt"),
                "--output-dir", str(cache), "--seed", str(seed),
                "--encoder-batch-size", "128", "--causal-batch-size", "128",
            ]
            code, rss = _run(build, run_dir / f"logs/{seed * 2:02d}_seed{seed}_cache.log", env, 3600)
            stages.append({"stage": "cache", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} causal feature cache failed")
            cache_manifest = load_json(cache / "manifest.json")
            if (
                cache_manifest.get("seed") != seed
                or cache_manifest.get("worlds") != 80
                or cache_manifest.get("observations") != 188126
                or cache_manifest.get("model_inference_frames") != 252430
                or cache_manifest.get("optimizer_steps") != 0
                or not 8 * 1024**3 < cache_manifest.get("cache_bytes", 0) < 10 * 1024**3
                or any(cache_manifest.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} causal cache evidence drift")
            shutil.copy2(cache / "manifest.json", manifests / f"seed{seed}_manifest.json")
            shutil.copy2(cache / "legacy_event_logits.npy", legacy / f"seed{seed}_legacy_event_logits.npy")
            train = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_spatial_event_set_v1.py"),
                "--feature-cache", str(cache), "--teacher-root", str(TEACHER),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed),
                "--epochs", "8", "--batch-size", "256", "--evaluation-batch-size", "256",
                "--learning-rate", "0.0003",
            ]
            code, rss = _run(train, run_dir / f"logs/{seed * 2 + 1:02d}_seed{seed}_training.log", env, 10800)
            stages.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} spatial set decoder training failed")
            seed_summary = load_json(models / f"seed{seed}/summary.json")
            if (
                seed_summary.get("optimizer_steps") != 4448
                or seed_summary.get("encoder_optimizer_steps") != 0
                or seed_summary.get("fit_rows") != 142184
                or seed_summary.get("selection_rows") != 45942
                or seed_summary.get("trainable_parameters") != 93638
                or seed_summary.get("loss_weights", {}).get("descriptor") != 0.0
                or any(seed_summary.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} training evidence drift")
            cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            shutil.rmtree(cache)
            deletion_records.append({
                "seed": seed, "deleted_regenerable_cache_bytes": cache_bytes,
                "retained_manifest": str((manifests / f"seed{seed}_manifest.json").relative_to(run_dir)),
                "retained_legacy_logits": str((legacy / f"seed{seed}_legacy_event_logits.npy").relative_to(run_dir)),
            })
        scratch.rmdir()
        write_json(run_dir / "artifacts/cache_deletion_audit.json", {
            "schema_version": "gse_spatial_event_set_cache_deletion_audit_v1",
            "records": deletion_records,
            "reason": "Each seed cache is deterministic scratch; retain manifest/digests and the small legacy-logit baseline only.",
        })
        evaluation_dir = run_dir / "metrics/capacity"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_spatial_event_set_capacity_v1.py"),
            "--teacher-root", str(TEACHER), "--dataset-run", str(DATASET),
            "--output-dir", str(evaluation_dir),
        ]
        for seed in (0, 1, 2):
            command.extend(("--legacy-logits", str(legacy / f"seed{seed}_legacy_event_logits.npy")))
        for seed in (0, 1, 2):
            command.extend(("--model-dir", str(models / f"seed{seed}")))
        for seed in (0, 1, 2):
            command.extend(("--projection", str(PROJECTION / f"seed{seed}_all_rows.npz")))
        code, rss = _run(command, run_dir / "logs/06_capacity_evaluation.log", env, 7200)
        stages.append({"stage": "capacity_evaluation", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("capacity evaluator program failure")
        evaluation = load_json(evaluation_dir / "summary.json")
        scientific_pass = evaluation.get("status") == PASS
        if scientific_pass != (code == 0):
            raise RuntimeError("capacity evaluator return/status mismatch")
        source_after = _verify(spec)
        if source_before != source_after:
            raise RuntimeError("spatial event capacity sources changed during run")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    optimizer_steps = sum(
        int(load_json(path).get("optimizer_steps", 0))
        for path in (run_dir / "artifacts/models").glob("seed*/summary.json")
    ) if (run_dir / "artifacts/models").exists() else 0
    summary = {
        "schema_version": "gse_spatial_event_set_capacity_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS,
        "error": error,
        "evaluation": evaluation,
        "stages": stages,
        "optimizer_steps": optimizer_steps,
        "encoder_optimizer_steps": 0,
        "deleted_regenerable_cache_bytes": sum(row["deleted_regenerable_cache_bytes"] for row in deletion_records),
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(source_before and source_before == source_after),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
