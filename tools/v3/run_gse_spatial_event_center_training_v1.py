#!/usr/bin/env python3
"""Execute, qualify and seal one spatial event-center training run."""

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


RUN_ID = "gate3_20260828_gse_spatial_event_center_training_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1"
FAIL = "FAIL_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SPATIAL_MODELS = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CAUSAL_TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SCALAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
PROJECTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    observed = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"spatial event-center frozen input drift: {relative}")
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
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("spatial event-center formal run may execute only once")
    started = time.monotonic()
    overall = FAIL
    error = None
    source_before = {}
    source_after = {}
    subprocesses = []
    ensemble = {}
    deletion_records = []
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("spatial event-center run scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if (
            spec.get("user_authorization", {}).get("status") != "APPROVED"
            or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1"
            or card.get("approval", {}).get("status") != "APPROVED"
        ):
            raise RuntimeError("spatial event-center Data Card authorization drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"spatial event-center frozen tool drift: {record['path']}")
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
            raise RuntimeError(f"spatial event-center environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free before sequential spatial caches")
        write_json(run_dir / "config/environment.json", {
            "executable": str(PYTHON), "versions": environment,
            "deterministic_algorithms": True, "sequential_temporary_cache": True,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
            "note": "Sequential three-seed spatial cache build and lateral/up event-center training.",
        })
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        scratch = run_dir / "scratch"
        models = run_dir / "artifacts/models"
        manifests = run_dir / "artifacts/cache_manifests"
        scratch.mkdir(); models.mkdir(parents=True); manifests.mkdir(parents=True)
        for seed in (0, 1, 2):
            cache = scratch / f"seed{seed}_cache"
            build = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/build_gse_spatial_event_center_cache_v1.py"),
                "--dataset-run", str(DATASET),
                "--teacher", str(CAUSAL_TEACHER / "artifacts/teacher_observations.jsonl"),
                "--checkpoint", str(SPATIAL_MODELS / f"artifacts/models/seed{seed}/best.pt"),
                "--output-dir", str(cache), "--seed", str(seed), "--batch-size", "128",
            ]
            code, rss = _run(build, run_dir / f"logs/{seed * 2:02d}_seed{seed}_cache.log", env, 1800)
            subprocesses.append({"stage": "cache", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} spatial cache build failed")
            cache_manifest = load_json(cache / "manifest.json")
            if (
                cache_manifest.get("unique_frames") != 252430
                or cache_manifest.get("causal_observations") != 188126
                or cache_manifest.get("cache_bytes") != 2526043720
                or cache_manifest.get("model_inference_frames") != 252430
                or cache_manifest.get("optimizer_steps") != 0
                or any(cache_manifest.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} spatial cache evidence drift")
            shutil.copy2(cache / "manifest.json", manifests / f"seed{seed}_manifest.json")
            train = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_spatial_event_center_v1.py"),
                "--spatial-cache", str(cache),
                "--action-cache", str(ACTION / "scratch/action_set_cache"),
                "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
                "--action-checkpoint", str(ACTION / f"artifacts/models/seed{seed}/best.pt"),
                "--scalar-checkpoint", str(SCALAR / f"artifacts/models/seed{seed}/best.pt"),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed),
                "--epochs", "10", "--identities-per-event", "16",
                "--learning-rate", "0.0005", "--evaluation-batch-size", "128",
            ]
            code, rss = _run(train, run_dir / f"logs/{seed * 2 + 1:02d}_seed{seed}_training.log", env, 7200)
            subprocesses.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} spatial decoder training failed")
            seed_summary = load_json(models / f"seed{seed}/summary.json")
            if (
                seed_summary.get("optimizer_steps") != 3960
                or seed_summary.get("backbone_optimizer_steps") != 0
                or seed_summary.get("fit_rows") != 25294
                or seed_summary.get("selection_rows") != 8839
                or any(seed_summary.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ):
                raise RuntimeError(f"seed{seed} spatial training evidence drift")
            cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            shutil.rmtree(cache)
            deletion_records.append({
                "seed": seed, "deleted_regenerable_cache_bytes": cache_bytes,
                "retained_manifest": str((manifests / f"seed{seed}_manifest.json").relative_to(run_dir)),
            })
        scratch.rmdir()
        write_json(run_dir / "artifacts/cache_deletion_audit.json", {
            "schema_version": "gse_spatial_event_center_cache_deletion_audit_v1",
            "records": deletion_records,
            "reason": "Per-seed frozen spatial arrays are deterministic scratch; manifests retain all array digests and provenance.",
        })
        evaluation = run_dir / "metrics/ensemble"
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py")]
        for seed in (0, 1, 2):
            command.extend((f"--seed{seed}", str(models / f"seed{seed}")))
        command.extend((
            "--baseline-projection", str(PROJECTION / "artifacts/projection/event_center_projection.npz"),
            "--action-cache", str(ACTION / "scratch/action_set_cache"),
            "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
            "--output-dir", str(evaluation),
        ))
        code, rss = _run(command, run_dir / "logs/06_ensemble.log", env, 600)
        subprocesses.append({"stage": "ensemble", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("spatial event-center ensemble evaluator program failure")
        ensemble = load_json(evaluation / "summary.json")
        scientific_pass = ensemble.get("status") == "PASS_GSE_SPATIAL_EVENT_CENTER_ENSEMBLE_V1"
        if (code == 0) != scientific_pass:
            raise RuntimeError("spatial event-center evaluator return/status mismatch")
        source_after = _verify(spec)
        if source_before != source_after:
            raise RuntimeError("spatial event-center sources changed during training")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    optimizer_steps = sum(
        int(load_json(path).get("optimizer_steps", 0))
        for path in (run_dir / "artifacts/models").glob("seed*/summary.json")
    ) if (run_dir / "artifacts/models").exists() else 0
    summary = {
        "schema_version": "gse_spatial_event_center_training_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "error": error, "ensemble": ensemble, "subprocesses": subprocesses,
        "optimizer_steps": optimizer_steps, "backbone_optimizer_steps": 0,
        "deleted_regenerable_cache_bytes": sum(value["deleted_regenerable_cache_bytes"] for value in deletion_records),
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
