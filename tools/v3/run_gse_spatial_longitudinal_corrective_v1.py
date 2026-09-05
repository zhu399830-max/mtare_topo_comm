#!/usr/bin/env python3
"""Execute, qualify and seal the spatial longitudinal corrective."""

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


RUN_ID = "gate3_20260828_gse_spatial_longitudinal_corrective_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_LONGITUDINAL_CORRECTIVE_V1"
FAIL = "FAIL_GSE_SPATIAL_LONGITUDINAL_CORRECTIVE_V1"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
GSE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
CAUSAL_TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
SCALAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
PROJECTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
SPATIAL = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"longitudinal corrective frozen input drift: {relative}")
        result[relative] = actual
    return result


def _peak_rss(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1].strip())
    return None


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return int(completed.returncode), _peak_rss(log)


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"; files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files: stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("longitudinal corrective may execute only once")
    started = time.monotonic(); overall = FAIL; error = None; before = after = {}; ensemble = {}; processes = []; deletions = []
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("longitudinal corrective scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_LONGITUDINAL_CORRECTIVE_V1" or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("longitudinal corrective Data Card drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"longitudinal corrective frozen tool drift: {record['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c", "import json,numcodecs,numpy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "numcodecs": "0.15.1", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected: raise RuntimeError(f"longitudinal corrective environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3: raise RuntimeError("less than 8 GiB free before longitudinal caches")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic_algorithms": True, "sequential_temporary_cache": True})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        scratch = run_dir / "scratch"; models = run_dir / "artifacts/models"; manifests = run_dir / "artifacts/cache_manifests"
        scratch.mkdir(); models.mkdir(parents=True); manifests.mkdir(parents=True)
        for seed in (0, 1, 2):
            cache = scratch / f"seed{seed}_cache"
            build = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/build_gse_spatial_event_center_cache_v1.py"),
                "--dataset-run", str(DATASET), "--teacher", str(CAUSAL_TEACHER / "artifacts/teacher_observations.jsonl"),
                "--checkpoint", str(GSE / f"artifacts/models/seed{seed}/best.pt"), "--output-dir", str(cache), "--seed", str(seed), "--batch-size", "128",
            ]
            code, rss = _run(build, run_dir / f"logs/{seed * 2:02d}_seed{seed}_cache.log", env, 1800); processes.append({"stage": "cache", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0: raise RuntimeError(f"seed{seed} longitudinal cache build failed")
            current_manifest = load_json(cache / "manifest.json"); old_manifest = load_json(SPATIAL / f"artifacts/cache_manifests/seed{seed}_manifest.json")
            if (
                current_manifest.get("cache_bytes") != 2526043720
                or current_manifest.get("array_sha256") != old_manifest.get("array_sha256")
                or current_manifest.get("checkpoint_sha256") != old_manifest.get("checkpoint_sha256")
                or current_manifest.get("teacher_sha256") != old_manifest.get("teacher_sha256")
                or any(current_manifest.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ): raise RuntimeError(f"seed{seed} longitudinal cache does not reproduce V1")
            shutil.copy2(cache / "manifest.json", manifests / f"seed{seed}_manifest.json")
            train = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_spatial_longitudinal_corrective_v1.py"),
                "--spatial-cache", str(cache), "--action-cache", str(ACTION / "scratch/action_set_cache"),
                "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
                "--action-checkpoint", str(ACTION / f"artifacts/models/seed{seed}/best.pt"),
                "--scalar-checkpoint", str(SCALAR / f"artifacts/models/seed{seed}/best.pt"),
                "--spatial-checkpoint", str(SPATIAL / f"artifacts/models/seed{seed}/best.pt"),
                "--initial-selection-output", str(SPATIAL / f"artifacts/models/seed{seed}/selection_outputs.npz"),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed), "--epochs", "10",
                "--identities-per-event", "16", "--learning-rate", "0.0005", "--evaluation-batch-size", "128",
            ]
            code, rss = _run(train, run_dir / f"logs/{seed * 2 + 1:02d}_seed{seed}_training.log", env, 7200); processes.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0: raise RuntimeError(f"seed{seed} longitudinal corrective training failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            if (
                summary.get("optimizer_steps") != 3960 or summary.get("spatial_decoder_optimizer_steps") != 0
                or summary.get("backbone_optimizer_steps") != 0 or summary.get("fit_rows") != 25294
                or summary.get("selection_rows") != 8839 or summary.get("transverse_max_drift_m") != 0.0
                or any(summary.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
            ): raise RuntimeError(f"seed{seed} longitudinal evidence drift")
            cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()); shutil.rmtree(cache)
            deletions.append({"seed": seed, "deleted_regenerable_cache_bytes": cache_bytes, "retained_manifest": str((manifests / f"seed{seed}_manifest.json").relative_to(run_dir))})
        scratch.rmdir(); write_json(run_dir / "artifacts/cache_deletion_audit.json", {"schema_version": "gse_spatial_longitudinal_cache_deletion_audit_v1", "records": deletions, "reason": "Exact V1 cache reproduction is deterministic scratch; array digests are retained."})
        evaluation = run_dir / "metrics/ensemble"; command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py")]
        for seed in (0, 1, 2): command.extend((f"--seed{seed}", str(models / f"seed{seed}")))
        command.extend(("--baseline-projection", str(PROJECTION / "artifacts/projection/event_center_projection.npz"), "--action-cache", str(ACTION / "scratch/action_set_cache"), "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"), "--output-dir", str(evaluation)))
        code, rss = _run(command, run_dir / "logs/06_ensemble.log", env, 600); processes.append({"stage": "ensemble", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2): raise RuntimeError("longitudinal ensemble evaluator program failure")
        ensemble = load_json(evaluation / "summary.json"); scientific_pass = ensemble.get("status") == "PASS_GSE_SPATIAL_EVENT_CENTER_ENSEMBLE_V1"
        if (code == 0) != scientific_pass: raise RuntimeError("longitudinal ensemble status mismatch")
        after = _verify(spec)
        if before != after: raise RuntimeError("longitudinal corrective sources changed")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    optimizer_steps = sum(int(load_json(path).get("optimizer_steps", 0)) for path in (run_dir / "artifacts/models").glob("seed*/summary.json")) if (run_dir / "artifacts/models").exists() else 0
    summary = {"schema_version": "gse_spatial_longitudinal_corrective_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "ensemble": ensemble, "subprocesses": processes, "optimizer_steps": optimizer_steps, "spatial_decoder_optimizer_steps": 0, "backbone_optimizer_steps": 0, "deleted_regenerable_cache_bytes": sum(value["deleted_regenerable_cache_bytes"] for value in deletions), "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0}
    write_json(run_dir / "metrics/summary.json", summary); write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2)); return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
