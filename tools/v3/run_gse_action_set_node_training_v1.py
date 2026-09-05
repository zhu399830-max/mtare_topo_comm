#!/usr/bin/env python3
"""Execute and seal the immutable C01--C08 action-set node capacity run."""

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


RUN_ID = "gate3_20260828_gse_action_set_node_training_v1_seed0"
PASS_STATUS = "PASS_GSE_ACTION_SET_NODE_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_ACTION_SET_NODE_TRAINING_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
BUILDER = PROJECT_ROOT / "tools/v3/build_gse_action_set_node_cache_v1.py"
TRAINER = PROJECT_ROOT / "tools/v3/train_gse_action_set_node_v1.py"
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_action_set_node_selection_v1.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _run(command: list[str], log: Path, env: dict[str, str]) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
            text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=14_400, check=False,
        )
    return int(completed.returncode), _peak_rss(log)


def _verify_frozen(spec: dict) -> dict[str, str]:
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
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("action-set capacity run may execute only once")
    started = time.monotonic()
    overall = FAIL_STATUS
    error = None
    source_before: dict[str, str] = {}
    source_after: dict[str, str] = {}
    subprocesses = []
    selection = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training" or spec.get("seed") != 0:
            raise RuntimeError("action-set formal scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("standing authorization is not bound to action-set training")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_ACTION_SET_NODE_TRAINING_V1":
            raise RuntimeError("action-set Data Card mismatch")
        for record in spec["frozen_tools"].values():
            if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")
        source_before = _verify_frozen(spec)
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,scipy,sys,torch;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "torch": "2.9.0+cu129", "cuda": "12.9", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected:
            raise RuntimeError(f"action-set environment drift: {environment}")
        if shutil.disk_usage(PROJECT_ROOT).free < 4 * 1024**3:
            raise RuntimeError("less than 4 GiB free before action-set training")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic_algorithms": True})
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        scratch = run_dir / "scratch/action_set_cache"
        models = run_dir / "artifacts/models"
        models.mkdir(parents=True)
        token_paths = [SOURCE / f"artifacts/models/seed{seed}/frozen_exit_token_outputs.npz" for seed in (0, 1, 2)]
        command = [str(PYTHON), str(BUILDER), "--teacher", str(TEACHER / "artifacts/teacher_observations.jsonl"), "--pair-cache", str(SOURCE / "artifacts/pair_cache/pairs.npz")]
        for seed, path in enumerate(token_paths):
            command.extend((f"--seed{seed}", str(path)))
        command.extend(("--output-dir", str(scratch)))
        code, rss = _run(command, run_dir / "logs/00_cache.log", env)
        subprocesses.append({"stage": "cache", "returncode": code, "peak_host_rss_kib": rss})
        if code != 0:
            raise RuntimeError("action-set cache builder failed")
        manifest = load_json(scratch / "manifest.json")
        if (
            manifest.get("causal_observations") != 188126
            or manifest.get("fit_observations") != 142184
            or manifest.get("selection_observations") != 45942
            or manifest.get("fit_decision_episodes") != 3282
            or manifest.get("selection_decision_episodes") != 1136
            or any(manifest.get(key) != 0 for key in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("action-set cache evidence drift")
        shutil.copy2(scratch / "manifest.json", run_dir / "artifacts/cache_manifest.json")
        shutil.copy2(scratch / "normalization_mean.npy", run_dir / "artifacts/normalization_mean.npy")
        shutil.copy2(scratch / "normalization_scale.npy", run_dir / "artifacts/normalization_scale.npy")
        for seed in (0, 1, 2):
            command = [
                str(PYTHON), str(TRAINER), "--cache-dir", str(scratch),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed),
                "--epochs", str(spec["hyperparameters"]["epochs"]),
                "--batch-size", str(spec["hyperparameters"]["training_batch_size"]),
                "--evaluation-batch-size", str(spec["hyperparameters"]["evaluation_batch_size"]),
                "--learning-rate", str(spec["hyperparameters"]["learning_rate"]),
                "--weight-decay", str(spec["hyperparameters"]["weight_decay"]),
            ]
            code, rss = _run(command, run_dir / f"logs/{seed + 1:02d}_seed{seed}_training.log", env)
            subprocesses.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"action-set seed{seed} training failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            if summary.get("optimizer_steps") != 13332 or summary.get("backbone_optimizer_steps") != 0:
                raise RuntimeError(f"action-set seed{seed} step count drift")
        selection_dir = run_dir / "metrics/selection"
        command = [str(PYTHON), str(EVALUATOR), "--cache-dir", str(scratch)]
        for seed in (0, 1, 2):
            command.extend((f"--seed{seed}", str(models / f"seed{seed}/selection_outputs.npz")))
        command.extend(("--output-dir", str(selection_dir)))
        code, rss = _run(command, run_dir / "logs/04_selection.log", env)
        subprocesses.append({"stage": "selection", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("action-set selection evaluator failed as a program")
        selection = load_json(selection_dir / "summary.json")
        scientific_pass = selection.get("status") == "PASS_GSE_ACTION_SET_NODE_SELECTION_V1"
        if (code == 0) != scientific_pass:
            raise RuntimeError("action-set evaluator return/status mismatch")
        cache_bytes = sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file())
        shutil.rmtree(run_dir / "scratch")
        write_json(run_dir / "artifacts/cache_deletion_audit.json", {"deleted_regenerable_cache_bytes": cache_bytes, "retained": ["cache_manifest.json", "normalization_mean.npy", "normalization_scale.npy"]})
        source_after = _verify_frozen(spec)
        if source_before != source_after:
            raise RuntimeError("frozen source changed during action-set training")
        overall = PASS_STATUS if scientific_pass else FAIL_STATUS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_action_set_node_training_outer_v1",
        "overall_status": overall,
        "scientific_pass": overall == PASS_STATUS,
        "error": error,
        "duration_seconds": time.monotonic() - started,
        "subprocesses": subprocesses,
        "selection": selection,
        "source_unchanged": bool(source_before and source_before == source_after),
        "optimizer_steps": 39996 if len([x for x in subprocesses if x["stage"] == "training" and x["returncode"] == 0]) == 3 else None,
        "backbone_optimizer_steps": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS_STATUS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
