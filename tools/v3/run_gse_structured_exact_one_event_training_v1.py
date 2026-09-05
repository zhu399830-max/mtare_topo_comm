#!/usr/bin/env python3
"""Execute and seal three-seed structured exact-one event training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_STRUCTURED_EXACT_ONE_EVENT_TRAINING_V1"
FAIL = "FAIL_GSE_STRUCTURED_EXACT_ONE_EVENT_TRAINING_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
UNIFIED = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
OLD_STATE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss(path: Path) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def run(command: list[str], log: Path, environment: dict[str, str], timeout: int) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    return int(completed.returncode), peak_rss(log)


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run_dir.name != run_id or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("structured exact-one training executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    selection: dict = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    subprocesses = []
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_STRUCTURED_EXACT_ONE_EVENT_TRAINING_V1" or card.get("approval", {}).get("authorized_operations") != ["training"]:
            raise RuntimeError(f"structured exact-one Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"], text=True))
        expected = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected:
            raise RuntimeError(f"structured exact-one environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic_algorithms": True})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        process_environment = os.environ.copy()
        process_environment.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        models = run_dir / "artifacts/models"
        models.mkdir(parents=True)
        observations = [UNIFIED / f"artifacts/unified_observation/seed{seed}_unified_observation_features.npy" for seed in range(3)]
        for seed in range(3):
            command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_structured_exact_one_event_v1.py"), "--cache-dir", str(ACTION / "scratch/action_set_cache"), "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed), "--epochs", "6", "--batch-size", "128", "--evaluation-batch-size", "512", "--learning-rate", "0.0003", "--weight-decay", "0.0001"]
            for path in observations:
                command.extend(("--observation", str(path)))
            code, rss = run(command, run_dir / f"logs/{seed:02d}_seed{seed}_training.log", process_environment, 14_400)
            subprocesses.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"structured exact-one seed{seed} training failed")
            seed_summary = load_json(models / f"seed{seed}/summary.json")
            if seed_summary.get("optimizer_steps") != 6_666 or seed_summary.get("parameters") != 240_101 or seed_summary.get("c08_checkpoint_observations") != 0 or seed_summary.get("perception_backbone_optimizer_steps") != 0:
                raise RuntimeError(f"structured exact-one seed{seed} optimization/split drift")
        evaluation = run_dir / "metrics/selection"
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_structured_exact_one_event_selection_v1.py"), "--cache-dir", str(ACTION / "scratch/action_set_cache"), "--old-state-summary", str(OLD_STATE / "metrics/summary.json"), "--output-dir", str(evaluation)]
        for seed in range(3):
            command.extend((f"--seed{seed}", str(models / f"seed{seed}/selection_outputs.npz")))
        code, rss = run(command, run_dir / "logs/03_selection.log", process_environment, 1_800)
        subprocesses.append({"stage": "selection", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("structured exact-one evaluator program failure")
        selection = load_json(evaluation / "summary.json")
        scientific_pass = selection.get("status") == "PASS_GSE_STRUCTURED_EXACT_ONE_EVENT_SELECTION_V1"
        if (code == 0) != scientific_pass:
            raise RuntimeError("structured exact-one selection status/return mismatch")
        required = [
            *[models / f"seed{seed}/{name}" for seed in range(3) for name in ("best.pt", "history.json", "selection_outputs.npz", "summary.json")],
            evaluation / "summary.json", evaluation / "selection_ensemble_outputs.npz", evaluation / "candidate_grid.csv", evaluation / "figure_source.json",
            *[evaluation / f"gse_structured_exact_one_event_selection_v1.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("structured exact-one evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("structured exact-one training changed a frozen source")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    total_steps = sum(6_666 for item in subprocesses if item["stage"] == "training" and item["returncode"] == 0)
    write_json(run_dir / "metrics/summary.json", {"schema_version": "gse_structured_exact_one_event_training_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "subprocesses": subprocesses, "selection": selection, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": total_steps, "perception_backbone_optimizer_steps": 0, "c08_checkpoint_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "optimizer_steps": total_steps, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
