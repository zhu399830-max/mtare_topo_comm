#!/usr/bin/env python3
"""Execute and seal three-seed circular slot-transport training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_THREE_SEED_TRAINING_V1"
FAIL = "FAIL_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_THREE_SEED_TRAINING_V1"
INNER_PASS = "PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_SELECTION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_THREE_SEED_TRAINING_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss(path: Path) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def execute(command: list[str], log: Path, environment: dict[str, str], timeout: int) -> tuple[int, int | None]:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False,
        )
    return int(completed.returncode), peak_rss(log)


def seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
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
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("slot-transport training executes exactly once")

    started = time.monotonic()
    overall = FAIL
    error = None
    selection: dict = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    processes: list[dict] = []
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"slot-transport training card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")

        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))",
        ], text=True))
        expected_environment = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if environment != expected_environment:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "deterministic_algorithms": True})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})

        process_environment = os.environ.copy()
        process_environment.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        models = run / "artifacts/models"
        models.mkdir(parents=True)
        for seed in range(3):
            command = [
                str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_cardinality_conditioned_circular_slot_transport_v1.py"),
                "--teacher-root", str(TEACHER / "artifacts/export/teacher"),
                "--source-root", str(DATASET / "artifacts/dataset/train"),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed),
                "--epochs", "10", "--batch-size", "128", "--evaluation-batch-size", "256",
                "--learning-rate", "0.0003", "--weight-decay", "0.0001",
            ]
            code, rss = execute(command, run / f"logs/{seed:02d}_seed{seed}_training.log", process_environment, 28800)
            processes.append({"stage": "training", "seed": seed, "returncode": code, "peak_host_rss_kib": rss})
            if code != 0:
                raise RuntimeError(f"seed{seed} training failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            if (summary.get("schema_version") != "gse_cardinality_conditioned_circular_slot_transport_training_seed_v1"
                    or summary.get("optimizer_steps") != 11370 or summary.get("parameters") != 787328
                    or summary.get("c08_checkpoint_observations") != 0
                    or summary.get("development_output_observations") != 45942):
                raise RuntimeError(f"seed{seed} split/optimization drift")

        evaluation = run / "metrics/selection"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1.py"),
            "--teacher-root", str(TEACHER / "artifacts/export/teacher"),
            "--source-root", str(DATASET / "artifacts/dataset/train"), "--output-dir", str(evaluation),
        ]
        for seed in range(3):
            command.extend(("--prediction-root", str(models / f"seed{seed}/development_predictions")))
        code, rss = execute(command, run / "logs/03_selection.log", process_environment, 3600)
        processes.append({"stage": "selection", "returncode": code, "peak_host_rss_kib": rss})
        if code not in (0, 2):
            raise RuntimeError("slot-transport evaluator program failure")
        selection = load_json(evaluation / "summary.json")
        passed = selection.get("status") == INNER_PASS and selection.get("scientific_pass") is True
        if (code == 0) != passed:
            raise RuntimeError("slot-transport selection status/return mismatch")

        parents = [path.stem for path in sorted((TEACHER / "artifacts/export/teacher/selection").glob("*.zarr"))]
        required = [
            *[models / f"seed{seed}/{name}" for seed in range(3) for name in ("best.pt", "history.json", "summary.json")],
            *[models / f"seed{seed}/development_predictions/{parent}.npz" for seed in range(3) for parent in parents],
            evaluation / "summary.json", evaluation / "per_world_metrics.csv", evaluation / "figure_source.json",
            *[evaluation / f"gse_cardinality_conditioned_circular_slot_transport_selection_v1.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("slot-transport training evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen source changed")
        overall = PASS if passed else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    steps = sum(11370 for item in processes if item["stage"] == "training" and item["returncode"] == 0)
    inference = sum(261422 for item in processes if item["stage"] == "training" and item["returncode"] == 0)
    write_json(run / "metrics/summary.json", {
        "schema_version": "gse_cardinality_conditioned_circular_slot_transport_three_seed_training_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "subprocesses": processes, "selection": selection, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after), "optimizer_steps": steps,
        "model_inference_observations": inference, "c08_checkpoint_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "optimizer_steps": steps,
                      "decision": selection.get("decision"), "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
