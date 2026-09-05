#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_STRUCTURED_POLAR_MULTIDEPTH_CAPACITY_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_MULTIDEPTH_CAPACITY_V1"
PYTHON = Path(
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
)
DATASET = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/dataset"
)
TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/export/teacher"
)
OLD_CAPACITY = PROJECT_ROOT / (
    "results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"
)
FREE_QUERY = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _run(command, log_path: Path, environment, timeout: int) -> int:
    with log_path.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if (
        run_dir.name != run_id
        or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED"
    ):
        raise RuntimeError("structured-polar capacity executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    evaluation: dict[str, object] = {}
    seed_summaries: list[dict[str, object]] = []
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncodes: dict[str, int] = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if (
            not validation.passed
            or card.get("status")
            != "APPROVED_FOR_SMOKE_AND_ONE_IMMUTABLE_GSE_STRUCTURED_POLAR_MULTIDEPTH_TRAINING_V1"
            or card.get("approval", {}).get("authorized_operations") != ["training"]
            or card.get("approval", {}).get("authorized_gates") != [3]
        ):
            raise RuntimeError(f"structured-polar training Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090 D":
            raise RuntimeError("formal structured-polar training GPU drift")
        environment_record = {
            "executable": str(PYTHON),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "zarr": zarr.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": ":4096:8",
        }
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "config/environment.json", environment_record)
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"},
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        )
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["OMP_NUM_THREADS"] = "4"
        model_dirs = [run_dir / f"artifacts/models/seed{seed}" for seed in (0, 1, 2)]
        training_commands = []
        for seed, model_dir in enumerate(model_dirs):
            training_commands.append(
                [
                    str(PYTHON),
                    str(PROJECT_ROOT / "tools/v3/train_gse_structured_polar_multidepth_event_v1.py"),
                    "--dataset-root",
                    str(DATASET),
                    "--teacher-root",
                    str(TEACHER),
                    "--output-dir",
                    str(model_dir),
                    "--seed",
                    str(seed),
                    "--epochs",
                    "8",
                    "--batch-size",
                    "128",
                    "--evaluation-batch-size",
                    "128",
                    "--learning-rate",
                    "3e-4",
                ]
            )
        evaluator = [
            str(PYTHON),
            str(PROJECT_ROOT / "tools/v3/evaluate_gse_structured_polar_multidepth_capacity_v1.py"),
            "--teacher-root",
            str(TEACHER),
        ]
        for model_dir in model_dirs:
            evaluator.extend(("--structured-model-dir", str(model_dir)))
        for seed in (0, 1, 2):
            evaluator.extend(
                (
                    "--free-query-model-dir",
                    str(FREE_QUERY / f"artifacts/models/seed{seed}"),
                    "--frozen-set-output",
                    str(OLD_CAPACITY / f"artifacts/models/seed{seed}/selection_outputs.npz"),
                )
            )
        evaluator.extend(
            (
                "--baseline-output",
                str(OLD_CAPACITY / "metrics/capacity/baseline_selection_outputs.npz"),
                "--output-dir",
                str(run_dir / "metrics/capacity"),
            )
        )
        write_json(run_dir / "config/commands.json", training_commands + [evaluator])
        for seed, (model_dir, command) in enumerate(zip(model_dirs, training_commands, strict=True)):
            code = _run(
                command,
                run_dir / f"logs/{seed:02d}_seed{seed}_training.log",
                environment,
                7200,
            )
            returncodes[f"seed{seed}"] = code
            if code != 0:
                raise RuntimeError(f"structured-polar seed {seed} training failure: {code}")
            summary = load_json(model_dir / "summary.json")
            if (
                summary.get("seed") != seed
                or summary.get("epochs") != 8
                or summary.get("optimizer_steps") != 9096
                or summary.get("trainable_parameters") != 172430
                or summary.get("fit_rows") != 142184
                or summary.get("selection_rows") != 45942
                or summary.get("smoke_limited") is not False
                or any(
                    summary.get(name) != 0
                    for name in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read")
                )
            ):
                raise RuntimeError(f"structured-polar seed {seed} summary drift")
            seed_summaries.append(summary)
        code = _run(
            evaluator,
            run_dir / "logs/03_capacity_evaluation.log",
            environment,
            3600,
        )
        returncodes["evaluation"] = code
        if code not in (0, 2):
            raise RuntimeError(f"structured-polar evaluator system failure: {code}")
        evaluation = load_json(run_dir / "metrics/capacity/summary.json")
        if (
            evaluation.get("status") not in (PASS, FAIL)
            or evaluation.get("population", {}).get("observations") != 45942
            or evaluation.get("population", {}).get("target_tokens") != 33145
            or evaluation.get("population", {}).get("second_depth_targets") != 79
            or any(
                evaluation.get(name) != 0
                for name in ("c09_worlds_read", "c10_worlds_read", "mtare_worlds_read")
            )
        ):
            raise RuntimeError("structured-polar capacity evaluation drift")
        required = [
            run_dir / "metrics/capacity/gse_structured_polar_multidepth_capacity_v1.png",
            run_dir / "metrics/capacity/gse_structured_polar_multidepth_capacity_v1.pdf",
            run_dir / "metrics/capacity/gse_structured_polar_multidepth_capacity_v1.svg",
            run_dir / "metrics/capacity/figure_source.json",
            run_dir / "metrics/capacity/capacity_table.csv",
            run_dir / "metrics/capacity/capacity_table.md",
        ]
        for model_dir in model_dirs:
            required.extend(
                (
                    model_dir / "best.pt",
                    model_dir / "selection_outputs.npz",
                    model_dir / "history.json",
                    model_dir / "summary.json",
                )
            )
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("structured-polar capacity evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("structured-polar capacity source changed")
        overall = str(evaluation["status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    optimizer_steps = sum(int(item.get("optimizer_steps", 0)) for item in seed_summaries)
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_structured_polar_multidepth_capacity_outer_v1",
            "overall_status": overall,
            "error": error,
            "returncodes": returncodes,
            "seed_summaries": seed_summaries,
            "evaluation": evaluation,
            "duration_seconds": time.monotonic() - started,
            "source_unchanged": bool(before and before == after),
            "optimizer_steps": optimizer_steps,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_id,
            "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
            "overall_status": overall,
            "error": error,
        },
    )
    entries = seal(run_dir)
    print(
        json.dumps(
            {
                "overall_status": overall,
                "error": error,
                "seal_entries": entries,
                "optimizer_steps": optimizer_steps,
            },
            indent=2,
        )
    )
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
