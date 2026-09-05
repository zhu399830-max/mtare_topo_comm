#!/usr/bin/env python3
"""Train one explicit primitive-relation seed on fit and select on C07."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.representation.primitive_relation_losses import primitive_relation_losses
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationNet
from mtare_topo.representation.primitive_relation_training import (
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    WEIGHT_DECAY,
    numpy_batch_to_torch,
    staged_training_loss,
    training_stage,
)


EXPECTED_FIT_ROWS = 426_552
EXPECTED_C07_ROWS = 64_644
EXPECTED_FIT_TASKS = 180
EXPECTED_C07_TASKS = 30
EXPECTED_FIT_BATCHES = 26_736
EXPECTED_C07_BATCHES = 522
MAXIMUM_PROCESS_MEMORY_BYTES = 16 * 1024**3
LOSS_NAMES = (
    "primitive_set_parameters", "surface_reconstruction", "ray_free_space",
    "port_relations", "temporal_equivariance", "uncertainty_calibration", "total",
)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_memory_bytes() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) == os.getpid():
            return int(memory) * 1024**2
    raise RuntimeError("formal primitive training process is absent from nvidia-smi")


def _configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS deterministic workspace contract drift")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


@torch.no_grad()
def evaluate_losses(
    model: PrimitiveRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    sums = defaultdict(float)
    rows = 0
    batches = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE,
        seed=0,
        epoch=0,
        shuffle=False,
    ):
        batch = numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid,
            batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        losses = primitive_relation_losses(prediction, batch.targets, batch.range_valid)
        row_count = len(numpy_batch.range_valid)
        for name in LOSS_NAMES:
            value = float(losses[name].detach())
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite C07 loss: {name}")
            sums[name] += value * row_count
        rows += row_count
        batches += 1
    if rows != EXPECTED_C07_ROWS:
        raise RuntimeError(f"C07 row-count drift: {rows}")
    if batches != EXPECTED_C07_BATCHES:
        raise RuntimeError(f"C07 batch-count drift: {batches}")
    return {
        "rows": rows,
        "batches": batches,
        **{name: sums[name] / rows for name in LOSS_NAMES},
    }


def _save_checkpoint(
    path: Path,
    *,
    model: PrimitiveRelationNet,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch: int,
    training_metrics: dict,
    selection_metrics: dict,
) -> None:
    payload = {
        "schema_version": "primitive_relation_checkpoint_v1",
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_metrics": training_metrics,
        "selection_metrics": selection_metrics,
        "training_contract": {
            "epochs": TRAINING_EPOCHS,
            "training_batch_size": TRAINING_BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "fit_rows": EXPECTED_FIT_ROWS,
            "c07_rows": EXPECTED_C07_ROWS,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-sensor-root", required=True, type=Path)
    parser.add_argument("--fit-teacher-root", required=True, type=Path)
    parser.add_argument("--c07-sensor-root", required=True, type=Path)
    parser.add_argument("--c07-teacher-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAINING_BATCH_SIZE)
    parser.add_argument("--evaluation-batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    args = parser.parse_args()
    if (
        args.epochs != TRAINING_EPOCHS
        or args.batch_size != TRAINING_BATCH_SIZE
        or args.evaluation_batch_size != EVALUATION_BATCH_SIZE
        or args.learning_rate != LEARNING_RATE
        or args.weight_decay != WEIGHT_DECAY
    ):
        raise ValueError("formal primitive-relation training hyperparameter drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("primitive-relation seed output exists; overwrite forbidden")
    output.mkdir(parents=True)
    started = time.monotonic()
    _configure_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal primitive-relation training requires CUDA")
    device = torch.device("cuda")
    fit = PrimitiveRelationBatchLoader(args.fit_sensor_root, args.fit_teacher_root)
    c07 = PrimitiveRelationBatchLoader(args.c07_sensor_root, args.c07_teacher_root)
    if len(fit) != EXPECTED_FIT_ROWS or len(fit.task_names) != EXPECTED_FIT_TASKS:
        raise RuntimeError("fit population drift")
    if len(c07) != EXPECTED_C07_ROWS or len(c07.task_names) != EXPECTED_C07_TASKS:
        raise RuntimeError("C07 population drift")
    if any("_C07__" in name or "_C08__" in name for name in fit.task_names):
        raise RuntimeError("selection task leaked into fit loader")
    if any("_C07__" not in name for name in c07.task_names):
        raise RuntimeError("C07 loader contains a non-C07 task")
    _write_json(output / "data_inventory.json", {
        "fit_rows": len(fit), "fit_tasks": list(fit.task_names),
        "c07_rows": len(c07), "c07_tasks": list(c07.task_names),
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
    })

    model = PrimitiveRelationNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    history: list[dict] = []
    best_epoch = -1
    best_selection_loss = float("inf")
    optimizer_steps = 0
    peak_process_memory = 0
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(TRAINING_EPOCHS):
        epoch_started = time.monotonic()
        model.train()
        sums = defaultdict(float)
        rows = 0
        batches = 0
        for numpy_batch in fit.iter_epoch(
            batch_size=TRAINING_BATCH_SIZE,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        ):
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            families = primitive_relation_losses(prediction, batch.targets, batch.range_valid)
            objective = staged_training_loss(families, epoch=epoch)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("non-finite primitive-relation training objective")
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True,
            )
            optimizer.step()
            row_count = len(numpy_batch.range_valid)
            for name in LOSS_NAMES:
                sums[name] += float(families[name].detach()) * row_count
            sums["stage_objective"] += float(objective.detach()) * row_count
            sums["gradient_norm"] += float(gradient_norm.detach()) * row_count
            rows += row_count
            batches += 1
            optimizer_steps += 1
            if batches % 200 == 0:
                process_memory = _process_memory_bytes()
                peak_process_memory = max(peak_process_memory, process_memory)
                if process_memory > MAXIMUM_PROCESS_MEMORY_BYTES:
                    raise RuntimeError(
                        f"primitive training process memory {process_memory} exceeds 16 GiB",
                    )
        if rows != EXPECTED_FIT_ROWS:
            raise RuntimeError(f"fit epoch row-count drift: {rows}")
        if batches != EXPECTED_FIT_BATCHES:
            raise RuntimeError(f"fit epoch batch-count drift: {batches}")
        training_metrics = {
            "rows": rows,
            "batches": batches,
            "stage": training_stage(epoch),
            **{name: sums[name] / rows for name in (*LOSS_NAMES, "stage_objective", "gradient_norm")},
        }
        selection = evaluate_losses(model, c07, device=device)
        process_memory = _process_memory_bytes()
        peak_process_memory = max(peak_process_memory, process_memory)
        if process_memory > MAXIMUM_PROCESS_MEMORY_BYTES:
            raise RuntimeError(
                f"primitive validation process memory {process_memory} exceeds 16 GiB",
            )
        record = {
            "epoch": epoch,
            "training": training_metrics,
            "selection": selection,
            "epoch_seconds": time.monotonic() - epoch_started,
            "optimizer_steps_cumulative": optimizer_steps,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "process_memory_bytes": process_memory,
        }
        history.append(record)
        checkpoint = output / f"epoch_{epoch:02d}.pt"
        _save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            seed=args.seed,
            epoch=epoch,
            training_metrics=training_metrics,
            selection_metrics=selection,
        )
        if float(selection["total"]) < best_selection_loss:
            best_selection_loss = float(selection["total"])
            best_epoch = epoch
        _write_json(output / "history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    if best_epoch < 0:
        raise RuntimeError("no primitive-relation checkpoint selected")
    if optimizer_steps != EXPECTED_FIT_BATCHES * TRAINING_EPOCHS:
        raise RuntimeError(f"primitive optimizer-step drift: {optimizer_steps}")
    selected_source = output / f"epoch_{best_epoch:02d}.pt"
    selected_path = output / "selected.pt"
    shutil.copy2(selected_source, selected_path)
    summary = {
        "schema_version": "primitive_relation_seed_training_v1",
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": TRAINING_EPOCHS,
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_c07_selection_loss": best_selection_loss,
        "selected_checkpoint": selected_path.name,
        "fit_rows_per_epoch": EXPECTED_FIT_ROWS,
        "c07_rows_per_epoch": EXPECTED_C07_ROWS,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_process_memory_bytes": peak_process_memory,
        "duration_seconds": time.monotonic() - started,
        "history": history,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
