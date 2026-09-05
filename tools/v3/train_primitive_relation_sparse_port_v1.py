#!/usr/bin/env python3
"""Train one sparse-port primitive-relation seed on fit and select on C07."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_port_relation_losses,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_sparse_port_training import (
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
from train_primitive_relation_model_v1 import (
    EXPECTED_C07_BATCHES,
    EXPECTED_C07_ROWS,
    EXPECTED_C07_TASKS,
    EXPECTED_FIT_BATCHES,
    EXPECTED_FIT_ROWS,
    EXPECTED_FIT_TASKS,
    LOSS_NAMES,
    MAXIMUM_PROCESS_MEMORY_BYTES,
    _configure_determinism,
    _process_memory_bytes,
    _write_json,
)


EXPECTED_PARAMETERS = 2_629_870


@torch.no_grad()
def evaluate_losses(
    model: SparsePortRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    sums = defaultdict(float)
    rows = batches = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        batch = numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid,
            batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        losses = sparse_port_relation_losses(
            prediction, batch.targets, batch.range_valid,
        )
        row_count = len(numpy_batch.range_valid)
        for name in LOSS_NAMES:
            value = float(losses[name].detach())
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite sparse-port C07 loss: {name}")
            sums[name] += value * row_count
        rows += row_count
        batches += 1
    if rows != EXPECTED_C07_ROWS or batches != EXPECTED_C07_BATCHES:
        raise RuntimeError(f"sparse-port C07 population drift: {rows}/{batches}")
    return {
        "rows": rows, "batches": batches,
        **{name: sums[name] / rows for name in LOSS_NAMES},
    }


def _save_checkpoint(
    path: Path,
    *,
    model: SparsePortRelationNet,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch: int,
    training_metrics: dict,
    selection_metrics: dict,
) -> None:
    payload = {
        "schema_version": "primitive_relation_sparse_port_checkpoint_v1",
        "seed": seed, "epoch": epoch,
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
            "stages": [training_stage(value) for value in range(TRAINING_EPOCHS)],
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
        raise ValueError("formal sparse-port training hyperparameter drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("sparse-port seed output exists; overwrite forbidden")
    output.mkdir(parents=True)
    started = time.monotonic()
    _configure_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal sparse-port training requires CUDA")
    device = torch.device("cuda")
    fit = PrimitiveRelationBatchLoader(args.fit_sensor_root, args.fit_teacher_root)
    c07 = PrimitiveRelationBatchLoader(args.c07_sensor_root, args.c07_teacher_root)
    if len(fit) != EXPECTED_FIT_ROWS or len(fit.task_names) != EXPECTED_FIT_TASKS:
        raise RuntimeError("sparse-port fit population drift")
    if len(c07) != EXPECTED_C07_ROWS or len(c07.task_names) != EXPECTED_C07_TASKS:
        raise RuntimeError("sparse-port C07 population drift")
    if any("_C07__" in name or "_C08__" in name for name in fit.task_names):
        raise RuntimeError("selection task leaked into sparse-port fit loader")
    if any("_C07__" not in name for name in c07.task_names):
        raise RuntimeError("sparse-port C07 loader contains non-C07 task")
    _write_json(output / "data_inventory.json", {
        "fit_rows": len(fit), "fit_tasks": list(fit.task_names),
        "c07_rows": len(c07), "c07_tasks": list(c07.task_names),
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
    })

    model = SparsePortRelationNet().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"sparse-port parameter drift: {parameter_count}")
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
        rows = batches = 0
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
            families = sparse_port_relation_losses(
                prediction, batch.targets, batch.range_valid,
            )
            objective = staged_training_loss(families, epoch=epoch)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("non-finite sparse-port training objective")
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
                        f"sparse-port process memory {process_memory} exceeds 16 GiB"
                    )
        if rows != EXPECTED_FIT_ROWS or batches != EXPECTED_FIT_BATCHES:
            raise RuntimeError(f"sparse-port fit epoch drift: {rows}/{batches}")
        training_metrics = {
            "rows": rows, "batches": batches, "stage": training_stage(epoch),
            **{
                name: sums[name] / rows
                for name in (*LOSS_NAMES, "stage_objective", "gradient_norm")
            },
        }
        selection = evaluate_losses(model, c07, device=device)
        process_memory = _process_memory_bytes()
        peak_process_memory = max(peak_process_memory, process_memory)
        if process_memory > MAXIMUM_PROCESS_MEMORY_BYTES:
            raise RuntimeError(
                f"sparse-port validation memory {process_memory} exceeds 16 GiB"
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
        raise RuntimeError("no sparse-port checkpoint selected")
    expected_steps = EXPECTED_FIT_BATCHES * TRAINING_EPOCHS
    if optimizer_steps != expected_steps:
        raise RuntimeError(f"sparse-port optimizer-step drift: {optimizer_steps}")
    selected_source = output / f"epoch_{best_epoch:02d}.pt"
    selected_path = output / "selected.pt"
    shutil.copy2(selected_source, selected_path)
    summary = {
        "schema_version": "primitive_relation_sparse_port_seed_training_v1",
        "seed": args.seed,
        "parameter_count": parameter_count,
        "epochs": TRAINING_EPOCHS,
        "stages": [training_stage(value) for value in range(TRAINING_EPOCHS)],
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
