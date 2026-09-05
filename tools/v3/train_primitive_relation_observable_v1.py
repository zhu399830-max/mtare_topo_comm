#!/usr/bin/env python3
"""Train one endpoint-observable relation seed with frozen V2 geometry."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.representation.primitive_relation_observable_initialization import (
    initialize_observable_relation_from_checkpoint,
    is_relation_trainable_parameter,
    set_observable_relation_training_mode,
)
from mtare_topo.representation.primitive_relation_observable_schedule import (
    EVALUATION_BATCH_SIZE, GRADIENT_CLIP_NORM, LEARNING_RATE,
    TRAINING_BATCH_SIZE, TRAINING_EPOCHS, WEIGHT_DECAY,
    relation_training_objective, training_stage,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
    observable_primitive_relation_losses,
)
from train_primitive_relation_model_v1 import (
    EXPECTED_C07_BATCHES, EXPECTED_C07_ROWS, EXPECTED_C07_TASKS,
    EXPECTED_FIT_BATCHES, EXPECTED_FIT_ROWS, EXPECTED_FIT_TASKS,
    LOSS_NAMES, MAXIMUM_PROCESS_MEMORY_BYTES, _configure_determinism,
    _process_memory_bytes, _write_json,
)


EXPECTED_PARAMETERS = 2_635_631
EXPECTED_PARAMETER_TENSORS = 197
EXPECTED_TRAINABLE_PARAMETERS = 742_149
EXPECTED_TRAINABLE_TENSORS = 64


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if is_relation_trainable_parameter(name):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_losses(model, loader, *, device: torch.device) -> dict[str, float | int]:
    model.eval()
    sums = defaultdict(float)
    rows = batches = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid, batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        losses = observable_primitive_relation_losses(
            prediction, batch.targets, batch.range_valid,
        )
        row_count = len(numpy_batch.base.range_valid)
        for name in LOSS_NAMES:
            value = float(losses[name].detach())
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite observable C07 loss: {name}")
            sums[name] += value * row_count
        relation_selection = float(relation_training_objective(losses).detach())
        if not np.isfinite(relation_selection):
            raise RuntimeError("non-finite observable C07 relation selection loss")
        sums["relation_selection_loss"] += relation_selection * row_count
        rows += row_count
        batches += 1
    if rows != EXPECTED_C07_ROWS or batches != EXPECTED_C07_BATCHES:
        raise RuntimeError(f"observable C07 population drift: {rows}/{batches}")
    return {
        "rows": rows, "batches": batches,
        **{name: sums[name] / rows for name in LOSS_NAMES},
        "relation_selection_loss": sums["relation_selection_loss"] / rows,
    }


def _save_checkpoint(
    path: Path, *, model, optimizer, seed: int, epoch: int,
    source_checkpoint: Path, initialization, training_metrics: dict,
    selection_metrics: dict,
) -> None:
    payload = {
        "schema_version": "primitive_relation_observable_checkpoint_v1",
        "seed": seed, "epoch": epoch,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha(source_checkpoint),
        "initialization_report": initialization.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_metrics": training_metrics,
        "selection_metrics": selection_metrics,
        "training_contract": {
            "epochs": TRAINING_EPOCHS,
            "training_batch_size": TRAINING_BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "fit_rows": EXPECTED_FIT_ROWS, "c07_rows": EXPECTED_C07_ROWS,
            "stages": [training_stage(value) for value in range(TRAINING_EPOCHS)],
            "selection_metric": "mean(port_relations,uncertainty_calibration)",
            "teacher_attachment_validity": "dual_endpoint_observed",
            "frozen_geometry_temporal": True,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-sensor-root", required=True, type=Path)
    parser.add_argument("--fit-teacher-root", required=True, type=Path)
    parser.add_argument("--fit-sidecar-root", required=True, type=Path)
    parser.add_argument("--c07-sensor-root", required=True, type=Path)
    parser.add_argument("--c07-teacher-root", required=True, type=Path)
    parser.add_argument("--c07-sidecar-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
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
        raise ValueError("formal observable relation hyperparameter drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("observable relation seed output exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    _configure_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal observable relation training requires CUDA")
    device = torch.device("cuda")
    fit = ObservablePrimitiveRelationBatchLoader(
        args.fit_sensor_root, args.fit_teacher_root, args.fit_sidecar_root,
    )
    c07 = ObservablePrimitiveRelationBatchLoader(
        args.c07_sensor_root, args.c07_teacher_root, args.c07_sidecar_root,
    )
    if len(fit) != EXPECTED_FIT_ROWS or len(fit.task_names) != EXPECTED_FIT_TASKS:
        raise RuntimeError("observable fit population drift")
    if len(c07) != EXPECTED_C07_ROWS or len(c07.task_names) != EXPECTED_C07_TASKS:
        raise RuntimeError("observable C07 population drift")
    if any("_C07__" in name or "_C08__" in name for name in fit.task_names):
        raise RuntimeError("selection task leaked into observable fit")
    if any("_C07__" not in name for name in c07.task_names):
        raise RuntimeError("observable C07 loader contains non-C07 task")
    _write_json(output / "data_inventory.json", {
        "fit_rows": len(fit), "fit_tasks": list(fit.task_names),
        "c07_rows": len(c07), "c07_tasks": list(c07.task_names),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha(args.source_checkpoint),
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
    })

    model, initialization = initialize_observable_relation_from_checkpoint(
        args.source_checkpoint, initialization_seed=202609020 + args.seed,
    )
    model = model.to(device)
    parameters = sum(value.numel() for value in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    trainable = [value for value in model.parameters() if value.requires_grad]
    frozen = [value for value in model.parameters() if not value.requires_grad]
    if (
        parameters != EXPECTED_PARAMETERS
        or parameter_tensors != EXPECTED_PARAMETER_TENSORS
        or sum(value.numel() for value in trainable) != EXPECTED_TRAINABLE_PARAMETERS
        or len(trainable) != EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError("observable parameter/freeze boundary drift")
    frozen_digest_before = _frozen_state_digest(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    history: list[dict] = []
    best_epoch = -1
    best_selection_loss = float("inf")
    optimizer_steps = 0
    peak_process_memory = 0
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(TRAINING_EPOCHS):
        epoch_started = time.monotonic()
        set_observable_relation_training_mode(model)
        sums = defaultdict(float)
        rows = batches = 0
        for numpy_batch in fit.iter_epoch(
            batch_size=TRAINING_BATCH_SIZE, seed=args.seed,
            epoch=epoch, shuffle=True,
        ):
            batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.range_valid, batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            families = observable_primitive_relation_losses(
                prediction, batch.targets, batch.range_valid,
            )
            objective = relation_training_objective(families)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("non-finite observable training objective")
            objective.backward()
            if any(
                value.grad is None or not bool(torch.isfinite(value.grad).all())
                for value in trainable
            ):
                raise RuntimeError("missing/non-finite observable trainable gradient")
            if any(value.grad is not None for value in frozen):
                raise RuntimeError("frozen geometry/temporal parameter received gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, GRADIENT_CLIP_NORM, error_if_nonfinite=True,
            )
            optimizer.step()
            row_count = len(numpy_batch.base.range_valid)
            for name in LOSS_NAMES:
                sums[name] += float(families[name].detach()) * row_count
            sums["relation_objective"] += float(objective.detach()) * row_count
            sums["gradient_norm"] += float(gradient_norm.detach()) * row_count
            rows += row_count
            batches += 1
            optimizer_steps += 1
            if batches % 200 == 0:
                process_memory = _process_memory_bytes()
                peak_process_memory = max(peak_process_memory, process_memory)
                if process_memory > MAXIMUM_PROCESS_MEMORY_BYTES:
                    raise RuntimeError("observable GPU process memory exceeds 16 GiB")
        if rows != EXPECTED_FIT_ROWS or batches != EXPECTED_FIT_BATCHES:
            raise RuntimeError(f"observable fit epoch drift: {rows}/{batches}")
        training_metrics = {
            "rows": rows, "batches": batches, "stage": training_stage(epoch),
            **{
                name: sums[name] / rows
                for name in (*LOSS_NAMES, "relation_objective", "gradient_norm")
            },
        }
        selection = evaluate_losses(model, c07, device=device)
        process_memory = _process_memory_bytes()
        peak_process_memory = max(peak_process_memory, process_memory)
        if process_memory > MAXIMUM_PROCESS_MEMORY_BYTES:
            raise RuntimeError("observable validation GPU memory exceeds 16 GiB")
        if _frozen_state_digest(model) != frozen_digest_before:
            raise RuntimeError("frozen geometry/temporal state changed")
        record = {
            "epoch": epoch, "training": training_metrics,
            "selection": selection,
            "epoch_seconds": time.monotonic() - epoch_started,
            "optimizer_steps_cumulative": optimizer_steps,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_process_memory_bytes": process_memory,
        }
        history.append(record)
        _save_checkpoint(
            output / f"epoch_{epoch:02d}.pt", model=model,
            optimizer=optimizer, seed=args.seed, epoch=epoch,
            source_checkpoint=args.source_checkpoint,
            initialization=initialization, training_metrics=training_metrics,
            selection_metrics=selection,
        )
        selection_loss = float(selection["relation_selection_loss"])
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_epoch = epoch
        _write_json(output / "history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    expected_steps = EXPECTED_FIT_BATCHES * TRAINING_EPOCHS
    if optimizer_steps != expected_steps or best_epoch < 0:
        raise RuntimeError("observable optimizer-step/checkpoint selection drift")
    shutil.copy2(output / f"epoch_{best_epoch:02d}.pt", output / "selected.pt")
    frozen_digest_after = _frozen_state_digest(model)
    if frozen_digest_after != frozen_digest_before:
        raise RuntimeError("frozen state changed after observable training")
    summary = {
        "schema_version": "primitive_relation_observable_seed_training_v1",
        "seed": args.seed, "parameter_count": parameters,
        "parameter_tensors": parameter_tensors,
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "trainable_tensors": EXPECTED_TRAINABLE_TENSORS,
        "frozen_parameters": parameters - EXPECTED_TRAINABLE_PARAMETERS,
        "epochs": TRAINING_EPOCHS,
        "stages": [training_stage(value) for value in range(TRAINING_EPOCHS)],
        "optimizer_steps": optimizer_steps, "best_epoch": best_epoch,
        "best_c07_relation_selection_loss": best_selection_loss,
        "selected_checkpoint": "selected.pt",
        "source_checkpoint_sha256": _sha(args.source_checkpoint),
        "initialization_report": initialization.__dict__,
        "frozen_state_sha256_before": frozen_digest_before,
        "frozen_state_sha256_after": frozen_digest_after,
        "fit_rows_per_epoch": EXPECTED_FIT_ROWS,
        "c07_rows_per_epoch": EXPECTED_C07_ROWS,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_process_memory_bytes": peak_process_memory,
        "duration_seconds": time.monotonic() - started, "history": history,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
