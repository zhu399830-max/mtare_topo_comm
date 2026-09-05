#!/usr/bin/env python3
"""Train one sensor-polar composition-anchor head with a frozen backbone."""

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

from mtare_topo.data.primitive_composition_anchor_batches import (
    CompositionAnchorBatchLoader,
)
from mtare_topo.representation.primitive_composition_anchor_model import (
    FrozenObservableCompositionAnchorNet,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    align_composition_anchor_targets,
    composition_anchor_numpy_batch_to_torch,
    composition_anchor_training_losses,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)
from train_primitive_relation_model_v1 import (
    _configure_determinism,
    _process_memory_bytes,
    _write_json,
)


TRAINING_EPOCHS = 3
TRAINING_BATCH_SIZE = 128
EVALUATION_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
EXPECTED_FIT_ROWS = 426_552
EXPECTED_FIT_TASKS = 180
EXPECTED_FIT_BATCHES = 3_411
EXPECTED_C07_ROWS = 64_644
EXPECTED_C07_TASKS = 30
EXPECTED_C07_BATCHES = 522
EXPECTED_BACKBONE_PARAMETERS = 2_635_631
EXPECTED_HEAD_PARAMETERS = 22_278
EXPECTED_TOTAL_PARAMETERS = 2_657_909
EXPECTED_HEAD_TENSORS = 8
MAXIMUM_GPU_PROCESS_BYTES = 16 * 1024**3
LOSS_NAMES = (
    "anchor_nll", "uncertainty_calibration", "compatibility",
    "overlap_hard_negative", "relation", "total",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_frozen_model(
    checkpoint_path: Path,
    *,
    seed: int,
) -> tuple[FrozenObservableCompositionAnchorNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity")
        != "dual_endpoint_observed"
    ):
        raise RuntimeError("composition-anchor source checkpoint identity drift")
    backbone = ObservableSparsePortRelationNet()
    backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.manual_seed(2_026_090_300 + seed)
    model = FrozenObservableCompositionAnchorNet(backbone)
    return model, checkpoint


def batch_losses(model, numpy_batch, *, device: torch.device):
    batch = composition_anchor_numpy_batch_to_torch(numpy_batch, device=device)
    prediction = model(
        batch.base.range_valid,
        batch.base.relative_translation_current_sensor_m,
        batch.base.relative_yaw_current_sensor_deg,
    )
    assignments = match_primitives(prediction.primitive, batch.base.targets)
    aligned = align_composition_anchor_targets(
        batch.anchor_current_sensor_m, batch.base.targets, assignments,
    )
    losses = composition_anchor_training_losses(prediction.composition, aligned)
    return losses, prediction, aligned


@torch.no_grad()
def evaluate_losses(
    model: FrozenObservableCompositionAnchorNet,
    loader: CompositionAnchorBatchLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    sums = defaultdict(float)
    rows = batches = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        losses, _, _ = batch_losses(model, numpy_batch, device=device)
        row_count = len(numpy_batch.base.base.range_valid)
        for name in LOSS_NAMES:
            value = float(losses[name].detach())
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite composition-anchor C07 loss: {name}")
            sums[name] += value * row_count
        rows += row_count
        batches += 1
    if rows != EXPECTED_C07_ROWS or batches != EXPECTED_C07_BATCHES:
        raise RuntimeError(f"composition-anchor C07 population drift: {rows}/{batches}")
    return {
        "rows": rows,
        "batches": batches,
        **{name: sums[name] / rows for name in LOSS_NAMES},
        "selection_loss": sums["total"] / rows,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: FrozenObservableCompositionAnchorNet,
    optimizer,
    seed: int,
    epoch: int,
    source_checkpoint: Path,
    training_metrics: dict,
    selection_metrics: dict,
) -> None:
    payload = {
        "schema_version": "primitive_composition_anchor_checkpoint_v1",
        "seed": seed,
        "epoch": epoch,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha(source_checkpoint),
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
            "selection_metric": "mean_C07_composition_anchor_total_loss",
            "teacher_attachment_validity": "dual_endpoint_observed",
            "anchor_target_frame": "current_sensor_xyz_yaw_only",
            "frozen_observable_backbone": True,
            "joint_loss_from_epoch_zero": True,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    for prefix in ("fit", "c07"):
        parser.add_argument(f"--{prefix}-sensor-root", required=True, type=Path)
        parser.add_argument(f"--{prefix}-teacher-root", required=True, type=Path)
        parser.add_argument(f"--{prefix}-observability-root", required=True, type=Path)
        parser.add_argument(f"--{prefix}-anchor-root", required=True, type=Path)
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
        raise ValueError("formal composition-anchor hyperparameter drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("composition-anchor seed output exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    _configure_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal composition-anchor training requires CUDA")
    device = torch.device("cuda")
    fit = CompositionAnchorBatchLoader(
        args.fit_sensor_root, args.fit_teacher_root,
        args.fit_observability_root, args.fit_anchor_root,
    )
    c07 = CompositionAnchorBatchLoader(
        args.c07_sensor_root, args.c07_teacher_root,
        args.c07_observability_root, args.c07_anchor_root,
    )
    if len(fit) != EXPECTED_FIT_ROWS or len(fit.task_names) != EXPECTED_FIT_TASKS:
        raise RuntimeError("composition-anchor fit population drift")
    if len(c07) != EXPECTED_C07_ROWS or len(c07.task_names) != EXPECTED_C07_TASKS:
        raise RuntimeError("composition-anchor C07 population drift")
    if any("_C07__" in name or "_C08__" in name for name in fit.task_names):
        raise RuntimeError("selection task leaked into composition-anchor fit")
    if any("_C07__" not in name for name in c07.task_names):
        raise RuntimeError("composition-anchor C07 loader contains non-C07 task")
    _write_json(output / "data_inventory.json", {
        "fit_rows": len(fit), "fit_tasks": list(fit.task_names),
        "c07_rows": len(c07), "c07_tasks": list(c07.task_names),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha(args.source_checkpoint),
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
    })

    model, source_payload = load_frozen_model(
        args.source_checkpoint, seed=args.seed,
    )
    model = model.to(device)
    all_parameters = list(model.parameters())
    trainable = [value for value in all_parameters if value.requires_grad]
    frozen = [value for value in all_parameters if not value.requires_grad]
    if (
        sum(value.numel() for value in all_parameters) != EXPECTED_TOTAL_PARAMETERS
        or sum(value.numel() for value in model.backbone.parameters()) != EXPECTED_BACKBONE_PARAMETERS
        or sum(value.numel() for value in trainable) != EXPECTED_HEAD_PARAMETERS
        or len(trainable) != EXPECTED_HEAD_TENSORS
    ):
        raise RuntimeError("composition-anchor parameter/freeze boundary drift")
    frozen_digest_before = _state_digest(model.backbone)
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    history = []
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
            optimizer.zero_grad(set_to_none=True)
            losses, _, _ = batch_losses(model, numpy_batch, device=device)
            objective = losses["total"]
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("non-finite composition-anchor training objective")
            objective.backward()
            if any(
                value.grad is None
                or not bool(torch.isfinite(value.grad).all())
                or not bool((value.grad != 0).any())
                for value in trainable
            ):
                raise RuntimeError("missing/non-finite/zero composition-anchor head gradient")
            if any(value.grad is not None for value in frozen):
                raise RuntimeError("frozen composition-anchor backbone received gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, GRADIENT_CLIP_NORM, error_if_nonfinite=True,
            )
            optimizer.step()
            row_count = len(numpy_batch.base.base.range_valid)
            for name in LOSS_NAMES:
                sums[name] += float(losses[name].detach()) * row_count
            sums["gradient_norm"] += float(gradient_norm.detach()) * row_count
            rows += row_count
            batches += 1
            optimizer_steps += 1
            if batches % 200 == 0:
                process_memory = _process_memory_bytes()
                peak_process_memory = max(peak_process_memory, process_memory)
                if process_memory > MAXIMUM_GPU_PROCESS_BYTES:
                    raise RuntimeError("composition-anchor GPU process memory exceeds 16 GiB")
        if rows != EXPECTED_FIT_ROWS or batches != EXPECTED_FIT_BATCHES:
            raise RuntimeError(f"composition-anchor fit epoch drift: {rows}/{batches}")
        training_metrics = {
            "rows": rows, "batches": batches,
            **{name: sums[name] / rows for name in (*LOSS_NAMES, "gradient_norm")},
        }
        selection = evaluate_losses(model, c07, device=device)
        process_memory = _process_memory_bytes()
        peak_process_memory = max(peak_process_memory, process_memory)
        if process_memory > MAXIMUM_GPU_PROCESS_BYTES:
            raise RuntimeError("composition-anchor C07 GPU memory exceeds 16 GiB")
        if _state_digest(model.backbone) != frozen_digest_before:
            raise RuntimeError("composition-anchor frozen backbone changed")
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
            training_metrics=training_metrics, selection_metrics=selection,
        )
        selection_loss = float(selection["selection_loss"])
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_epoch = epoch
        _write_json(output / "history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    expected_steps = EXPECTED_FIT_BATCHES * TRAINING_EPOCHS
    if optimizer_steps != expected_steps or best_epoch < 0:
        raise RuntimeError("composition-anchor optimizer-step/checkpoint selection drift")
    shutil.copy2(output / f"epoch_{best_epoch:02d}.pt", output / "selected.pt")
    frozen_digest_after = _state_digest(model.backbone)
    if frozen_digest_after != frozen_digest_before:
        raise RuntimeError("composition-anchor frozen state changed after training")
    summary = {
        "schema_version": "primitive_composition_anchor_seed_training_v1",
        "seed": args.seed,
        "parameter_count": EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters": EXPECTED_HEAD_PARAMETERS,
        "trainable_tensors": EXPECTED_HEAD_TENSORS,
        "frozen_parameters": EXPECTED_BACKBONE_PARAMETERS,
        "epochs": TRAINING_EPOCHS,
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_c07_selection_loss": best_selection_loss,
        "selected_checkpoint": "selected.pt",
        "source_checkpoint_sha256": _sha(args.source_checkpoint),
        "source_selected_epoch": int(source_payload["epoch"]),
        "frozen_state_sha256_before": frozen_digest_before,
        "frozen_state_sha256_after": frozen_digest_after,
        "fit_rows_per_epoch": EXPECTED_FIT_ROWS,
        "c07_rows_per_epoch": EXPECTED_C07_ROWS,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_process_memory_bytes": peak_process_memory,
        "duration_seconds": time.monotonic() - started,
        "history": history,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
