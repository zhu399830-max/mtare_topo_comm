#!/usr/bin/env python3
"""Train one seed of the train-only physics-guided GSE slope corrective."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_slope_corrective_dataset import (
    EXPECTED_FIT_SEQUENCES,
    EXPECTED_SELECTION_SEQUENCES,
    load_slope_corrective_partition,
)
from mtare_topo.representation.gse_slope_corrective import (
    PhysicsGuidedSlopeResidualNet,
    slope_corrective_loss,
)


PASS_STATUS = "PASS_GSE_SLOPE_CORRECTIVE_TRAINING_SEED_V1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _batches(count: int, batch_size: int, *, order: np.ndarray | None = None):
    indices = np.arange(count, dtype=np.int64) if order is None else order
    if indices.shape != (count,) or len(np.unique(indices)) != count:
        raise RuntimeError("corrective batch order is not a permutation")
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def evaluate(
    model: PhysicsGuidedSlopeResidualNet,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 4096,
    preserve_outputs: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    corrected_blocks: list[np.ndarray] = []
    scale_blocks: list[np.ndarray] = []
    count = len(arrays["features"])
    for indices in _batches(count, batch_size):
        features = torch.from_numpy(arrays["features"][indices]).to(device)
        prior = torch.from_numpy(arrays["prior_slope_deg"][indices]).to(device)
        output = model(features, prior)
        corrected_blocks.append(output["slope_deg"].float().cpu().numpy())
        scale_blocks.append(output["error_scale_deg"].float().cpu().numpy())
    corrected = np.concatenate(corrected_blocks).astype(np.float32, copy=False)
    scales = np.concatenate(scale_blocks).astype(np.float32, copy=False)
    target = arrays["target_slope_deg"].astype(np.float64, copy=False)
    prior = arrays["prior_slope_deg"].astype(np.float64, copy=False)
    current = arrays["current_slope_deg"].astype(np.float64, copy=False)
    corrected64 = corrected.astype(np.float64, copy=False)
    if not np.all(np.isfinite(corrected64)) or not np.all(np.isfinite(scales)):
        raise RuntimeError("corrective evaluation produced non-finite values")
    absolute = {
        "current": np.abs(current - target),
        "five_frame_prior": np.abs(prior - target),
        "corrected": np.abs(corrected64 - target),
    }
    parents = arrays["parent_id"]
    per_world = []
    for parent_id in sorted(np.unique(parents).tolist()):
        mask = parents == parent_id
        per_world.append(
            {
                "parent_id": str(parent_id),
                "sequences": int(mask.sum()),
                "current_mae_deg": float(absolute["current"][mask].mean()),
                "five_frame_prior_mae_deg": float(absolute["five_frame_prior"][mask].mean()),
                "corrected_mae_deg": float(absolute["corrected"][mask].mean()),
            }
        )
    per_family = []
    family_ids = np.asarray([str(value).rsplit("_C", 1)[0] for value in parents], dtype="U64")
    for family in sorted(np.unique(family_ids).tolist()):
        mask = family_ids == family
        prior_mae = float(absolute["five_frame_prior"][mask].mean())
        corrected_mae = float(absolute["corrected"][mask].mean())
        per_family.append(
            {
                "family": str(family),
                "sequences": int(mask.sum()),
                "five_frame_prior_mae_deg": prior_mae,
                "corrected_mae_deg": corrected_mae,
                "relative_improvement": (prior_mae - corrected_mae) / prior_mae,
            }
        )
    prior_mae = float(absolute["five_frame_prior"].mean())
    corrected_mae = float(absolute["corrected"].mean())
    metrics = {
        "sequences": count,
        "current_frame_mae_deg": float(absolute["current"].mean()),
        "five_frame_prior_mae_deg": prior_mae,
        "corrected_mae_deg": corrected_mae,
        "relative_improvement_over_five_frame_prior": (prior_mae - corrected_mae) / prior_mae,
        "mean_predicted_error_scale_deg": float(scales.mean()),
        "mean_absolute_scale_error_deg": float(np.abs(scales.astype(np.float64) - absolute["corrected"]).mean()),
        "per_world": per_world,
        "per_topology_family": per_family,
    }
    outputs = None
    if preserve_outputs:
        outputs = {
            "global_sequence_index": arrays["global_sequence_index"].astype(np.int64),
            "parent_id": parents.astype("U64"),
            "target_slope_deg": target.astype(np.float32),
            "current_slope_deg": current.astype(np.float32),
            "five_frame_prior_slope_deg": prior.astype(np.float32),
            "corrected_slope_deg": corrected,
            "predicted_error_scale_deg": scales,
        }
    return metrics, outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if (
        args.seed not in (0, 1, 2)
        or args.epochs != 50
        or args.batch_size != 1024
        or args.learning_rate != 1e-3
        or args.weight_decay != 1e-4
        or args.patience != 8
    ):
        raise ValueError("formal slope corrective optimization contract drift")
    if not torch.cuda.is_available():
        raise RuntimeError("formal slope corrective training requires the frozen CUDA environment")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    fit = load_slope_corrective_partition(args.cache_dir, "fit")
    selection = load_slope_corrective_partition(args.cache_dir, "selection")
    if len(fit["features"]) != EXPECTED_FIT_SEQUENCES or len(selection["features"]) != EXPECTED_SELECTION_SEQUENCES:
        raise RuntimeError("formal slope corrective sample count drift")
    model = PhysicsGuidedSlopeResidualNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    initial_metrics, _ = evaluate(model, selection, device)
    best_mae = float(initial_metrics["corrected_mae_deg"])
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    started = time.monotonic()
    torch.save(
        {
            "schema_version": "gse_slope_corrective_checkpoint_v1",
            "model": model.state_dict(),
            "seed": args.seed,
            "epoch": 0,
            "selection_metric": "selection_corrected_slope_mae_deg",
            "selection_value": best_mae,
            "config": vars(args),
        },
        output_dir / "best.pt",
    )
    initial_record = {"epoch": 0, "train_loss": None, "selection": initial_metrics, "optimizer_steps_total": 0}
    history.append(initial_record)
    (output_dir / "epoch_metrics.jsonl").write_text(
        json.dumps(initial_record, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    generator = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(fit["features"])).astype(np.int64, copy=False)
        sums = {"total": 0.0, "slope": 0.0, "uncertainty": 0.0, "residual_regularization": 0.0}
        seen = 0
        epoch_started = time.monotonic()
        for indices in _batches(len(order), args.batch_size, order=order):
            features = torch.from_numpy(fit["features"][indices]).to(device)
            prior = torch.from_numpy(fit["prior_slope_deg"][indices]).to(device)
            target = torch.from_numpy(fit["target_slope_deg"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, prior)
            losses = slope_corrective_loss(
                output,
                target,
                maximum_residual_deg=model.config.maximum_residual_deg,
            )
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite slope corrective gradient")
            optimizer.step()
            optimizer_steps += 1
            size = len(indices)
            seen += size
            for key in sums:
                sums[key] += float(losses[key].detach().cpu()) * size
        if seen != EXPECTED_FIT_SEQUENCES:
            raise RuntimeError("slope corrective epoch did not visit every fit sequence exactly once")
        selection_metrics, _ = evaluate(model, selection, device)
        record = {
            "epoch": epoch,
            "train_loss": {key: value / seen for key, value in sums.items()},
            "selection": selection_metrics,
            "duration_seconds": time.monotonic() - epoch_started,
            "optimizer_steps_total": optimizer_steps,
        }
        history.append(record)
        with (output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        score = float(selection_metrics["corrected_mae_deg"])
        if score < best_mae - 1e-8:
            best_mae = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema_version": "gse_slope_corrective_checkpoint_v1",
                    "model": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "selection_metric": "selection_corrected_slope_mae_deg",
                    "selection_value": score,
                    "config": vars(args),
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= args.patience:
            break
    torch.save(
        {
            "schema_version": "gse_slope_corrective_checkpoint_v1",
            "model": model.state_dict(),
            "seed": args.seed,
            "epoch": history[-1]["epoch"],
            "config": vars(args),
        },
        output_dir / "last.pt",
    )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics, outputs = evaluate(model, selection, device, preserve_outputs=True)
    if outputs is None or not math.isclose(float(final_metrics["corrected_mae_deg"]), best_mae, abs_tol=1e-7):
        raise RuntimeError("best slope corrective checkpoint reproduction drift")
    np.savez_compressed(output_dir / "selection_outputs.npz", **outputs)
    (output_dir / "best_selection_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "gse_slope_corrective_training_seed_v1",
        "overall_status": PASS_STATUS,
        "seed": args.seed,
        "epochs_completed": len(history) - 1,
        "best_epoch": best_epoch,
        "selection_metric": "selection_corrected_slope_mae_deg",
        "fit_sequences_per_epoch": EXPECTED_FIT_SEQUENCES,
        "selection_sequences_per_evaluation": EXPECTED_SELECTION_SEQUENCES,
        "optimizer_steps": optimizer_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_selection": final_metrics,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
