#!/usr/bin/env python3
"""Train one 93,638-parameter spatial event set decoder on frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_spatial_event_set_cache import (
    SpatialEventSetFeatureCache,
    fit_only_loss_weights,
    load_spatial_event_teacher,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import select_fixed_grid_threshold
from mtare_topo.representation.gse_spatial_event_set import (
    SpatialEventSetDecoder,
    SpatialEventSetLossWeights,
    spatial_event_set_loss,
)


THRESHOLD_GRID = tuple(round(value * 0.05, 2) for value in range(1, 20))
LOSS_WEIGHTS = SpatialEventSetLossWeights(
    presence=1.0,
    event_type=1.0,
    position=5.0,
    descriptor=0.0,
    uncertainty=0.01,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roll_features_and_targets(features, targets, shifts):
    import torch

    directional = features["directional"]
    bins = directional.shape[-1]
    source = (
        torch.arange(bins, device=directional.device)[None, :] - shifts[:, None]
    ) % bins
    rolled = torch.gather(
        directional,
        2,
        source[:, None, :].expand(-1, directional.shape[1], -1),
    )
    angle = shifts.to(dtype=directional.dtype) * (2.0 * torch.pi / bins)
    cosine = torch.cos(angle)[:, None]
    sine = torch.sin(angle)[:, None]
    xyz = targets["event_relative_xyz_m"]
    rotated = xyz.clone()
    rotated[..., 0] = cosine * xyz[..., 0] - sine * xyz[..., 1]
    rotated[..., 1] = sine * xyz[..., 0] + cosine * xyz[..., 1]
    result_targets = dict(targets)
    result_targets["event_relative_xyz_m"] = rotated
    return {"context": features["context"], "directional": rolled}, result_targets


def _batch(cache, teacher, rows: np.ndarray, device):
    import torch

    feature_values = cache.gather(rows)
    target_values = teacher.targets(rows)
    features = {
        "context": torch.from_numpy(feature_values["context"]).to(device),
        "directional": torch.from_numpy(feature_values["directional"]).to(device),
    }
    targets = {
        name: torch.from_numpy(value).to(device)
        for name, value in target_values.items()
    }
    return features, targets


def _evaluate_loss(decoder, cache, teacher, rows, device, batch_size, loss_weight_values):
    import torch

    decoder.eval()
    totals = {name: 0.0 for name in (
        "total", "presence", "event_type", "position", "descriptor", "uncertainty"
    )}
    weight = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            features, targets = _batch(cache, teacher, selected, device)
            outputs = decoder(features)
            losses = spatial_event_set_loss(
                outputs,
                targets,
                weights=LOSS_WEIGHTS,
                presence_positive_weight=loss_weight_values["presence_positive_weight"],
                event_type_class_weights=torch.tensor(
                    loss_weight_values["event_type_class_weights"], device=device
                ),
            )
            for name in totals:
                totals[name] += float(losses[name].cpu()) * len(selected)
            weight += len(selected)
    return {name: value / weight for name, value in totals.items()}


def _infer(decoder, cache, rows, device, batch_size):
    import torch

    confidence = []
    event_type = []
    relative_xyz = []
    uncertainty = []
    decoder.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            values = cache.gather(selected)
            outputs = decoder(
                {
                    "context": torch.from_numpy(values["context"]).to(device),
                    "directional": torch.from_numpy(values["directional"]).to(device),
                }
            )
            confidence.append(outputs["event_confidence"].cpu().numpy().astype(np.float32))
            event_type.append(outputs["event_type_logits"].argmax(dim=-1).cpu().numpy().astype(np.int8))
            relative_xyz.append(outputs["event_relative_xyz_m"].cpu().numpy().astype(np.float32))
            uncertainty.append(outputs["event_uncertainty_m"].cpu().numpy().astype(np.float32))
    return {
        "confidence": np.concatenate(confidence),
        "event_type": np.concatenate(event_type),
        "relative_xyz_m": np.concatenate(relative_xyz),
        "uncertainty_m": np.concatenate(uncertainty),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal spatial event set training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    teacher = load_spatial_event_teacher(args.teacher_root.resolve())
    cache = SpatialEventSetFeatureCache(args.feature_cache.resolve())
    if not np.array_equal(cache.global_sequence_index, teacher.global_sequence_index):
        raise RuntimeError("feature cache/Teacher join drift")
    if cache.manifest.get("seed") != args.seed:
        raise RuntimeError("feature cache seed drift")
    loss_weight_values = fit_only_loss_weights(teacher)
    fit_rows = teacher.fit_rows
    selection_rows = teacher.selection_rows
    decoder = SpatialEventSetDecoder().to(device)
    parameter_count = sum(parameter.numel() for parameter in decoder.parameters())
    if parameter_count != 93_638:
        raise RuntimeError("spatial event decoder parameter drift")
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    best_selection_loss = float("inf")
    best_epoch = -1
    optimizer_steps = 0
    common = {
        "schema_version": "gse_spatial_event_set_checkpoint_v1",
        "seed": args.seed,
        "feature_cache_manifest_sha256": _sha256(args.feature_cache / "manifest.json"),
        "teacher_contract": "fixed16_native_los_c01_c08_v1",
        "trainable_parameters": parameter_count,
        "descriptor_loss_weight": 0.0,
    }
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 10_000 + epoch)
        order = fit_rows[rng.permutation(len(fit_rows))]
        decoder.train()
        totals = {name: 0.0 for name in (
            "total", "presence", "event_type", "position", "descriptor", "uncertainty"
        )}
        observed = 0
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            features, targets = _batch(cache, teacher, rows, device)
            shifts = torch.from_numpy(
                rng.integers(0, 180, size=len(rows), dtype=np.int64)
            ).to(device)
            features, targets = _roll_features_and_targets(features, targets, shifts)
            outputs = decoder(features)
            losses = spatial_event_set_loss(
                outputs,
                targets,
                weights=LOSS_WEIGHTS,
                presence_positive_weight=loss_weight_values["presence_positive_weight"],
                event_type_class_weights=torch.tensor(
                    loss_weight_values["event_type_class_weights"], device=device
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            for name in totals:
                totals[name] += float(losses[name].detach().cpu()) * len(rows)
            observed += len(rows)
        selection_loss = _evaluate_loss(
            decoder,
            cache,
            teacher,
            selection_rows,
            device,
            args.evaluation_batch_size,
            loss_weight_values,
        )
        record = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "fit_loss": {name: value / observed for name, value in totals.items()},
            "selection_loss": selection_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selection_loss["total"] < best_selection_loss:
            best_selection_loss = selection_loss["total"]
            best_epoch = epoch
            torch.save(
                {**common, "epoch": epoch, "decoder": decoder.state_dict(), "selection_loss": selection_loss},
                output / "best.pt",
            )

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    decoder.load_state_dict(best["decoder"], strict=True)
    predictions = _infer(
        decoder, cache, selection_rows, device, args.evaluation_batch_size
    )
    targets = teacher.targets(selection_rows)
    best_metrics, threshold_records = select_fixed_grid_threshold(
        predictions, targets, THRESHOLD_GRID, maximum_error_m=4.0
    )
    np.savez_compressed(
        output / "selection_outputs.npz",
        observation_row=selection_rows,
        global_sequence_index=teacher.global_sequence_index[selection_rows],
        confidence=predictions["confidence"],
        event_type=predictions["event_type"],
        relative_xyz_m=predictions["relative_xyz_m"],
        uncertainty_m=predictions["uncertainty_m"],
    )
    summary = {
        "schema_version": "gse_spatial_event_set_seed_training_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_selection_loss": best_selection_loss,
        "selected_threshold_metrics": best_metrics,
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_records": threshold_records,
        "fit_rows": len(fit_rows),
        "selection_rows": len(selection_rows),
        "optimizer_steps": optimizer_steps,
        "encoder_optimizer_steps": 0,
        "trainable_parameters": parameter_count,
        "loss_weights": {
            "presence": LOSS_WEIGHTS.presence,
            "event_type": LOSS_WEIGHTS.event_type,
            "position": LOSS_WEIGHTS.position,
            "descriptor": LOSS_WEIGHTS.descriptor,
            "uncertainty": LOSS_WEIGHTS.uncertainty,
        },
        "fit_only_frequency_weights": loss_weight_values,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
