#!/usr/bin/env python3
"""Train one twelve-frame causal episode detector from a frozen scan cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_causal_episode_cache import CausalEpisodeEmbeddingDataset
from mtare_topo.data.gse_causal_episode_sampler import EpisodePreservingBatchSampler
from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_causal_event_triggers,
    select_structural_threshold,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    CausalEpisodeDetector,
    causal_episode_multiple_instance_loss,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _collate(samples: list[dict]):
    import torch

    result = {}
    for key in ("pooled", "directional", "history_mask", "baseline_event_logits"):
        result[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
    for key in ("event_index", "episode_id", "boundary_offset_m", "boundary_valid", "observation_row"):
        result[key] = torch.from_numpy(np.asarray([sample[key] for sample in samples]))
    return result


def _forward(model, batch, device):
    import torch

    return model(
        batch["pooled"].to(device=device, dtype=torch.float32),
        batch["directional"].to(device=device, dtype=torch.float32),
        batch["history_mask"].to(device=device),
        batch["baseline_event_logits"].to(device=device, dtype=torch.float32),
    )


def _infer(model, dataset, *, device, batch_size: int) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate, pin_memory=False,
    )
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            values = _forward(model, batch, device)
            output["probability"].append(values["event_probability"].cpu().numpy())
            output["boundary_offset_m"].append(values["boundary_offset_m"].cpu().numpy())
            output["uncertainty"].append(values["uncertainty"].cpu().numpy())
            output["observation_row"].append(batch["observation_row"].numpy())
    return {key: np.concatenate(chunks) for key, chunks in output.items()}


def _episode_mil_score(
    probability: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    predicted_boundary: np.ndarray,
    boundary_target: np.ndarray,
    boundary_valid: np.ndarray,
) -> dict[str, float]:
    eps = 1e-8
    structural = 1.0 - probability[:, 0]
    corridor = event_target == 0
    negative = float(-np.log(np.clip(1.0 - structural[corridor], eps, 1.0)).mean())
    terms = []
    for episode in np.unique(episode_id[episode_id >= 0]):
        rows = episode_id == episode
        event = int(np.unique(event_target[rows]).item())
        terms.append(-math.log(float(np.clip(np.max(probability[rows, event]), eps, 1.0))))
    positive = float(np.mean(terms))
    if np.any(boundary_valid):
        difference = np.abs(predicted_boundary[boundary_valid] - boundary_target[boundary_valid]) / 11.0
        boundary = float(np.mean(np.where(difference < 1.0, .5 * difference**2, difference - .5)))
        boundary_mae = float(np.mean(np.abs(
            predicted_boundary[boundary_valid] - boundary_target[boundary_valid]
        )))
    else:
        boundary = 0.0
        boundary_mae = 0.0
    return {
        "corridor_negative": negative,
        "positive_episode_joint": positive,
        "boundary_offset": boundary,
        "boundary_mae_m": boundary_mae,
        "total": negative + positive + boundary,
    }


def _frame_metrics(probability: np.ndarray, target: np.ndarray) -> dict:
    predicted = np.argmax(probability, axis=1)
    per_event = {}
    f1 = []
    for index, name in enumerate(EVENT_NAMES):
        true_positive = int(np.sum((predicted == index) & (target == index)))
        predicted_count = int(np.sum(predicted == index))
        target_count = int(np.sum(target == index))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / target_count if target_count else 0.0
        value = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(value)
        per_event[name] = {
            "precision": precision, "recall": recall, "f1": value,
            "target_rows": target_count, "predicted_rows": predicted_count,
        }
    return {"accuracy": float(np.mean(predicted == target)), "macro_f1": float(np.mean(f1)), "per_event": per_event}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--baseline-features", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("detector output already exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)

    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("formal causal episode training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != args.seed or manifest.get("causal_observations") != 188126:
        raise RuntimeError("cache seed/population drift")
    partition = np.load(args.cache_dir / "partition_code.npy")
    fit_indices = np.flatnonzero(partition == 0)
    selection_indices = np.flatnonzero(partition == 1)
    if len(fit_indices) != 142184 or len(selection_indices) != 45942:
        raise RuntimeError("fit/selection count drift")
    fit = CausalEpisodeEmbeddingDataset(args.cache_dir, args.baseline_features, fit_indices)
    selection = CausalEpisodeEmbeddingDataset(
        args.cache_dir, args.baseline_features, selection_indices
    )
    sampler = EpisodePreservingBatchSampler(
        fit.episode_id, batch_size=args.batch_size, seed=args.seed
    )
    loader = DataLoader(
        fit, batch_sampler=sampler, num_workers=0, collate_fn=_collate, pin_memory=False
    )
    teacher = _read_jsonl(args.teacher.resolve())
    if len(teacher) != 188126:
        raise RuntimeError("training Teacher count drift")
    traversal = np.asarray([str(row["traversal_id"]) for row in teacher])
    sequence = np.asarray([int(row["sequence_index"]) for row in teacher], dtype=np.int64)
    event_all = np.load(args.cache_dir / "event_index.npy")
    episode_all = np.load(args.cache_dir / "episode_id.npy")
    boundary_target_all = np.load(args.cache_dir / "boundary_offset_m.npy")
    boundary_valid_all = np.load(args.cache_dir / "boundary_valid.npy")

    model = CausalEpisodeDetector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_score = math.inf
    best_epoch = -1
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        train_sums = defaultdict(float)
        steps = 0
        for batch in loader:
            outputs = _forward(model, batch, device)
            losses = causal_episode_multiple_instance_loss(
                outputs,
                batch["event_index"].to(device=device, dtype=torch.int64),
                batch["episode_id"].to(device=device, dtype=torch.int64),
                boundary_offset_target_m=batch["boundary_offset_m"].to(device=device, dtype=torch.float32),
                boundary_offset_valid=batch["boundary_valid"].to(device=device, dtype=torch.bool),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in losses.items():
                train_sums[key] += float(value.detach().cpu())
            steps += 1
        inferred = _infer(
            model, selection, device=device, batch_size=args.evaluation_batch_size
        )
        if not np.array_equal(inferred["observation_row"], selection_indices):
            raise RuntimeError("selection inference row order drift")
        selection_score = _episode_mil_score(
            inferred["probability"], event_all[selection_indices], selection.episode_id,
            inferred["boundary_offset_m"], boundary_target_all[selection_indices],
            boundary_valid_all[selection_indices],
        )
        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": steps,
            "train": {key: value / steps for key, value in train_sums.items()},
            "selection": selection_score,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if selection_score["total"] < best_score:
            best_score = selection_score["total"]
            best_epoch = epoch
            torch.save({
                "schema_version": "gse_causal_episode_detector_checkpoint_v1",
                "seed": args.seed,
                "epoch": epoch,
                "model": model.state_dict(),
                "selection_mil_total": best_score,
                "cache_manifest_sha256": manifest["array_sha256"],
            }, args.output_dir / "best.pt")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    inferred = _infer(model, selection, device=device, batch_size=args.evaluation_batch_size)
    probability = inferred["probability"].astype(np.float64)
    threshold_error = None
    trigger_metrics = None
    try:
        threshold_metrics = select_structural_threshold(
            probability, event_all[selection_indices], selection.episode_id,
            traversal[selection_indices], sequence[selection_indices],
            inferred["boundary_offset_m"], inferred["uncertainty"], minimum_precision=.98,
        )
        trigger_metrics = evaluate_causal_event_triggers(
            probability, event_all[selection_indices], selection.episode_id,
            traversal[selection_indices], sequence[selection_indices],
            inferred["boundary_offset_m"], inferred["uncertainty"],
            structural_threshold=threshold_metrics["structural_threshold"],
        )
    except RuntimeError as exc:
        threshold_error = str(exc)
    frame_metrics = _frame_metrics(probability, event_all[selection_indices])
    final_mil = _episode_mil_score(
        probability, event_all[selection_indices], selection.episode_id,
        inferred["boundary_offset_m"], boundary_target_all[selection_indices],
        boundary_valid_all[selection_indices],
    )
    np.savez_compressed(
        args.output_dir / "selection_outputs.npz",
        observation_row=selection_indices,
        probability=probability.astype(np.float32),
        boundary_offset_m=inferred["boundary_offset_m"].astype(np.float32),
        uncertainty=inferred["uncertainty"].astype(np.float32),
        event_index=event_all[selection_indices],
        episode_id=selection.episode_id,
    )
    summary = {
        "schema_version": "gse_causal_episode_detector_training_seed_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "optimizer_steps": int(sum(row["optimizer_steps"] for row in history)),
        "backbone_optimizer_steps": 0,
        "fit_observations": len(fit_indices),
        "selection_observations": len(selection_indices),
        "selection_mil": final_mil,
        "selection_frame_metrics": frame_metrics,
        "selection_trigger_metrics": trigger_metrics,
        "selection_trigger_threshold_error": threshold_error,
        "boundary_valid_selection_rows": int(boundary_valid_all[selection_indices].sum()),
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "strict_test_worlds_read": 0,
        "c09_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
