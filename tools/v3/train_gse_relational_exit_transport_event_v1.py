#!/usr/bin/env python3
"""Train one relational exit-token transport event seed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_causal_episode_sampler import EpisodePreservingBatchSampler
from mtare_topo.data.gse_relational_exit_transport_cache import RelationalExitTransportDataset
from mtare_topo.representation.gse_relational_exit_transport_event import (
    RelationalExitTokenTransportEventModel,
    parameter_count,
    relational_event_episode_loss,
)


def _collate(samples: list[dict]) -> dict:
    import torch

    return {
        "tokens": torch.from_numpy(np.stack([sample["tokens"] for sample in samples])),
        "geometry_context": torch.from_numpy(np.stack([sample["geometry_context"] for sample in samples])),
        "history_mask": torch.from_numpy(np.stack([sample["history_mask"] for sample in samples])),
        "decision_target": torch.from_numpy(np.asarray([sample["decision_target"] for sample in samples])),
        "episode_id": torch.from_numpy(np.asarray([sample["episode_id"] for sample in samples])),
        "observation_row": torch.from_numpy(np.asarray([sample["observation_row"] for sample in samples])),
    }


def _forward(model, batch, device):
    import torch

    return model(
        batch["tokens"].to(device=device, dtype=torch.float32),
        batch["geometry_context"].to(device=device, dtype=torch.float32),
        batch["history_mask"].to(device=device, dtype=torch.bool),
    )


def _decision_probability(predicted):
    import torch

    event = predicted["event_probability"]
    commit = predicted["commit_probability"]
    structural = commit[:, None] * event[:, 1:]
    result = torch.cat((1.0 - structural.sum(dim=1, keepdim=True), structural), dim=1)
    if not bool(torch.isfinite(result).all()) or not bool(torch.allclose(result.sum(dim=1), torch.ones_like(commit), atol=1e-6)):
        raise RuntimeError("relational decision probability contract drift")
    return result


def _infer(model, dataset, *, device, batch_size: int) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate, pin_memory=False)
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predicted = _forward(model, batch, device)
            output["probability"].append(_decision_probability(predicted).cpu().numpy())
            output["event_probability"].append(predicted["event_probability"].cpu().numpy())
            output["commit_probability"].append(predicted["commit_probability"].cpu().numpy())
            output["provisional_probability"].append(predicted["provisional_probability"].cpu().numpy())
            output["uncertainty"].append(predicted["uncertainty"].cpu().numpy())
            output["observation_row"].append(batch["observation_row"].numpy())
    return {key: np.concatenate(value) for key, value in output.items()}


def _selection_mil(probability: np.ndarray, target: np.ndarray, episode: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    negative = target == 0
    negative_loss = float(-np.log(np.clip(probability[negative, 0], eps, 1.0)).mean())
    positive = []
    for bag in np.unique(episode[episode >= 0]):
        rows = episode == bag
        event = int(np.unique(target[rows]).item())
        positive.append(-math.log(float(np.clip(np.max(probability[rows, event]), eps, 1.0))))
    positive_loss = float(np.mean(positive))
    return {"negative": negative_loss, "positive_episode_joint": positive_loss, "total": negative_loss + positive_loss}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--observation", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("relational event seed output exists; overwrite is forbidden")
    if len(args.observation) != 3:
        raise RuntimeError("relational event training requires three frozen observation arrays")
    args.output_dir.mkdir(parents=True)
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("formal relational event training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    partition = np.load(args.cache_dir / "partition_code.npy")
    fit_rows = np.flatnonzero(partition == 0)
    selection_rows = np.flatnonzero(partition == 1)
    if len(fit_rows) != 142_184 or len(selection_rows) != 45_942:
        raise RuntimeError("relational event split population drift")
    fit = RelationalExitTransportDataset(args.cache_dir, args.observation, fit_rows)
    selection = RelationalExitTransportDataset(args.cache_dir, args.observation, selection_rows)
    sampler = EpisodePreservingBatchSampler(fit.episode_id, batch_size=args.batch_size, seed=args.seed)
    loader = DataLoader(fit, batch_sampler=sampler, num_workers=0, collate_fn=_collate, pin_memory=False)
    model = RelationalExitTokenTransportEventModel(
        token_normalization_mean=np.load(args.cache_dir / "normalization_mean.npy"),
        token_normalization_scale=np.load(args.cache_dir / "normalization_scale.npy"),
    ).to(device)
    if parameter_count() != 240_101:
        raise RuntimeError("relational event parameter count drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_score = math.inf
    best_epoch = -1
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        totals = defaultdict(float)
        steps = 0
        for batch in loader:
            predicted = _forward(model, batch, device)
            losses = relational_event_episode_loss(
                predicted,
                batch["decision_target"].to(device=device, dtype=torch.int64),
                batch["episode_id"].to(device=device, dtype=torch.int64),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu())
            steps += 1
        inferred = _infer(model, selection, device=device, batch_size=args.evaluation_batch_size)
        if not np.array_equal(inferred["observation_row"], selection_rows):
            raise RuntimeError("relational event selection row order drift")
        score = _selection_mil(
            inferred["probability"],
            np.asarray(selection.target_all[selection_rows], dtype=np.int64),
            selection.episode_id,
        )
        record = {"epoch": epoch, "optimizer_steps": steps, "train": {key: value / steps for key, value in totals.items()}, "selection": score}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if score["total"] < best_score:
            best_score = score["total"]
            best_epoch = epoch
            torch.save({
                "schema_version": "gse_relational_exit_transport_checkpoint_v1",
                "seed": args.seed,
                "epoch": epoch,
                "model": model.state_dict(),
                "selection_mil_total": best_score,
                "cache_array_sha256": manifest["array_sha256"],
                "observation_paths": [str(path.resolve()) for path in args.observation],
            }, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    inferred = _infer(model, selection, device=device, batch_size=args.evaluation_batch_size)
    np.savez_compressed(
        args.output_dir / "selection_outputs.npz",
        observation_row=selection_rows,
        probability=inferred["probability"].astype(np.float32),
        event_probability=inferred["event_probability"].astype(np.float32),
        commit_probability=inferred["commit_probability"].astype(np.float32),
        provisional_probability=inferred["provisional_probability"].astype(np.float32),
        uncertainty=inferred["uncertainty"].astype(np.float32),
        decision_target=np.asarray(selection.target_all[selection_rows], dtype=np.int8),
        decision_episode_id=selection.episode_id,
    )
    summary = {
        "schema_version": "gse_relational_exit_transport_training_seed_v1",
        "seed": args.seed,
        "parameters": 240_101,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "optimizer_steps": int(sum(record["optimizer_steps"] for record in history)),
        "perception_backbone_optimizer_steps": 0,
        "fit_observations": len(fit_rows),
        "selection_observations": len(selection_rows),
        "selection_mil": _selection_mil(inferred["probability"], np.asarray(selection.target_all[selection_rows], dtype=np.int64), selection.episode_id),
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
