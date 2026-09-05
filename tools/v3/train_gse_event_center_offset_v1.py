#!/usr/bin/env python3
"""Train one lightweight event-center offset head on a frozen action encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import (
    EventCenterOffsetHead,
    event_center_offset_loss,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collate(dataset: ActionSetNodeDataset, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = [dataset[int(value)] for value in rows]
    return (
        np.stack([value["tokens"] for value in samples]),
        np.stack([value["history_mask"] for value in samples]),
    )


def _infer(base, head, dataset, target_rows, *, device, batch_size=512):
    import torch

    output = []
    base.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, len(target_rows), batch_size):
            local = np.arange(start, min(len(target_rows), start + batch_size))
            tokens, mask = _collate(dataset, local)
            context = base(
                torch.from_numpy(tokens).to(device),
                torch.from_numpy(mask).to(device),
            )["causal_context"]
            output.append(head(context).cpu().numpy())
    return np.concatenate(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--event-center-teacher", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-per-event", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal event-center training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    with np.load(args.event_center_teacher.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8)
        valid = archive["valid_mask"].astype(bool)
        target = archive["signed_center_offset_m"].astype(np.float32)
        event_name = archive["event"].astype(str)
        global_index = archive["global_sequence_index"].astype(np.int64)
    cache_global = np.load(args.action_cache / "global_sequence_index.npy")
    if not np.array_equal(cache_global, global_index):
        raise RuntimeError("event-center/action-cache identity drift")
    fit_rows = np.flatnonzero(valid & (partition == 0))
    selection_rows = np.flatnonzero(valid & (partition == 1))
    event_index = np.where(event_name == "junction", 1, np.where(event_name == "terminal", 2, 0)).astype(np.int64)
    if len(fit_rows) != 25_294 or len(selection_rows) != 8_839:
        raise RuntimeError("event-center train split drift")
    fit_dataset = ActionSetNodeDataset(args.action_cache, fit_rows)
    selection_dataset = ActionSetNodeDataset(args.action_cache, selection_rows)
    checkpoint = torch.load(args.action_checkpoint.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1" or checkpoint.get("seed") != args.seed:
        raise RuntimeError("event-center action checkpoint drift")
    base = ActionSetNodeDetector().to(device)
    base.load_state_dict(checkpoint["model"], strict=True); base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    head = EventCenterOffsetHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    fit_event = event_index[fit_rows]
    local_by_event = {value: np.flatnonzero(fit_event == value) for value in (1, 2)}
    steps_per_epoch = int(np.ceil(max(len(value) for value in local_by_event.values()) / args.batch_per_event))
    history = []
    best_mae = float("inf")
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        pools = {}
        for event, values in local_by_event.items():
            needed = steps_per_epoch * args.batch_per_event
            pools[event] = rng.choice(values, size=needed, replace=needed > len(values))
        totals = {"junction": 0.0, "terminal": 0.0, "total": 0.0}
        head.train()
        for step in range(steps_per_epoch):
            local = np.concatenate([
                pools[event][step * args.batch_per_event:(step + 1) * args.batch_per_event]
                for event in (1, 2)
            ])
            local = local[rng.permutation(len(local))]
            tokens, mask = _collate(fit_dataset, local)
            with torch.no_grad():
                context = base(torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device))["causal_context"]
            predicted = head(context)
            losses = event_center_offset_loss(
                predicted,
                torch.from_numpy(target[fit_rows[local]]).to(device),
                torch.from_numpy(event_index[fit_rows[local]]).to(device),
            )
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); optimizer.step()
            for key in totals: totals[key] += float(losses[key].detach().cpu())
        predicted = _infer(base, head, selection_dataset, selection_rows, device=device)
        mae = float(np.mean(np.abs(predicted - target[selection_rows])))
        record = {"epoch": epoch, "steps": steps_per_epoch, "train": {key: value / steps_per_epoch for key, value in totals.items()}, "selection_mae_m": mae}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        if mae < best_mae:
            best_mae = mae
            torch.save({
                "schema_version": "gse_event_center_offset_checkpoint_v1", "seed": args.seed,
                "epoch": epoch, "head": head.state_dict(), "selection_mae_m": mae,
                "action_checkpoint_sha256": _sha(args.action_checkpoint.resolve()),
                "teacher_sha256": _sha(args.event_center_teacher.resolve()),
            }, args.output_dir / "best.pt")
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(best["head"], strict=True)
    predicted = _infer(base, head, selection_dataset, selection_rows, device=device)
    per_event = {}
    fit_median = {}
    baseline_prediction = np.zeros(len(selection_rows), dtype=np.float32)
    for event, name in ((1, "junction"), (2, "terminal")):
        fit_median[name] = float(np.median(target[fit_rows][fit_event == event]))
        mask = event_index[selection_rows] == event
        baseline_prediction[mask] = fit_median[name]
        per_event[name] = {
            "rows": int(np.sum(mask)),
            "mae_m": float(np.mean(np.abs(predicted[mask] - target[selection_rows][mask]))),
        }
    baseline_mae = float(np.mean(np.abs(baseline_prediction - target[selection_rows])))
    summary = {
        "schema_version": "gse_event_center_offset_training_seed_v1", "seed": args.seed,
        "best_epoch": int(best["epoch"]), "epochs": args.epochs,
        "optimizer_steps": args.epochs * steps_per_epoch, "backbone_optimizer_steps": 0,
        "fit_rows": len(fit_rows), "selection_rows": len(selection_rows),
        "selection_mae_m": float(np.mean(np.abs(predicted - target[selection_rows]))),
        "fit_event_median_baseline_m": fit_median, "selection_baseline_mae_m": baseline_mae,
        "relative_mae_improvement": 1.0 - float(np.mean(np.abs(predicted - target[selection_rows]))) / baseline_mae,
        "per_event": per_event, "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "selection_outputs.npz", observation_row=selection_rows,
        global_sequence_index=global_index[selection_rows], predicted_offset_m=predicted.astype(np.float32),
        target_offset_m=target[selection_rows], event_index=event_index[selection_rows].astype(np.int8),
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
