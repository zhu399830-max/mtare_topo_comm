#!/usr/bin/env python3
"""Train an identity-balanced cross-traversal event-center offset head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import (
    EventCenterOffsetHead, event_center_offset_loss, event_center_relative_loss,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collate(dataset: ActionSetNodeDataset, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = [dataset[int(value)] for value in rows]
    return np.stack([value["tokens"] for value in samples]), np.stack([value["history_mask"] for value in samples])


def _infer(base, head, dataset, count, *, device, batch_size=512):
    import torch
    output = []
    base.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            local = np.arange(start, min(count, start + batch_size))
            tokens, mask = _collate(dataset, local)
            context = base(torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device))["causal_context"]
            output.append(head(context).cpu().numpy())
    return np.concatenate(output)


def _groups(local_rows, identity, traversal, event_index):
    grouped = {1: defaultdict(lambda: defaultdict(list)), 2: defaultdict(lambda: defaultdict(list))}
    for local, row in enumerate(local_rows):
        grouped[int(event_index[row])][str(identity[row])][str(traversal[row])].append(local)
    return {
        event: [value for value in identities.values() if len(value) >= 2]
        for event, identities in grouped.items()
    }


def _sample_batch(groups, rng, identities_per_event):
    local_rows = []
    pairs = []
    for event in (1, 2):
        choices = rng.choice(len(groups[event]), size=identities_per_event, replace=identities_per_event > len(groups[event]))
        for choice in choices:
            views = groups[event][int(choice)]
            trace_names = sorted(views)
            selected = rng.choice(len(trace_names), size=2, replace=False)
            left = int(rng.choice(views[trace_names[int(selected[0])]]))
            right = int(rng.choice(views[trace_names[int(selected[1])]]))
            pair_start = len(local_rows)
            local_rows.extend((left, right)); pairs.append((pair_start, pair_start + 1))
    return np.asarray(local_rows, dtype=np.int64), np.asarray(pairs, dtype=np.int64)


def _cross_view_metrics(predicted, rows, identity, traversal, sensor, tangent, target):
    predicted_center = sensor[rows] + predicted[:, None] * tangent[rows]
    target_center = sensor[rows] + target[rows, None] * tangent[rows]
    grouped = defaultdict(lambda: defaultdict(list))
    for local, row in enumerate(rows):
        grouped[str(identity[row])][str(traversal[row])].append(local)
    macro_error, macro_within = [], []
    for views in grouped.values():
        names = sorted(views); errors, within = [], []
        for left_index, left_name in enumerate(names):
            left = np.asarray(views[left_name], dtype=np.int64)
            for right_name in names[left_index + 1:]:
                right = np.asarray(views[right_name], dtype=np.int64)
                predicted_delta = (predicted_center[left, None] - predicted_center[right[None, :]]).reshape(-1, 3)
                target_delta = (target_center[left, None] - target_center[right[None, :]]).reshape(-1, 3)
                errors.extend(np.linalg.norm(predicted_delta - target_delta, axis=1).tolist())
                within.extend((np.linalg.norm(predicted_delta, axis=1) <= 4.0).tolist())
        if errors:
            macro_error.append(float(np.mean(errors))); macro_within.append(float(np.mean(within)))
    return {
        "identity_macro_relative_vector_error_m": float(np.mean(macro_error)),
        "identity_macro_within_4m_fraction": float(np.mean(macro_within)),
        "multi_traversal_identities": len(macro_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--event-center-teacher", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--initial-center-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--identities-per-event", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(); started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("formal pair-consistency training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    with np.load(args.event_center_teacher.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8); valid = archive["valid_mask"].astype(bool)
        target = archive["signed_center_offset_m"].astype(np.float32); event_name = archive["event"].astype(str)
        identity = archive["identity"].astype(str); tangent = archive["route_tangent_xyz"].astype(np.float32)
        oracle_center = archive["oracle_longitudinal_center_xyz_m"].astype(np.float32)
        global_index = archive["global_sequence_index"].astype(np.int64)
    cache_global = np.load(args.action_cache / "global_sequence_index.npy")
    traversal = np.load(args.action_cache / "traversal_id.npy").astype(str)
    if not np.array_equal(cache_global, global_index): raise RuntimeError("pair-consistency/action-cache identity drift")
    sensor = oracle_center - target[:, None] * tangent
    fit_rows = np.flatnonzero(valid & (partition == 0)); selection_rows = np.flatnonzero(valid & (partition == 1))
    event_index = np.where(event_name == "junction", 1, np.where(event_name == "terminal", 2, 0)).astype(np.int64)
    if len(fit_rows) != 25_294 or len(selection_rows) != 8_839: raise RuntimeError("pair-consistency split drift")
    fit_dataset = ActionSetNodeDataset(args.action_cache, fit_rows); selection_dataset = ActionSetNodeDataset(args.action_cache, selection_rows)
    fit_groups = _groups(fit_rows, identity, traversal, event_index)
    if any(len(fit_groups[value]) == 0 for value in (1, 2)): raise RuntimeError("pair-consistency groups are empty")
    checkpoint = torch.load(args.action_checkpoint.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1" or checkpoint.get("seed") != args.seed:
        raise RuntimeError("pair-consistency action checkpoint drift")
    base = ActionSetNodeDetector().to(device); base.load_state_dict(checkpoint["model"], strict=True); base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    head = EventCenterOffsetHead().to(device)
    initial = torch.load(args.initial_center_checkpoint.resolve(), map_location=device, weights_only=False)
    if initial.get("schema_version") != "gse_event_center_offset_checkpoint_v1" or initial.get("seed") != args.seed:
        raise RuntimeError("pair-consistency initial center checkpoint drift")
    head.load_state_dict(initial["head"], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(fit_rows) / (4 * args.identities_per_event)))
    fit_event = event_index[fit_rows]
    local_by_event = {value: np.flatnonzero(fit_event == value) for value in (1, 2)}
    initial_prediction = _infer(base, head, selection_dataset, len(selection_rows), device=device)
    initial_cross = _cross_view_metrics(initial_prediction, selection_rows, identity, traversal, sensor, tangent, target)
    initial_mae = float(np.mean(np.abs(initial_prediction - target[selection_rows])))
    best_score = (initial_cross["identity_macro_relative_vector_error_m"], initial_mae)
    torch.save({
        "schema_version": "gse_event_center_dual_batch_corrective_checkpoint_v1", "seed": args.seed,
        "epoch": -1, "head": head.state_dict(), "selection_mae_m": initial_mae,
        "selection_cross_view": initial_cross, "initial_center_checkpoint_sha256": _sha(args.initial_center_checkpoint.resolve()),
        "action_checkpoint_sha256": _sha(args.action_checkpoint.resolve()), "teacher_sha256": _sha(args.event_center_teacher.resolve()),
    }, args.output_dir / "best.pt")
    history = []
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch); head.train()
        totals = {"junction": 0.0, "terminal": 0.0, "relative_center": 0.0, "total": 0.0}
        direct_pools = {}
        for event, values in local_by_event.items():
            needed = steps_per_epoch * 32
            direct_pools[event] = rng.choice(values, size=needed, replace=needed > len(values))
        for step in range(steps_per_epoch):
            pair_local, pair_index = _sample_batch(fit_groups, rng, args.identities_per_event)
            pair_rows = fit_rows[pair_local]; pair_tokens, pair_mask = _collate(fit_dataset, pair_local)
            direct_local = np.concatenate([
                direct_pools[event][step * 32:(step + 1) * 32] for event in (1, 2)
            ])
            direct_local = direct_local[rng.permutation(len(direct_local))]
            direct_rows = fit_rows[direct_local]; direct_tokens, direct_mask = _collate(fit_dataset, direct_local)
            with torch.no_grad():
                direct_context = base(torch.from_numpy(direct_tokens).to(device), torch.from_numpy(direct_mask).to(device))["causal_context"]
                pair_context = base(torch.from_numpy(pair_tokens).to(device), torch.from_numpy(pair_mask).to(device))["causal_context"]
            direct_predicted = head(direct_context); pair_predicted = head(pair_context)
            direct_losses = event_center_offset_loss(
                direct_predicted, torch.from_numpy(target[direct_rows]).to(device),
                torch.from_numpy(event_index[direct_rows]).to(device),
            )
            relative = event_center_relative_loss(
                pair_predicted, torch.from_numpy(target[pair_rows]).to(device),
                torch.from_numpy(event_index[pair_rows]).to(device),
                torch.from_numpy(sensor[pair_rows]).to(device), torch.from_numpy(tangent[pair_rows]).to(device),
                torch.from_numpy(pair_index).to(device),
            )
            losses = {"junction": direct_losses["junction"], "terminal": direct_losses["terminal"],
                      "relative_center": relative, "total": direct_losses["total"] + relative}
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); optimizer.step()
            for key in totals: totals[key] += float(losses[key].detach().cpu())
        predicted = _infer(base, head, selection_dataset, len(selection_rows), device=device)
        mae = float(np.mean(np.abs(predicted - target[selection_rows])))
        cross = _cross_view_metrics(predicted, selection_rows, identity, traversal, sensor, tangent, target)
        record = {"epoch": epoch, "steps": steps_per_epoch, "train": {key: value / steps_per_epoch for key, value in totals.items()}, "selection_mae_m": mae, **cross}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        score = (cross["identity_macro_relative_vector_error_m"], mae)
        if mae <= initial_mae and score < best_score:
            best_score = score
            torch.save({
                "schema_version": "gse_event_center_dual_batch_corrective_checkpoint_v1", "seed": args.seed,
                "epoch": epoch, "head": head.state_dict(), "selection_mae_m": mae,
                "selection_cross_view": cross, "initial_center_checkpoint_sha256": _sha(args.initial_center_checkpoint.resolve()),
                "action_checkpoint_sha256": _sha(args.action_checkpoint.resolve()),
                "teacher_sha256": _sha(args.event_center_teacher.resolve()),
            }, args.output_dir / "best.pt")
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False); head.load_state_dict(best["head"], strict=True)
    predicted = _infer(base, head, selection_dataset, len(selection_rows), device=device)
    cross = _cross_view_metrics(predicted, selection_rows, identity, traversal, sensor, tangent, target)
    per_event = {}
    for event, name in ((1, "junction"), (2, "terminal")):
        mask = event_index[selection_rows] == event
        per_event[name] = {"rows": int(np.sum(mask)), "mae_m": float(np.mean(np.abs(predicted[mask] - target[selection_rows][mask])))}
    summary = {
        "schema_version": "gse_event_center_dual_batch_corrective_seed_v1", "seed": args.seed,
        "best_epoch": int(best["epoch"]), "epochs": args.epochs, "optimizer_steps": args.epochs * steps_per_epoch,
        "backbone_optimizer_steps": 0, "fit_rows": len(fit_rows), "selection_rows": len(selection_rows),
        "selection_mae_m": float(np.mean(np.abs(predicted - target[selection_rows]))), "selection_cross_view": cross,
        "initial_selection_mae_m": initial_mae, "initial_selection_cross_view": initial_cross,
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
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
