#!/usr/bin/env python3
"""Train one height-aware five-frame event-center spatial decoder."""

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
from mtare_topo.data.gse_spatial_event_center_cache import SpatialEventCenterCache
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import EventCenterOffsetHead
from mtare_topo.representation.gse_spatial_event_center import SpatialEventCenterDecoder
from mtare_topo.teacher.gse_event_center_teacher import local_event_center_vectors
from train_gse_event_center_pair_consistency_v1 import _collate, _groups, _sample_batch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _global_centers(vector: np.ndarray, rows: np.ndarray, sensor: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return sensor[rows] + np.einsum("nij,ni->nj", basis[rows], vector)


def _cross_metrics(
    center: np.ndarray, rows: np.ndarray, identity: np.ndarray,
    traversal: np.ndarray, target_center: np.ndarray,
) -> dict[str, float | int]:
    if target_center.shape != center.shape:
        raise ValueError("spatial center targets must align with selected rows")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for local, row in enumerate(rows):
        grouped[str(identity[row])][str(traversal[row])].append(local)
    macro_error: list[float] = []
    macro_within: list[float] = []
    for views in grouped.values():
        names = sorted(views)
        errors: list[float] = []
        within: list[bool] = []
        for left_index, left_name in enumerate(names):
            left = np.asarray(views[left_name], dtype=np.int64)
            for right_name in names[left_index + 1:]:
                right = np.asarray(views[right_name], dtype=np.int64)
                predicted_delta = (center[left, None] - center[right[None, :]]).reshape(-1, 3)
                target_delta = (target_center[left, None] - target_center[right[None, :]]).reshape(-1, 3)
                errors.extend(np.linalg.norm(predicted_delta - target_delta, axis=1).tolist())
                within.extend((np.linalg.norm(predicted_delta, axis=1) <= 4.0).tolist())
        if errors:
            macro_error.append(float(np.mean(errors)))
            macro_within.append(float(np.mean(within)))
    return {
        "identity_macro_relative_vector_error_m": float(np.mean(macro_error)),
        "identity_macro_within_4m_fraction": float(np.mean(macro_within)),
        "multi_traversal_identities": len(macro_error),
    }


def _spatial_tensors(cache: SpatialEventCenterCache, rows: np.ndarray, device):
    import torch

    features = cache.gather(rows)
    return (
        torch.from_numpy(features["azimuth"]).to(device),
        torch.from_numpy(features["elevation"]).to(device),
        torch.from_numpy(features["pooled"]).to(device),
    )


def _longitudinal(
    base: ActionSetNodeDetector, scalar: EventCenterOffsetHead,
    dataset: ActionSetNodeDataset, local_rows: np.ndarray, device,
):
    import torch

    tokens, mask = _collate(dataset, local_rows)
    with torch.no_grad():
        context = base(
            torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device),
        )["causal_context"]
        return scalar(context)


def _infer(
    base: ActionSetNodeDetector, scalar: EventCenterOffsetHead,
    decoder: SpatialEventCenterDecoder, action_dataset: ActionSetNodeDataset,
    spatial_cache: SpatialEventCenterCache, global_rows: np.ndarray, device,
    batch_size: int = 128,
) -> np.ndarray:
    import torch

    output = []
    base.eval(); scalar.eval(); decoder.eval()
    with torch.inference_mode():
        for start in range(0, len(global_rows), batch_size):
            stop = min(len(global_rows), start + batch_size)
            local = np.arange(start, stop, dtype=np.int64)
            longitudinal = _longitudinal(base, scalar, action_dataset, local, device)
            azimuth, elevation, pooled = _spatial_tensors(spatial_cache, global_rows[local], device)
            output.append(decoder(azimuth, elevation, pooled, longitudinal).cpu().numpy())
    return np.concatenate(output)


def _direct_transverse_loss(predicted, target, event):
    import torch
    from torch.nn import functional as F

    terms = []
    values = {}
    for code, name in ((1, "junction"), (2, "terminal")):
        mask = event == code
        if not bool(mask.any()):
            raise ValueError("spatial center direct batch lacks one event")
        loss = F.smooth_l1_loss(predicted[mask, 1:], target[mask, 1:], beta=0.5)
        values[name] = loss
        terms.append(loss)
    values["total"] = torch.stack(terms).mean()
    return values


def _relative_loss(predicted_center, target_center, pair_index):
    from torch.nn import functional as F

    predicted_delta = predicted_center[pair_index[:, 0]] - predicted_center[pair_index[:, 1]]
    target_delta = target_center[pair_index[:, 0]] - target_center[pair_index[:, 1]]
    return F.smooth_l1_loss(predicted_delta, target_delta, beta=1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-cache", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--scalar-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--identities-per-event", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal spatial event-center training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    with np.load(args.teacher.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8)
        valid = archive["valid_mask"].astype(bool)
        identity = archive["identity"].astype(str)
        event_name = archive["event"].astype(str)
        tangent = archive["route_tangent_xyz"].astype(np.float32)
        objective = archive["objective_center_xyz_m"].astype(np.float32)
        longitudinal_target = archive["signed_center_offset_m"].astype(np.float32)
        oracle = archive["oracle_longitudinal_center_xyz_m"].astype(np.float32)
        global_index = archive["global_sequence_index"].astype(np.int64)
    traversal = np.load(args.action_cache / "traversal_id.npy").astype(str)
    cache_global = np.load(args.action_cache / "global_sequence_index.npy")
    if not np.array_equal(global_index, cache_global):
        raise RuntimeError("spatial/action/Teacher observation identity drift")
    sensor = oracle - longitudinal_target[:, None] * tangent
    local = local_event_center_vectors(sensor, objective, tangent, valid)
    basis = local["route_local_basis"]
    target = local["local_center_vector_m"]
    event = np.where(event_name == "junction", 1, np.where(event_name == "terminal", 2, 0)).astype(np.int64)
    fit = np.flatnonzero(valid & (partition == 0))
    selection = np.flatnonzero(valid & (partition == 1))
    if len(fit) != 25294 or len(selection) != 8839:
        raise RuntimeError("spatial event-center decision split drift")
    spatial_cache = SpatialEventCenterCache(args.spatial_cache)
    if len(spatial_cache.references) != len(global_index):
        raise RuntimeError("spatial event-center observation population drift")
    fit_dataset = ActionSetNodeDataset(args.action_cache, fit)
    selection_dataset = ActionSetNodeDataset(args.action_cache, selection)
    groups = _groups(fit, identity, traversal, event)
    if any(len(groups[code]) == 0 for code in (1, 2)):
        raise RuntimeError("spatial event-center cross-traversal groups are empty")

    action_checkpoint = torch.load(args.action_checkpoint.resolve(), map_location=device, weights_only=False)
    scalar_checkpoint = torch.load(args.scalar_checkpoint.resolve(), map_location=device, weights_only=False)
    if (
        action_checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1"
        or action_checkpoint.get("seed") != args.seed
        or scalar_checkpoint.get("schema_version") != "gse_event_center_dual_batch_corrective_checkpoint_v1"
        or scalar_checkpoint.get("seed") != args.seed
    ):
        raise RuntimeError("spatial event-center initializer drift")
    base = ActionSetNodeDetector().to(device)
    base.load_state_dict(action_checkpoint["model"], strict=True)
    scalar = EventCenterOffsetHead().to(device)
    scalar.load_state_dict(scalar_checkpoint["head"], strict=True)
    for module in (base, scalar):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    decoder = SpatialEventCenterDecoder().to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = int(np.ceil(len(fit) / 64))
    fit_event = event[fit]
    by_event = {code: np.flatnonzero(fit_event == code) for code in (1, 2)}

    initial_vector = _infer(
        base, scalar, decoder, selection_dataset, spatial_cache, selection, device,
        args.evaluation_batch_size,
    )
    initial_center = _global_centers(initial_vector, selection, sensor, basis)
    initial_cross = _cross_metrics(initial_center, selection, identity, traversal, objective[selection])
    initial_component_mae = np.mean(np.abs(initial_vector - target[selection]), axis=0)
    initial_global_mae = float(np.mean(np.linalg.norm(initial_center - objective[selection], axis=1)))
    best_score = (initial_cross["identity_macro_relative_vector_error_m"], initial_global_mae)
    history = []
    checkpoint_common = {
        "schema_version": "gse_spatial_event_center_checkpoint_v1",
        "seed": args.seed,
        "action_checkpoint_sha256": _sha(args.action_checkpoint.resolve()),
        "scalar_checkpoint_sha256": _sha(args.scalar_checkpoint.resolve()),
        "spatial_cache_manifest_sha256": _sha(args.spatial_cache / "manifest.json"),
        "teacher_sha256": _sha(args.teacher.resolve()),
    }
    torch.save({
        **checkpoint_common, "epoch": -1, "decoder": decoder.state_dict(),
        "selection_cross_view": initial_cross,
        "selection_global_center_mae_m": initial_global_mae,
    }, args.output_dir / "best.pt")

    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        pools = {
            code: rng.choice(values, size=steps * 32, replace=steps * 32 > len(values))
            for code, values in by_event.items()
        }
        totals = {"direct": 0.0, "relative": 0.0, "total": 0.0}
        decoder.train()
        for step in range(steps):
            pair_local, pair_index = _sample_batch(groups, rng, args.identities_per_event)
            pair_rows = fit[pair_local]
            direct_local = np.concatenate([
                pools[code][step * 32:(step + 1) * 32] for code in (1, 2)
            ])
            direct_local = direct_local[rng.permutation(len(direct_local))]
            direct_rows = fit[direct_local]
            direct_longitudinal = _longitudinal(base, scalar, fit_dataset, direct_local, device)
            pair_longitudinal = _longitudinal(base, scalar, fit_dataset, pair_local, device)
            direct_features = _spatial_tensors(spatial_cache, direct_rows, device)
            pair_features = _spatial_tensors(spatial_cache, pair_rows, device)
            direct_prediction = decoder(*direct_features, direct_longitudinal)
            pair_prediction = decoder(*pair_features, pair_longitudinal)
            direct_losses = _direct_transverse_loss(
                direct_prediction, torch.from_numpy(target[direct_rows]).to(device),
                torch.from_numpy(event[direct_rows]).to(device),
            )
            pair_basis = torch.from_numpy(basis[pair_rows]).to(device)
            pair_sensor = torch.from_numpy(sensor[pair_rows]).to(device)
            predicted_center = pair_sensor + torch.einsum("nij,ni->nj", pair_basis, pair_prediction)
            relative = _relative_loss(
                predicted_center, torch.from_numpy(objective[pair_rows]).to(device),
                torch.from_numpy(pair_index).to(device),
            )
            loss = direct_losses["total"] + relative
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            totals["direct"] += float(direct_losses["total"].detach().cpu())
            totals["relative"] += float(relative.detach().cpu())
            totals["total"] += float(loss.detach().cpu())

        predicted = _infer(
            base, scalar, decoder, selection_dataset, spatial_cache, selection,
            device, args.evaluation_batch_size,
        )
        center = _global_centers(predicted, selection, sensor, basis)
        cross = _cross_metrics(center, selection, identity, traversal, objective[selection])
        component_mae = np.mean(np.abs(predicted - target[selection]), axis=0)
        global_mae = float(np.mean(np.linalg.norm(center - objective[selection], axis=1)))
        record = {
            "epoch": epoch, "steps": steps,
            "train": {key: value / steps for key, value in totals.items()},
            "selection_component_mae_m": component_mae.tolist(),
            "selection_global_center_mae_m": global_mae,
            **cross,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        score = (cross["identity_macro_relative_vector_error_m"], global_mae)
        if score < best_score:
            best_score = score
            torch.save({
                **checkpoint_common, "epoch": epoch, "decoder": decoder.state_dict(),
                "selection_cross_view": cross,
                "selection_global_center_mae_m": global_mae,
            }, args.output_dir / "best.pt")

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    decoder.load_state_dict(best["decoder"], strict=True)
    predicted = _infer(
        base, scalar, decoder, selection_dataset, spatial_cache, selection,
        device, args.evaluation_batch_size,
    )
    center = _global_centers(predicted, selection, sensor, basis)
    cross = _cross_metrics(center, selection, identity, traversal, objective[selection])
    component_mae = np.mean(np.abs(predicted - target[selection]), axis=0)
    summary = {
        "schema_version": "gse_spatial_event_center_seed_v1",
        "seed": args.seed, "best_epoch": int(best["epoch"]),
        "epochs": args.epochs, "optimizer_steps": args.epochs * steps,
        "backbone_optimizer_steps": 0,
        "fit_rows": len(fit), "selection_rows": len(selection),
        "selection_component_mae_m": component_mae.tolist(),
        "initial_selection_component_mae_m": initial_component_mae.tolist(),
        "selection_global_center_mae_m": float(np.mean(np.linalg.norm(center - objective[selection], axis=1))),
        "initial_selection_global_center_mae_m": initial_global_mae,
        "selection_cross_view": cross,
        "initial_selection_cross_view": initial_cross,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "selection_outputs.npz",
        observation_row=selection,
        global_sequence_index=global_index[selection],
        predicted_local_vector_m=predicted.astype(np.float32),
        predicted_center_xyz_m=center.astype(np.float32),
        target_local_vector_m=target[selection],
        target_center_xyz_m=objective[selection],
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
