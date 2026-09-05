#!/usr/bin/env python3
"""Train one spatial-context longitudinal event-center corrective."""

from __future__ import annotations

import argparse
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
from mtare_topo.representation.gse_spatial_event_center import (
    SpatialEventCenterDecoder,
    SpatialLongitudinalCorrector,
)
from mtare_topo.teacher.gse_event_center_teacher import local_event_center_vectors
from train_gse_event_center_pair_consistency_v1 import _groups, _sample_batch
from train_gse_spatial_event_center_v1 import (
    _cross_metrics,
    _global_centers,
    _infer,
    _longitudinal,
    _relative_loss,
    _spatial_tensors,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_longitudinal_loss(predicted, target, event):
    import torch
    from torch.nn import functional as F

    terms = []
    values = {}
    for code, name in ((1, "junction"), (2, "terminal")):
        mask = event == code
        if not bool(mask.any()):
            raise ValueError("longitudinal direct batch lacks one event")
        loss = F.smooth_l1_loss(predicted[mask, 0], target[mask, 0], beta=1.0)
        values[name] = loss; terms.append(loss)
    values["total"] = torch.stack(terms).mean()
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-cache", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--scalar-checkpoint", required=True, type=Path)
    parser.add_argument("--spatial-checkpoint", required=True, type=Path)
    parser.add_argument("--initial-selection-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--identities-per-event", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False); started = time.monotonic()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal spatial longitudinal training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; device = torch.device("cuda")
    with np.load(args.teacher.resolve(), allow_pickle=False) as archive:
        partition = archive["partition_code"].astype(np.uint8); valid = archive["valid_mask"].astype(bool)
        identity = archive["identity"].astype(str); event_name = archive["event"].astype(str)
        tangent = archive["route_tangent_xyz"].astype(np.float32); objective = archive["objective_center_xyz_m"].astype(np.float32)
        longitudinal_target = archive["signed_center_offset_m"].astype(np.float32)
        oracle = archive["oracle_longitudinal_center_xyz_m"].astype(np.float32)
        global_index = archive["global_sequence_index"].astype(np.int64)
    traversal = np.load(args.action_cache / "traversal_id.npy").astype(str)
    cache_global = np.load(args.action_cache / "global_sequence_index.npy")
    if not np.array_equal(global_index, cache_global):
        raise RuntimeError("longitudinal corrective observation identity drift")
    sensor = oracle - longitudinal_target[:, None] * tangent
    local = local_event_center_vectors(sensor, objective, tangent, valid)
    basis = local["route_local_basis"]; target = local["local_center_vector_m"]
    event = np.where(event_name == "junction", 1, np.where(event_name == "terminal", 2, 0)).astype(np.int64)
    fit = np.flatnonzero(valid & (partition == 0)); selection = np.flatnonzero(valid & (partition == 1))
    if len(fit) != 25294 or len(selection) != 8839:
        raise RuntimeError("longitudinal corrective split drift")
    cache = SpatialEventCenterCache(args.spatial_cache)
    fit_dataset = ActionSetNodeDataset(args.action_cache, fit); selection_dataset = ActionSetNodeDataset(args.action_cache, selection)
    groups = _groups(fit, identity, traversal, event)

    action_checkpoint = torch.load(args.action_checkpoint.resolve(), map_location=device, weights_only=False)
    scalar_checkpoint = torch.load(args.scalar_checkpoint.resolve(), map_location=device, weights_only=False)
    spatial_checkpoint = torch.load(args.spatial_checkpoint.resolve(), map_location=device, weights_only=False)
    if (
        action_checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1"
        or action_checkpoint.get("seed") != args.seed
        or scalar_checkpoint.get("schema_version") != "gse_event_center_dual_batch_corrective_checkpoint_v1"
        or scalar_checkpoint.get("seed") != args.seed
        or spatial_checkpoint.get("schema_version") != "gse_spatial_event_center_checkpoint_v1"
        or spatial_checkpoint.get("seed") != args.seed
    ):
        raise RuntimeError("longitudinal corrective checkpoint drift")
    base = ActionSetNodeDetector().to(device); base.load_state_dict(action_checkpoint["model"], strict=True)
    scalar = EventCenterOffsetHead().to(device); scalar.load_state_dict(scalar_checkpoint["head"], strict=True)
    for module in (base, scalar):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    spatial = SpatialEventCenterDecoder().to(device); spatial.load_state_dict(spatial_checkpoint["decoder"], strict=True)
    corrector = SpatialLongitudinalCorrector(spatial).to(device); corrector.freeze_spatial_decoder()
    trainable = list(corrector.longitudinal_residual.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    steps = int(np.ceil(len(fit) / 64)); fit_event = event[fit]
    by_event = {code: np.flatnonzero(fit_event == code) for code in (1, 2)}

    initial_vector = _infer(base, scalar, corrector, selection_dataset, cache, selection, device, args.evaluation_batch_size)
    with np.load(args.initial_selection_output.resolve(), allow_pickle=False) as archive:
        if (
            not np.array_equal(archive["observation_row"], selection)
            or not np.array_equal(archive["global_sequence_index"], global_index[selection])
            or not np.allclose(archive["predicted_local_vector_m"], initial_vector, atol=1e-6, rtol=0.0)
        ):
            raise RuntimeError("longitudinal corrective initialization does not reproduce V1")
    initial_center = _global_centers(initial_vector, selection, sensor, basis)
    initial_cross = _cross_metrics(initial_center, selection, identity, traversal, objective[selection])
    initial_component_mae = np.mean(np.abs(initial_vector - target[selection]), axis=0)
    initial_global_mae = float(np.mean(np.linalg.norm(initial_center - objective[selection], axis=1)))
    best_score = (initial_cross["identity_macro_relative_vector_error_m"], initial_global_mae)
    common = {
        "schema_version": "gse_spatial_longitudinal_corrective_checkpoint_v1", "seed": args.seed,
        "action_checkpoint_sha256": _sha(args.action_checkpoint.resolve()),
        "scalar_checkpoint_sha256": _sha(args.scalar_checkpoint.resolve()),
        "spatial_checkpoint_sha256": _sha(args.spatial_checkpoint.resolve()),
        "spatial_cache_manifest_sha256": _sha(args.spatial_cache / "manifest.json"),
        "teacher_sha256": _sha(args.teacher.resolve()),
    }
    torch.save({**common, "epoch": -1, "longitudinal_residual": corrector.longitudinal_residual.state_dict(), "selection_cross_view": initial_cross, "selection_global_center_mae_m": initial_global_mae}, args.output_dir / "best.pt")
    history = []
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        pools = {code: rng.choice(values, size=steps * 32, replace=steps * 32 > len(values)) for code, values in by_event.items()}
        totals = {"direct": 0.0, "relative": 0.0, "total": 0.0}
        corrector.train(); corrector.spatial_decoder.eval()
        for step in range(steps):
            pair_local, pair_index = _sample_batch(groups, rng, args.identities_per_event); pair_rows = fit[pair_local]
            direct_local = np.concatenate([pools[code][step * 32:(step + 1) * 32] for code in (1, 2)])
            direct_local = direct_local[rng.permutation(len(direct_local))]; direct_rows = fit[direct_local]
            direct_long = _longitudinal(base, scalar, fit_dataset, direct_local, device)
            pair_long = _longitudinal(base, scalar, fit_dataset, pair_local, device)
            direct_prediction = corrector(*_spatial_tensors(cache, direct_rows, device), direct_long)
            pair_prediction = corrector(*_spatial_tensors(cache, pair_rows, device), pair_long)
            direct_losses = _direct_longitudinal_loss(
                direct_prediction, torch.from_numpy(target[direct_rows]).to(device), torch.from_numpy(event[direct_rows]).to(device),
            )
            predicted_center = torch.from_numpy(sensor[pair_rows]).to(device) + torch.einsum(
                "nij,ni->nj", torch.from_numpy(basis[pair_rows]).to(device), pair_prediction,
            )
            relative = _relative_loss(
                predicted_center, torch.from_numpy(objective[pair_rows]).to(device), torch.from_numpy(pair_index).to(device),
            )
            loss = direct_losses["total"] + relative
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
            totals["direct"] += float(direct_losses["total"].detach().cpu()); totals["relative"] += float(relative.detach().cpu()); totals["total"] += float(loss.detach().cpu())
        predicted = _infer(base, scalar, corrector, selection_dataset, cache, selection, device, args.evaluation_batch_size)
        center = _global_centers(predicted, selection, sensor, basis)
        cross = _cross_metrics(center, selection, identity, traversal, objective[selection])
        component_mae = np.mean(np.abs(predicted - target[selection]), axis=0)
        global_mae = float(np.mean(np.linalg.norm(center - objective[selection], axis=1)))
        transverse_drift = float(np.max(np.abs(predicted[:, 1:] - initial_vector[:, 1:])))
        record = {"epoch": epoch, "steps": steps, "train": {key: value / steps for key, value in totals.items()}, "selection_component_mae_m": component_mae.tolist(), "selection_global_center_mae_m": global_mae, "transverse_max_drift_m": transverse_drift, **cross}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        score = (cross["identity_macro_relative_vector_error_m"], global_mae)
        if component_mae[0] <= initial_component_mae[0] and transverse_drift == 0.0 and score < best_score:
            best_score = score
            torch.save({**common, "epoch": epoch, "longitudinal_residual": corrector.longitudinal_residual.state_dict(), "selection_cross_view": cross, "selection_global_center_mae_m": global_mae}, args.output_dir / "best.pt")

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    corrector.longitudinal_residual.load_state_dict(best["longitudinal_residual"], strict=True)
    predicted = _infer(base, scalar, corrector, selection_dataset, cache, selection, device, args.evaluation_batch_size)
    center = _global_centers(predicted, selection, sensor, basis)
    cross = _cross_metrics(center, selection, identity, traversal, objective[selection])
    component_mae = np.mean(np.abs(predicted - target[selection]), axis=0)
    transverse_drift = float(np.max(np.abs(predicted[:, 1:] - initial_vector[:, 1:])))
    summary = {
        "schema_version": "gse_spatial_longitudinal_corrective_seed_v1", "seed": args.seed,
        "best_epoch": int(best["epoch"]), "epochs": args.epochs, "optimizer_steps": args.epochs * steps,
        "spatial_decoder_optimizer_steps": 0, "backbone_optimizer_steps": 0,
        "fit_rows": len(fit), "selection_rows": len(selection),
        "selection_component_mae_m": component_mae.tolist(), "initial_selection_component_mae_m": initial_component_mae.tolist(),
        "selection_global_center_mae_m": float(np.mean(np.linalg.norm(center - objective[selection], axis=1))),
        "initial_selection_global_center_mae_m": initial_global_mae,
        "selection_cross_view": cross, "initial_selection_cross_view": initial_cross,
        "transverse_max_drift_m": transverse_drift,
        "duration_seconds": time.monotonic() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    np.savez_compressed(
        args.output_dir / "selection_outputs.npz", observation_row=selection,
        global_sequence_index=global_index[selection], predicted_local_vector_m=predicted.astype(np.float32),
        predicted_center_xyz_m=center.astype(np.float32), target_local_vector_m=target[selection], target_center_xyz_m=objective[selection],
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
