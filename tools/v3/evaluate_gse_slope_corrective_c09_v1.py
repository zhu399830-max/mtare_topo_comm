#!/usr/bin/env python3
"""Evaluate three frozen slope-corrective seeds on all ten C09 worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_slope_corrective_dataset import (
    APPROVED_CORRECTIVE_PARENTS,
    normalize_features,
    validate_causal_references,
)
from mtare_topo.representation.gse_slope_corrective import (
    PhysicsGuidedSlopeResidualNet,
    RAW_FEATURES_PER_FRAME,
    slope_frame_features,
)
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


PASS_STATUS = "PASS_GSE_SLOPE_CORRECTIVE_C09_EVALUATION_V1"
EXPECTED_PARENTS = tuple(sorted(f"{parent.rsplit('_C', 1)[0]}_C09" for parent in APPROVED_CORRECTIVE_PARENTS if parent.endswith("_C07")))


def _metrics(target: np.ndarray, current: np.ndarray, prior: np.ndarray, corrected: np.ndarray, parents: np.ndarray) -> dict[str, Any]:
    current_error = np.abs(current.astype(np.float64) - target)
    prior_error = np.abs(prior.astype(np.float64) - target)
    corrected_error = np.abs(corrected.astype(np.float64) - target)
    per_world = []
    for parent in sorted(np.unique(parents).tolist()):
        mask = parents == parent
        prior_mae = float(prior_error[mask].mean())
        corrected_mae = float(corrected_error[mask].mean())
        per_world.append(
            {
                "parent_id": str(parent),
                "sequences": int(mask.sum()),
                "current_mae_deg": float(current_error[mask].mean()),
                "five_frame_prior_mae_deg": prior_mae,
                "corrected_mae_deg": corrected_mae,
                "relative_improvement_over_five_frame_prior": (prior_mae - corrected_mae) / prior_mae,
            }
        )
    prior_mae = float(prior_error.mean())
    corrected_mae = float(corrected_error.mean())
    return {
        "sequences": len(target),
        "current_frame_mae_deg": float(current_error.mean()),
        "five_frame_prior_mae_deg": prior_mae,
        "corrected_mae_deg": corrected_mae,
        "relative_improvement_over_five_frame_prior": (prior_mae - corrected_mae) / prior_mae,
        "per_world": per_world,
    }


@torch.no_grad()
def _predict(model: PhysicsGuidedSlopeResidualNet, features: np.ndarray, prior: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predicted = []
    scales = []
    for start in range(0, len(features), 4096):
        stop = min(start + 4096, len(features))
        output = model(torch.from_numpy(features[start:stop]).to(device), torch.from_numpy(prior[start:stop]).to(device))
        predicted.append(output["slope_deg"].float().cpu().numpy())
        scales.append(output["error_scale_deg"].float().cpu().numpy())
    return np.concatenate(predicted), np.concatenate(scales)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--corrective-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset_run.resolve()
    corrective = args.corrective_run.resolve()
    root = dataset / "artifacts/dataset/validation"
    paths = sorted(root.glob("*.zarr"))
    if tuple(path.stem for path in paths) != EXPECTED_PARENTS:
        raise RuntimeError("C09 corrective world set drift")
    normalization = json.loads((corrective / "artifacts/slope_corrective_cache/normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    predictor = RangeGeometryBaseline()
    feature_blocks = []
    prior_blocks = []
    current_blocks = []
    target_blocks = []
    index_blocks = []
    parent_blocks = []
    frame_total = 0
    world_evidence = []
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        parent = path.stem
        if group.attrs.get("split") != "validation" or group.attrs.get("parent_id") != parent:
            raise RuntimeError(f"C09 corrective shard identity drift: {parent}")
        frame_count = int(group["range_m"].shape[0])
        sequence_count = int(group["local_frame_references"].shape[0])
        references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
        validate_causal_references(
            references,
            np.asarray(group["global_frame_references"][:], dtype=np.int64),
            np.asarray(group["global_frame_index"][:], dtype=np.int64),
            np.asarray(group["local_frame_index"][:], dtype=np.int64),
            frame_count=frame_count,
        )
        geometry = np.asarray(group["geometry"][:], dtype=np.float32)
        mask = np.asarray(group["geometry_valid_mask"][:], dtype=np.uint8)
        if geometry.shape != (sequence_count, 4) or np.any(mask[:, 2] != 1) or not np.all(np.isfinite(geometry[:, 2])):
            raise RuntimeError(f"C09 corrective slope teacher drift: {parent}")
        frame_features = np.empty((frame_count, RAW_FEATURES_PER_FRAME), dtype=np.float32)
        for frame in range(frame_count):
            frame_features[frame] = slope_frame_features(
                predictor.predict(np.asarray(group["range_m"][frame]), np.asarray(group["valid_mask"][frame]))
            )
        sequence_features = frame_features[references]
        prior = sequence_features[:, :, 0].mean(axis=1, dtype=np.float64).astype(np.float32)
        feature_blocks.append(sequence_features)
        prior_blocks.append(prior)
        current_blocks.append(sequence_features[:, -1, 0].astype(np.float32))
        target_blocks.append(geometry[:, 2])
        index_blocks.append(np.asarray(group["global_sequence_index"][:], dtype=np.int64))
        parent_blocks.append(np.full(sequence_count, parent, dtype="U64"))
        frame_total += frame_count
        np.savez_compressed(
            output / f"{parent}_features.npz",
            features=sequence_features,
            prior_slope_deg=prior,
            current_slope_deg=sequence_features[:, -1, 0],
            target_slope_deg=geometry[:, 2],
            global_sequence_index=index_blocks[-1],
            local_frame_references=references.astype(np.int32),
        )
        world_evidence.append({"parent_id": parent, "unique_frames": frame_count, "sequences": sequence_count, "source_shard": str(path.relative_to(dataset))})
    raw_features = np.concatenate(feature_blocks)
    prior = np.concatenate(prior_blocks)
    current = np.concatenate(current_blocks)
    target = np.concatenate(target_blocks)
    indices = np.concatenate(index_blocks)
    parents = np.concatenate(parent_blocks)
    if len(target) != 24462 or frame_total != 32678 or len(np.unique(indices)) != len(indices):
        raise RuntimeError("C09 corrective exact count or identity drift")
    features = normalize_features(raw_features, mean, scale)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_rows = []
    for seed in (0, 1, 2):
        checkpoint_path = corrective / f"artifacts/models/seed{seed}/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != "gse_slope_corrective_checkpoint_v1" or checkpoint.get("seed") != seed:
            raise RuntimeError(f"corrective checkpoint identity drift: seed {seed}")
        model = PhysicsGuidedSlopeResidualNet().to(device)
        model.load_state_dict(checkpoint["model"])
        corrected, error_scale = _predict(model, features, prior, device)
        metrics = _metrics(target, current, prior, corrected, parents)
        metrics.update({"seed": seed, "checkpoint_epoch": int(checkpoint["epoch"]), "predicted_error_scale_mean_deg": float(error_scale.mean())})
        seed_rows.append(metrics)
        np.savez_compressed(
            output / f"seed{seed}_outputs.npz",
            global_sequence_index=indices,
            parent_id=parents,
            target_slope_deg=target,
            current_slope_deg=current,
            five_frame_prior_slope_deg=prior,
            corrected_slope_deg=corrected,
            predicted_error_scale_deg=error_scale,
        )
        (output / f"seed{seed}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "gse_slope_corrective_c09_evaluation_v1",
        "overall_status": PASS_STATUS,
        "validation_worlds": 10,
        "validation_sequences": 24462,
        "validation_unique_frames": 32678,
        "seeds": seed_rows,
        "world_evidence": world_evidence,
        "normalization_source": str((corrective / "artifacts/slope_corrective_cache/normalization.json").relative_to(PROJECT_ROOT)),
        "optimizer_steps": 0,
        "model_updates": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": PASS_STATUS, "seeds": [{"seed": row["seed"], "corrected_mae_deg": row["corrected_mae_deg"]} for row in seed_rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
