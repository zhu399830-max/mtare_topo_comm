from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learning.structural_learning.dataset import StructuralSurfaceDataset
from learning.structural_learning.feasibility_model import TinySurfaceCompletionNet


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default="results/structural_dataset_v3")
    parser.add_argument("--pairs", type=int, default=4096)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = torch.load(run_dir / "checkpoint" / "best.pt", map_location="cuda", weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    kwargs = {"base_channels": int(model_cfg["base_channels"]), "latent_dim": int(model_cfg["latent_dim"])}
    model_a = TinySurfaceCompletionNet(**kwargs).cuda().eval()
    model_b = TinySurfaceCompletionNet(**kwargs).cuda().eval()
    model_a.load_state_dict(checkpoint["model"]); model_b.load_state_dict(checkpoint["model"])
    dataset = StructuralSurfaceDataset(args.dataset_dir, "val")
    features, counts, densities, teacher_masks = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), 32):
            samples = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
            x_np = np.stack([sample["input_surface"] for sample in samples])
            x = torch.from_numpy(x_np).cuda()
            pred_a, feat = model_a(x, return_features=True)
            pred_b = model_b(x)
            if start == 0:
                restore_error = float(torch.max(torch.abs(pred_a - pred_b)))
            features.append(torch.nn.functional.normalize(feat, dim=1).cpu().numpy())
            mask = x_np[:, 0] > 0.5
            counts.extend(mask.sum(axis=(1, 2)).astype(float).tolist())
            densities.extend([float(x_np[i, 1][mask[i]].mean()) for i in range(len(samples))])
            teacher_masks.extend([(sample["teacher_surface"][0] > 0.5).reshape(-1) for sample in samples])
    features = np.concatenate(features); counts = np.asarray(counts); densities = np.asarray(densities); teacher_masks = np.stack(teacher_masks)
    rng = np.random.default_rng(20260807)
    left = rng.integers(0, len(dataset), size=args.pairs)
    right = rng.integers(0, len(dataset), size=args.pairs)
    valid = left != right; left, right = left[valid], right[valid]
    feature_distance = 1.0 - np.sum(features[left] * features[right], axis=1)
    count_difference = np.abs(counts[left] - counts[right]) / np.maximum(np.maximum(counts[left], counts[right]), 1.0)
    density_difference = np.abs(densities[left] - densities[right])
    intersection = np.logical_and(teacher_masks[left], teacher_masks[right]).sum(axis=1)
    union = np.logical_or(teacher_masks[left], teacher_masks[right]).sum(axis=1)
    geometry_distance = 1.0 - intersection / np.maximum(union, 1)
    output = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_restore_max_abs_error": restore_error,
        "checkpoint_restore_exact": bool(restore_error == 0.0),
        "validation_samples": len(dataset),
        "pair_count": int(len(left)),
        "spearman_feature_vs_teacher_geometry_distance": spearman(feature_distance, geometry_distance),
        "spearman_feature_vs_input_surface_count_difference": spearman(feature_distance, count_difference),
        "spearman_feature_vs_input_density_difference": spearman(feature_distance, density_difference),
        "interpretation": "Correlations are descriptive only; stronger geometry correlation than count/density correlation argues against a purely scalar-density representation.",
    }
    path = run_dir / "metrics" / "representation_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
