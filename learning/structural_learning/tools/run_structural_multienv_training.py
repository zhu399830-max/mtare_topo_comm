from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, mannwhitneyu, rankdata
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.dataset import StructuralSurfaceDataset
from learning.structural_learning.feasibility_model import completion_loss
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.tools.run_bottleneck_feasibility import make_perturbations


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: np.ndarray | list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_surface": torch.from_numpy(np.stack([sample["input_surface"] for sample in samples])),
        "teacher_surface": torch.from_numpy(np.stack([sample["teacher_surface"] for sample in samples])),
    }


def select_channels(x: torch.Tensor, channels: list[int]) -> torch.Tensor:
    return x[:, channels]


def count_surface(prediction: torch.Tensor, teacher: torch.Tensor, threshold: float) -> np.ndarray:
    pred = prediction[:, 0] >= threshold
    target = teacher[:, 0] >= 0.5
    return np.asarray(
        [
            float((pred & target).sum()),
            float((pred & ~target).sum()),
            float((~pred & target).sum()),
        ]
    )


def surface_scores(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn = counts
    return {
        "iou": float(tp / max(tp + fp + fn, 1.0)),
        "dice": float(2.0 * tp / max(2.0 * tp + fp + fn, 1.0)),
        "precision": float(tp / max(tp + fp, 1.0)),
        "recall": float(tp / max(tp + fn, 1.0)),
    }


def attribute_error(prediction: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    mask = teacher[:, :1] >= 0.5
    denom = float(mask.sum().item() * 3 + 1)
    err = torch.abs(prediction[:, 1:] - teacher[:, 1:]) * mask
    return {
        "attribute_mae": float(err.sum().item() / denom),
        "mean_height_mae": float((torch.abs(prediction[:, 2:3] - teacher[:, 2:3]) * mask).sum().item() / max(float(mask.sum().item()), 1.0)),
        "height_span_mae": float((torch.abs(prediction[:, 3:4] - teacher[:, 3:4]) * mask).sum().item() / max(float(mask.sum().item()), 1.0)),
    }


@torch.no_grad()
def completion_eval(model: ForcedGlobalBottleneckNet, dataset: Dataset, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), collate_fn=collate)
    model.eval()
    losses: list[float] = []
    model_counts = np.zeros(3)
    input_counts = np.zeros(3)
    model_attr = []
    input_attr = []
    for batch in loader:
        x4 = batch["input_surface"].to(device)
        teacher = batch["teacher_surface"].to(device)
        prediction = model(select_channels(x4, list(cfg["model"]["input_channels"])))
        loss, _ = completion_loss(prediction, teacher, float(cfg["loss"]["positive_weight"]))
        losses.append(float(loss))
        model_counts += count_surface(prediction, teacher, float(cfg["audit"]["surface_threshold"]))
        input_counts += count_surface(x4, teacher, float(cfg["audit"]["surface_threshold"]))
        model_attr.append(attribute_error(prediction, teacher))
        input_attr.append(attribute_error(x4, teacher))
    return {
        "loss": float(np.mean(losses)),
        "model": {**surface_scores(model_counts), **{key: float(np.mean([row[key] for row in model_attr])) for key in model_attr[0]}},
        "input_copy_baseline": {**surface_scores(input_counts), **{key: float(np.mean([row[key] for row in input_attr])) for key in input_attr[0]}},
    }


def degrade(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    out = x.clone()
    batch, _, height, width = out.shape
    rows, cols = torch.meshgrid(torch.arange(height, device=x.device), torch.arange(width, device=x.device), indexing="ij")
    cy = (height - 1) / 2
    cx = (width - 1) / 2
    for i in range(batch):
        mode = i % 4
        if mode == 0:
            keep = torch.rand((height, width), generator=generator, device=x.device) > 0.40
        elif mode == 1:
            angle = torch.atan2(cy - rows, cols - cx)
            center = (torch.rand((), generator=generator, device=x.device) * 2 - 1) * math.pi
            delta = torch.atan2(torch.sin(angle - center), torch.cos(angle - center))
            keep = torch.abs(delta) > math.radians(35)
        elif mode == 2:
            radius = torch.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
            limit = torch.randint(30, 46, (1,), generator=generator, device=x.device)
            keep = radius <= limit
        else:
            probability = torch.linspace(0.35, 0.9, width, device=x.device)[None, :].expand(height, width)
            keep = torch.rand((height, width), generator=generator, device=x.device) < probability
        out[i] *= keep
    return out


def geometry_similarity_loss(latent: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    z = F.normalize(latent, dim=1)
    feature_similarity = (z @ z.T + 1.0) / 2.0
    masks = teacher[:, 0].flatten(1)
    intersection = masks @ masks.T
    sums = masks.sum(1)
    union = sums[:, None] + sums[None, :] - intersection
    teacher_iou = intersection / (union + 1.0)
    upper = torch.triu(torch.ones_like(feature_similarity, dtype=torch.bool), diagonal=1)
    return F.mse_loss(feature_similarity[upper], teacher_iou[upper])


def train_step(model: ForcedGlobalBottleneckNet, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, device: torch.device, cfg: dict[str, Any], generator: torch.Generator, constraints: bool = True) -> dict[str, float]:
    model.train()
    x4 = batch["input_surface"].to(device)
    x = select_channels(x4, list(cfg["model"]["input_channels"]))
    teacher = batch["teacher_surface"].to(device)
    optimizer.zero_grad(set_to_none=True)
    pred, z = model(x, return_features=True)
    base, parts = completion_loss(pred, teacher, float(cfg["loss"]["positive_weight"]))
    total = base
    result = {"completion": float(base.detach()), "bce": float(parts["bce"]), "dice": float(parts["dice"]), "attribute": float(parts["attribute"])}
    if constraints:
        pred2, z2 = model(degrade(x, generator), return_features=True)
        degraded, _ = completion_loss(pred2, teacher, float(cfg["loss"]["positive_weight"]))
        consistency = (1.0 - F.cosine_similarity(F.normalize(z, dim=1), F.normalize(z2, dim=1), dim=1)).mean()
        geometry = geometry_similarity_loss(z, teacher)
        total = base + float(cfg["loss"]["degraded_completion_weight"]) * degraded + float(cfg["loss"]["latent_consistency_weight"]) * consistency + float(cfg["loss"]["teacher_geometry_weight"]) * geometry
        result.update({"degraded_completion": float(degraded.detach()), "consistency": float(consistency.detach()), "geometry": float(geometry.detach())})
    if not torch.isfinite(total):
        raise RuntimeError("non-finite train loss")
    total.backward()
    optimizer.step()
    result["total"] = float(total.detach())
    return result


def load_base(path: Path, cfg: dict[str, Any], device: torch.device) -> ForcedGlobalBottleneckNet:
    model = ForcedGlobalBottleneckNet(input_channels=len(cfg["model"]["input_channels"]), latent_dim=int(cfg["model"]["latent_dim"]), base_channels=int(cfg["model"]["base_channels"])).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_projection(path: Path, cfg: dict[str, Any], device: torch.device) -> StructuralProjectionHead:
    projection = StructuralProjectionHead(int(cfg["model"]["latent_dim"]), int(cfg["projection"]["hidden_dim"]), int(cfg["projection"]["output_dim"])).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    projection.load_state_dict(checkpoint["projection"])
    projection.eval()
    return projection


@torch.no_grad()
def encode_base(model: ForcedGlobalBottleneckNet, surfaces: np.ndarray, cfg: dict[str, Any], device: torch.device, batch_size: int = 64) -> np.ndarray:
    outputs = []
    channels = list(cfg["model"]["input_channels"])
    for start in range(0, len(surfaces), batch_size):
        batch = torch.from_numpy(surfaces[start : start + batch_size, channels]).to(device)
        outputs.append(F.normalize(model.encode(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def project_latent(projection: StructuralProjectionHead, latent: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    for start in range(0, len(latent), 256):
        outputs.append(projection(torch.from_numpy(latent[start : start + 256]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def encode_perturbed_base(model: ForcedGlobalBottleneckNet, surfaces: np.ndarray, cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    outputs = []
    channels = list(cfg["model"]["input_channels"])
    for start in range(0, len(surfaces), 64):
        batch = torch.from_numpy(surfaces[start : start + 64, channels]).to(device)
        changed = make_perturbations(batch)["random_surface_drop"]
        outputs.append(F.normalize(model.encode(changed), dim=1).cpu().numpy())
    return np.concatenate(outputs)


def surface_descriptors(surfaces: np.ndarray, device: torch.device) -> np.ndarray:
    mask = torch.from_numpy(surfaces[:, :1]).to(device)
    mask = F.max_pool2d((mask > 0.5).float(), kernel_size=3, stride=1, padding=1)
    scales = []
    for size in (20, 10, 5):
        pooled = F.adaptive_avg_pool2d(mask, (size, size)).flatten(1)
        scales.append(F.normalize(pooled, dim=1))
    descriptor = F.normalize(torch.cat(scales, dim=1), dim=1)
    return descriptor.cpu().numpy()


def train_projection(base: ForcedGlobalBottleneckNet, train_samples: list[dict[str, Any]], cfg: dict[str, Any], device: torch.device) -> tuple[StructuralProjectionHead, dict[str, Any], np.ndarray, np.ndarray]:
    inputs = np.stack([sample["input_surface"] for sample in train_samples]).astype(np.float32)
    teachers = np.stack([sample["teacher_surface"] for sample in train_samples]).astype(np.float32)
    base_latent = encode_base(base, inputs, cfg, device)
    perturbed_latent = encode_perturbed_base(base, inputs, cfg, device)
    teacher_descriptor = surface_descriptors(teachers, device)
    centered = teacher_descriptor - teacher_descriptor.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[: int(cfg["projection"]["teacher_pca_dim"])]
    target = centered @ components.T
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    projection = StructuralProjectionHead(int(cfg["model"]["latent_dim"]), int(cfg["projection"]["hidden_dim"]), int(cfg["projection"]["output_dim"])).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=float(cfg["projection"]["learning_rate"]), weight_decay=float(cfg["projection"]["weight_decay"]))
    base_tensor = torch.from_numpy(base_latent).to(device)
    pert_tensor = torch.from_numpy(perturbed_latent).to(device)
    target_tensor = torch.from_numpy(target.astype(np.float32)).to(device)
    history = []
    for step in range(1, int(cfg["projection"]["steps"]) + 1):
        projection.train()
        optimizer.zero_grad(set_to_none=True)
        pred = projection(base_tensor)
        altered = projection(pert_tensor)
        cosine = (1.0 - (pred * target_tensor).sum(1)).mean()
        mse = F.mse_loss(pred, target_tensor)
        consistency = (1.0 - (pred * altered).sum(1)).mean()
        total = cosine + float(cfg["projection"]["target_mse_weight"]) * mse + float(cfg["projection"]["perturbation_consistency_weight"]) * consistency
        if not torch.isfinite(total):
            raise RuntimeError("non-finite projection loss")
        total.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == int(cfg["projection"]["steps"]):
            history.append({"step": step, "total": float(total.detach()), "teacher_cosine": float(cosine.detach()), "target_mse": float(mse.detach()), "perturbation_consistency": float(consistency.detach())})
    projection.eval()
    return projection, {"history": history, "initial_loss": history[0], "final_loss": history[-1], "pca_source": "train split only: tunnel + urban"}, teacher_descriptor.mean(axis=0), components


def load_samples(root: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "path": str(path),
                    "input_surface": data["input_surface"].astype(np.float32),
                    "teacher_surface": data["teacher_surface"].astype(np.float32),
                    "pose": data["center_pose"].astype(np.float64),
                    "stamp_ns": int(data["stamp_ns"]),
                    "dataset_source": str(data["dataset_source"]),
                    "environment": str(data["environment"]),
                    "trajectory": str(data["robot_or_trajectory"]),
                    "robot": str(data["robot_or_trajectory"]),
                    "raw_point_count": int(data["raw_point_count"]),
                    "support": float(data["geometry_input_teacher_support_ratio"]),
                    "density_proxy": float(data["input_surface"][1][data["input_surface"][0] > 0.5].mean()) if np.any(data["input_surface"][0] > 0.5) else 0.0,
                }
            )
    if not rows:
        raise RuntimeError(f"no samples in {root / split}")
    return rows


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.sum(left * right, axis=-1), 0.0, 2.0)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def distribution_test(stable: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    stable = np.asarray(stable, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)
    stable = stable[np.isfinite(stable)]
    changed = changed[np.isfinite(changed)]
    if len(stable) == 0 or len(changed) == 0:
        return {
            "stable": describe(stable),
            "changed": describe(changed),
            "median_ratio_changed_over_stable": 0.0,
            "probability_changed_greater_than_stable": 0.0,
            "one_sided_p_value": 1.0,
            "distribution_overlap_one_minus_ks": 1.0,
        }
    test = mannwhitneyu(changed, stable, alternative="greater")
    ks = ks_2samp(stable, changed, alternative="two-sided", method="auto")
    stable_sorted = np.sort(stable)
    greater_count = np.searchsorted(stable_sorted, changed, side="left").sum()
    pair_count = len(stable) * len(changed)
    return {
        "stable": describe(stable),
        "changed": describe(changed),
        "median_ratio_changed_over_stable": float(np.median(changed) / max(np.median(stable), 1e-8)),
        "probability_changed_greater_than_stable": float(greater_count / max(pair_count, 1)),
        "one_sided_p_value": float(test.pvalue),
        "distribution_overlap_one_minus_ks": float(1.0 - ks.statistic),
    }


def adjacent_edges(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> list[tuple[int, int, float, float]]:
    edges = []
    for trajectory in sorted({sample["trajectory"] for sample in samples}):
        order = sorted((index for index, sample in enumerate(samples) if sample["trajectory"] == trajectory), key=lambda index: samples[index]["stamp_ns"])
        for left, right in zip(order[:-1], order[1:]):
            dt = (samples[right]["stamp_ns"] - samples[left]["stamp_ns"]) / 1e9
            step = float(np.linalg.norm(samples[right]["pose"][:3] - samples[left]["pose"][:3]))
            if 0.0 < dt <= float(cfg["audit"]["trajectory_max_dt_s"]) and step <= float(cfg["audit"]["trajectory_max_step_m"]):
                edges.append((left, right, dt, step))
    return edges


def retrieval_audit(latent: np.ndarray, teacher: np.ndarray, samples: list[dict[str, Any]], minimum_distance: float, k: int, similar_threshold: float) -> dict[str, Any]:
    xyz = np.stack([sample["pose"][:3] for sample in samples])
    precisions = []
    top_geometry = []
    all_geometry = []
    for query in range(len(samples)):
        candidates = np.flatnonzero(np.linalg.norm(xyz - xyz[query], axis=1) >= minimum_distance)
        if len(candidates) < k:
            continue
        latent_distance = cosine_distance(latent[candidates], latent[query][None])
        teacher_distance = cosine_distance(teacher[candidates], teacher[query][None])
        selected = np.argsort(latent_distance)[:k]
        top_geometry.extend(teacher_distance[selected])
        all_geometry.extend(teacher_distance)
        precisions.append(float(np.mean(teacher_distance[selected] <= similar_threshold)))
    prevalence = float(np.mean(np.asarray(all_geometry) <= similar_threshold)) if all_geometry else 0.0
    return {"queries": len(precisions), "k": k, "top_k_similar_precision": float(np.mean(precisions)) if precisions else 0.0, "random_candidate_similar_prevalence": prevalence, "precision_lift": float((np.mean(precisions) if precisions else 0.0) / max(prevalence, 1e-8)), "top_k_teacher_geometry_distance": describe(top_geometry)}


def representation_eval(name: str, samples: list[dict[str, Any]], base: ForcedGlobalBottleneckNet, projection: StructuralProjectionHead, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    inputs = np.stack([sample["input_surface"] for sample in samples]).astype(np.float32)
    teachers = np.stack([sample["teacher_surface"] for sample in samples]).astype(np.float32)
    latent = project_latent(projection, encode_base(base, inputs, cfg, device), device)
    input_desc = surface_descriptors(inputs, device)
    teacher_desc = surface_descriptors(teachers, device)
    cells = (inputs[:, 0] > 0.5).sum(axis=(1, 2)).astype(np.float64)
    density = np.asarray([sample["density_proxy"] for sample in samples], dtype=np.float64)
    edges = adjacent_edges(samples, cfg)
    if edges:
        edge_left = np.asarray([edge[0] for edge in edges])
        edge_right = np.asarray([edge[1] for edge in edges])
        teacher_edge = cosine_distance(teacher_desc[edge_left], teacher_desc[edge_right])
        latent_edge = cosine_distance(latent[edge_left], latent[edge_right])
        stable = teacher_edge <= np.quantile(teacher_edge, float(cfg["audit"]["stable_quantile"]))
        changed = teacher_edge >= np.quantile(teacher_edge, float(cfg["audit"]["change_quantile"]))
        stable_change = distribution_test(latent_edge[stable], latent_edge[changed])
    else:
        teacher_edge = latent_edge = np.asarray([])
        stable_change = {"stable": {"count": 0}, "changed": {"count": 0}, "median_ratio_changed_over_stable": 0.0, "probability_changed_greater_than_stable": 0.0, "distribution_overlap_one_minus_ks": 1.0}
    xyz = np.stack([sample["pose"][:3] for sample in samples])
    left, right = np.triu_indices(len(samples), k=1)
    spatial = np.linalg.norm(xyz[left] - xyz[right], axis=1)
    keep = spatial >= float(cfg["audit"]["cross_location_min_distance_m"])
    left, right, spatial = left[keep], right[keep], spatial[keep]
    pair_latent = cosine_distance(latent[left], latent[right])
    pair_teacher = cosine_distance(teacher_desc[left], teacher_desc[right])
    pair_input = cosine_distance(input_desc[left], input_desc[right])
    pair_cells = np.abs(cells[left] - cells[right]) / np.maximum(np.maximum(cells[left], cells[right]), 1)
    pair_density = np.abs(density[left] - density[right])
    pair_env = np.asarray([samples[l]["environment"] != samples[r]["environment"] for l, r in zip(left, right)], dtype=np.float64)
    pair_traj = np.asarray([samples[l]["trajectory"] != samples[r]["trajectory"] for l, r in zip(left, right)], dtype=np.float64)
    similar_threshold = float(np.quantile(pair_teacher, float(cfg["audit"]["cross_similar_quantile"]))) if len(pair_teacher) else 0.0
    different_threshold = float(np.quantile(pair_teacher, float(cfg["audit"]["cross_different_quantile"]))) if len(pair_teacher) else 0.0
    similar = pair_teacher <= similar_threshold
    different = pair_teacher >= different_threshold
    perturb = perturbation_eval(base, projection, inputs, pair_latent[different], cfg, device)
    return {
        "name": name,
        "samples": len(samples),
        "valid_edges": len(edges),
        "stable_change": stable_change,
        "teacher_latent_edge_spearman": spearman(teacher_edge, latent_edge) if len(edges) else 0.0,
        "cross_location": {
            "pair_count": int(len(pair_latent)),
            "similar_structure_distance": describe(pair_latent[similar]),
            "different_structure_distance": describe(pair_latent[different]),
            "distance_test": distribution_test(pair_latent[similar], pair_latent[different]) if np.any(similar) and np.any(different) else {},
            "retrieval": retrieval_audit(latent, teacher_desc, samples, float(cfg["audit"]["cross_location_min_distance_m"]), int(cfg["audit"]["retrieval_k"]), similar_threshold),
        },
        "nuisance_spearman": {
            "teacher_geometry_difference": spearman(pair_latent, pair_teacher),
            "input_coverage_shape_difference": spearman(pair_latent, pair_input),
            "surface_cell_count_difference": spearman(pair_latent, pair_cells),
            "density_proxy_difference": spearman(pair_latent, pair_density),
            "environment_identity_difference": spearman(pair_latent, pair_env),
            "trajectory_identity_difference": spearman(pair_latent, pair_traj),
        },
        "perturbation": perturb,
    }


@torch.no_grad()
def perturbation_eval(base: ForcedGlobalBottleneckNet, projection: StructuralProjectionHead, inputs: np.ndarray, reference: np.ndarray, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    indices = np.linspace(0, len(inputs) - 1, min(int(cfg["audit"]["perturbation_samples"]), len(inputs)), dtype=int)
    batch = torch.from_numpy(inputs[indices][:, list(cfg["model"]["input_channels"])]).to(device)
    original = projection(base.encode(batch))
    result = {}
    for name, changed in make_perturbations(batch).items():
        altered = projection(base.encode(changed))
        dist = torch.clamp(1.0 - (original * altered).sum(1), min=0).cpu().numpy()
        ref = np.resize(reference, len(dist)) if len(reference) else np.ones(len(dist), dtype=np.float64)
        result[name] = {"same_sample": describe(dist), "different_structure_reference": describe(ref), "same_less_fraction": float(np.mean(dist < ref))}
    return result


class MTAREWorld(Dataset):
    def __init__(self, root: Path, world: str) -> None:
        self.files = sorted((root / world / "samples").glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"no M-TARE samples for {world}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with np.load(self.files[index], allow_pickle=False) as data:
            return {
                "input_surface": data["input_surface"].astype(np.float32),
                "teacher_surface": data["teacher_surface"].astype(np.float32),
                "pose": data["center_pose"].astype(np.float64),
                "stamp_ns": int(data["stamp_ns"]),
                "environment": str(data["world"]),
                "trajectory": str(data["world"]),
                "robot": str(data["world"]),
                "density_proxy": float(data["input_surface"][1][data["input_surface"][0] > 0.5].mean()) if np.any(data["input_surface"][0] > 0.5) else 0.0,
            }


def mtare_samples(root: Path, world: str) -> list[dict[str, Any]]:
    dataset = MTAREWorld(root, world)
    return [dataset[index] for index in range(len(dataset))]


def ku_quality_groups(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[int]]:
    high = float(cfg["ku_quality_groups"]["high_support_min"])
    medium = float(cfg["ku_quality_groups"]["medium_support_min"])
    groups = {"high_support": [], "medium_support": [], "low_support": []}
    for index, sample in enumerate(samples):
        if sample["support"] >= high:
            groups["high_support"].append(index)
        elif sample["support"] >= medium:
            groups["medium_support"].append(index)
        else:
            groups["low_support"].append(index)
    return groups


def save_completion_preview(path: Path, dataset: Dataset, old_model: ForcedGlobalBottleneckNet, new_model: ForcedGlobalBottleneckNet, cfg: dict[str, Any], device: torch.device) -> None:
    indices = np.linspace(0, len(dataset) - 1, min(int(cfg["audit"]["preview_samples"]), len(dataset)), dtype=int)
    samples = [dataset[int(index)] for index in indices]
    x4 = torch.from_numpy(np.stack([sample["input_surface"] for sample in samples])).to(device)
    teacher = torch.from_numpy(np.stack([sample["teacher_surface"] for sample in samples])).to(device)
    x = select_channels(x4, list(cfg["model"]["input_channels"]))
    with torch.no_grad():
        old_pred = old_model(x)
        new_pred = new_model(x)
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3 * rows), squeeze=False)
    for row in range(rows):
        for ax, tensor, title in (
            (axes[row, 0], x4[row, 0], "input"),
            (axes[row, 1], old_pred[row, 0], "old"),
            (axes[row, 2], new_pred[row, 0], "new"),
            (axes[row, 3], teacher[row, 0], "teacher"),
        ):
            ax.imshow(tensor.detach().cpu(), origin="upper", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/structural_multienv_training.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["experiment"]["output_root"]) / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    for name in ("config", "train_logs", "encoder_checkpoint", "projection_checkpoint", "completion_metrics", "representation_metrics", "nuisance_audit", "perlin2_results", "ku_results", "mtare_comparison", "previews"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, root / "config" / "config.yaml")
    write_json(root / "config" / "provenance.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": command_output(["git", "rev-parse", "HEAD"]), "git_status": command_output(["git", "status", "--short"])})
    (root / "config" / "package_list.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"])["stdout"], encoding="utf-8")
    seed_all(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]))
    device = torch.device("cuda")
    dataset_root = Path(cfg["experiment"]["dataset_dir"])
    train_dataset = StructuralSurfaceDataset(dataset_root, "train")
    val_dataset = StructuralSurfaceDataset(dataset_root, "val")
    test_dataset = StructuralSurfaceDataset(dataset_root, "test")
    train_samples = load_samples(dataset_root, "train")
    val_samples = load_samples(dataset_root, "val")
    test_samples = load_samples(dataset_root, "test")
    old_base = load_base(Path(cfg["experiment"]["old_encoder_checkpoint"]), cfg, device)
    old_projection = load_projection(Path(cfg["experiment"]["old_projection_checkpoint"]), cfg, device)
    for parameter in list(old_base.parameters()) + list(old_projection.parameters()):
        parameter.requires_grad_(False)
    model_kwargs = {"input_channels": len(cfg["model"]["input_channels"]), "latent_dim": int(cfg["model"]["latent_dim"]), "base_channels": int(cfg["model"]["base_channels"])}
    fixed_indices = np.linspace(0, len(train_dataset) - 1, int(cfg["overfit"]["samples"]), dtype=int).tolist()
    fixed = Subset(train_dataset, fixed_indices)
    fixed_batch = next(iter(DataLoader(fixed, batch_size=int(cfg["overfit"]["batch_size"]), collate_fn=collate)))
    overfit_model = ForcedGlobalBottleneckNet(**model_kwargs).to(device)
    overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), lr=float(cfg["overfit"]["learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(int(cfg["experiment"]["seed"]))
    overfit_losses = []
    for _ in range(int(cfg["overfit"]["steps"])):
        overfit_losses.append(train_step(overfit_model, fixed_batch, overfit_optimizer, device, cfg, generator, constraints=False)["total"])
    overfit = {"samples": len(fixed), "initial_loss": overfit_losses[0], "middle_loss": overfit_losses[len(overfit_losses) // 2], "final_loss": overfit_losses[-1], "pass": bool(overfit_losses[-1] < overfit_losses[0] * 0.45), "finite": bool(np.isfinite(overfit_losses).all())}
    write_json(root / "train_logs" / "overfit.json", overfit)
    new_base = ForcedGlobalBottleneckNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(new_base.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"]["weight_decay"]))
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, collate_fn=collate, generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])))
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, collate_fn=collate)
    history = []
    best_loss = float("inf")
    train_generator = torch.Generator(device=device).manual_seed(int(cfg["experiment"]["seed"]) + 1)
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        rows = [train_step(new_base, batch, optimizer, device, cfg, train_generator, constraints=True) for batch in train_loader]
        val_metrics = completion_eval(new_base, val_dataset, device, cfg)
        row = {
            "epoch": epoch,
            "train_completion_loss": float(np.mean([item["completion"] for item in rows])),
            "train_total_loss": float(np.mean([item["total"] for item in rows])),
            "perlin2_val_loss": val_metrics["loss"],
            "perlin2_val_iou": val_metrics["model"]["iou"],
            "perlin2_val_dice": val_metrics["model"]["dice"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model": new_base.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg}, root / "encoder_checkpoint" / "best.pt")
    torch.save({"model": new_base.state_dict(), "optimizer": optimizer.state_dict(), "epoch": len(history), "config": cfg}, root / "encoder_checkpoint" / "last.pt")
    best = torch.load(root / "encoder_checkpoint" / "best.pt", map_location=device, weights_only=False)
    new_base.load_state_dict(best["model"])
    new_base.eval()
    reloaded = load_base(root / "encoder_checkpoint" / "best.pt", cfg, device)
    probe = next(iter(val_loader))["input_surface"][:4].to(device)
    with torch.no_grad():
        reload_max_abs_diff = float(torch.max(torch.abs(new_base(select_channels(probe, list(cfg["model"]["input_channels"]))) - reloaded(select_channels(probe, list(cfg["model"]["input_channels"]))))).cpu())
    projection, projection_log, pca_mean, pca_components = train_projection(new_base, train_samples, cfg, device)
    torch.save({"projection": projection.state_dict(), "pca_mean": pca_mean, "pca_components": pca_components, "config": cfg, "encoder_checkpoint_sha256": sha256(root / "encoder_checkpoint" / "best.pt")}, root / "projection_checkpoint" / "projection.pt")
    reloaded_projection = load_projection(root / "projection_checkpoint" / "projection.pt", cfg, device)
    with torch.no_grad():
        latent_probe = new_base.encode(select_channels(probe, list(cfg["model"]["input_channels"])))
        projection_reload_max_abs_diff = float(torch.max(torch.abs(projection(latent_probe) - reloaded_projection(latent_probe))).cpu())
    train_log = {"overfit": overfit, "history": history, "best_epoch": int(best["epoch"]), "checkpoint_reload_max_abs_diff": reload_max_abs_diff, "projection_training": projection_log, "projection_reload_max_abs_diff": projection_reload_max_abs_diff, "nan_inf": False}
    write_json(root / "train_logs" / "summary.json", train_log)
    completion = {
        "perlin2": {"old_model": completion_eval(old_base, val_dataset, device, cfg), "new_model": completion_eval(new_base, val_dataset, device, cfg)},
        "ku": {"old_model": completion_eval(old_base, test_dataset, device, cfg), "new_model": completion_eval(new_base, test_dataset, device, cfg)},
    }
    groups = ku_quality_groups(test_samples, cfg)
    completion["ku"]["quality_groups"] = {}
    for group, indices in groups.items():
        subset = Subset(test_dataset, indices)
        completion["ku"]["quality_groups"][group] = {"samples": len(indices), "old_model": completion_eval(old_base, subset, device, cfg) if indices else {}, "new_model": completion_eval(new_base, subset, device, cfg) if indices else {}}
    write_json(root / "completion_metrics" / "summary.json", completion)
    save_completion_preview(root / "previews" / "perlin2_completion_old_new.png", val_dataset, old_base, new_base, cfg, device)
    save_completion_preview(root / "previews" / "ku_completion_old_new.png", test_dataset, old_base, new_base, cfg, device)
    rep = {
        "perlin2": {"old_model": representation_eval("perlin2_old", val_samples, old_base, old_projection, cfg, device), "new_model": representation_eval("perlin2_new", val_samples, new_base, projection, cfg, device)},
        "ku": {"old_model": representation_eval("ku_old", test_samples, old_base, old_projection, cfg, device), "new_model": representation_eval("ku_new", test_samples, new_base, projection, cfg, device)},
    }
    write_json(root / "representation_metrics" / "summary.json", rep)
    write_json(root / "perlin2_results" / "summary.json", {"completion": completion["perlin2"], "representation": rep["perlin2"]})
    write_json(root / "ku_results" / "summary.json", {"completion": completion["ku"], "representation": rep["ku"]})
    nuisance = {
        "perlin2": {"old_model": rep["perlin2"]["old_model"]["nuisance_spearman"], "new_model": rep["perlin2"]["new_model"]["nuisance_spearman"]},
        "ku": {"old_model": rep["ku"]["old_model"]["nuisance_spearman"], "new_model": rep["ku"]["new_model"]["nuisance_spearman"]},
    }
    pooled = {"old_model": representation_eval("pooled_old", val_samples + test_samples, old_base, old_projection, cfg, device), "new_model": representation_eval("pooled_new", val_samples + test_samples, new_base, projection, cfg, device)}
    nuisance["pooled_perlin2_ku"] = {"old_model": pooled["old_model"]["nuisance_spearman"], "new_model": pooled["new_model"]["nuisance_spearman"]}
    write_json(root / "nuisance_audit" / "summary.json", nuisance)
    mtare_root = Path(cfg["experiment"]["mtare_dataset_dir"])
    mtare = {}
    for world in ("campus", "indoor"):
        samples = mtare_samples(mtare_root, world)
        mtare[world] = {
            "old_model": representation_eval(f"{world}_old", samples, old_base, old_projection, cfg, device),
            "new_model": representation_eval(f"{world}_new", samples, new_base, projection, cfg, device),
        }
    write_json(root / "mtare_comparison" / "summary.json", mtare)
    old_hash = {"encoder": sha256(Path(cfg["experiment"]["old_encoder_checkpoint"])), "projection": sha256(Path(cfg["experiment"]["old_projection_checkpoint"]))}
    new_hash = {"encoder": sha256(root / "encoder_checkpoint" / "best.pt"), "projection": sha256(root / "projection_checkpoint" / "projection.pt")}
    train_info = json.loads((dataset_root / "dataset_stats.json").read_text(encoding="utf-8"))
    def ratio(metric: dict[str, Any]) -> float:
        return float(metric["stable_change"]["median_ratio_changed_over_stable"])
    def geom_minus_coverage(metric: dict[str, Any]) -> float:
        s = metric["nuisance_spearman"]
        return float(s["teacher_geometry_difference"] - abs(s["input_coverage_shape_difference"]))
    improvements = {
        "perlin2_iou_delta": completion["perlin2"]["new_model"]["model"]["iou"] - completion["perlin2"]["old_model"]["model"]["iou"],
        "ku_iou_delta": completion["ku"]["new_model"]["model"]["iou"] - completion["ku"]["old_model"]["model"]["iou"],
        "ku_change_ratio_delta": ratio(rep["ku"]["new_model"]) - ratio(rep["ku"]["old_model"]),
        "campus_change_ratio_delta": ratio(mtare["campus"]["new_model"]) - ratio(mtare["campus"]["old_model"]),
        "indoor_change_ratio_delta": ratio(mtare["indoor"]["new_model"]) - ratio(mtare["indoor"]["old_model"]),
        "ku_geometry_minus_coverage_delta": geom_minus_coverage(rep["ku"]["new_model"]) - geom_minus_coverage(rep["ku"]["old_model"]),
        "campus_geometry_minus_coverage_delta": geom_minus_coverage(mtare["campus"]["new_model"]) - geom_minus_coverage(mtare["campus"]["old_model"]),
        "indoor_geometry_minus_coverage_delta": geom_minus_coverage(mtare["indoor"]["new_model"]) - geom_minus_coverage(mtare["indoor"]["old_model"]),
    }
    positive = sum(value > 0 for value in improvements.values())
    status = "MULTI_ENV_MODEL_PASS" if positive >= 6 and improvements["ku_iou_delta"] >= 0 and improvements["campus_change_ratio_delta"] > 0 and improvements["indoor_change_ratio_delta"] >= -0.05 else "MULTI_ENV_MODEL_MIXED" if positive >= 3 and completion["perlin2"]["new_model"]["model"]["iou"] > completion["perlin2"]["new_model"]["input_copy_baseline"]["iou"] else "MULTI_ENV_MODEL_FAIL"
    summary = {
        "status": status,
        "dataset": {"path": str(dataset_root), "split_counts": train_info["split_counts"], "fixed_split": True},
        "old_checkpoint_sha256": old_hash,
        "new_checkpoint_sha256": new_hash,
        "model": {"architecture": "ForcedGlobalBottleneckNet + StructuralProjectionHead", "encoder_parameters": sum(p.numel() for p in new_base.parameters()), "projection_parameters": sum(p.numel() for p in projection.parameters()), "input_channels": ["surface_mask", "mean_height", "height_span"]},
        "selection": {"encoder_best_epoch": int(best["epoch"]), "criterion": "minimum perlin2 validation completion loss", "projection_pca_source": "train only"},
        "training": train_log,
        "completion": completion,
        "representation": rep,
        "nuisance": nuisance,
        "mtare_comparison": mtare,
        "improvements": improvements,
        "environment": {"hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }
    write_json(root / "summary.json", summary)
    print(json.dumps({"result_dir": str(root), "status": status, "best_epoch": int(best["epoch"]), "improvements": improvements}, indent=2), flush=True)


if __name__ == "__main__":
    main()
