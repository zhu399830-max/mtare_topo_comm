from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import mannwhitneyu, rankdata
from torch.nn import functional as F

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.tools.run_bottleneck_feasibility import make_perturbations


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def distribution_test(stable: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    stable = np.asarray(stable, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)
    test = mannwhitneyu(changed, stable, alternative="greater")
    probability = float(np.mean(changed[:, None] > stable[None, :]))
    return {
        "stable": describe(stable),
        "changed": describe(changed),
        "median_ratio_changed_over_stable": float(np.median(changed) / max(np.median(stable), 1e-8)),
        "probability_changed_greater_than_stable": probability,
        "mann_whitney_u": float(test.statistic),
        "one_sided_p_value": float(test.pvalue),
    }


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.sum(left * right, axis=-1), 0.0, 2.0)


def load_split(root: Path, split: str) -> list[dict[str, Any]]:
    samples = []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            samples.append(
                {
                    "path": str(path),
                    "input": data["input_surface"].astype(np.float32),
                    "teacher": data["teacher_surface"].astype(np.float32),
                    "pose": data["center_pose"].astype(np.float64),
                    "stamp_ns": int(data["stamp_ns"]),
                    "robot": str(data["robot"]),
                    "key": int(data["key"]),
                    "region_id": str(data["region_id"]),
                }
            )
    if not samples:
        raise RuntimeError(f"no samples in {root / split}")
    return samples


def surface_descriptors(surfaces: np.ndarray, device: torch.device) -> np.ndarray:
    mask = torch.from_numpy(surfaces[:, :1]).to(device)
    mask = F.max_pool2d((mask > 0.5).float(), kernel_size=3, stride=1, padding=1)
    scales = []
    for size in (20, 10, 5):
        pooled = F.adaptive_avg_pool2d(mask, (size, size)).flatten(1)
        scales.append(F.normalize(pooled, dim=1))
    descriptor = F.normalize(torch.cat(scales, dim=1), dim=1)
    return descriptor.cpu().numpy()


@torch.no_grad()
def encode(model: ForcedGlobalBottleneckNet, surfaces: np.ndarray, channels: list[int], device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(surfaces), 64):
        batch = torch.from_numpy(surfaces[start : start + 64, channels]).to(device)
        latent = F.normalize(model.encode(batch), dim=1)
        output.append(latent.cpu().numpy())
    return np.concatenate(output)


def adjacent_edges(samples: list[dict[str, Any]], max_step: float, max_dt: float) -> list[tuple[int, int, float, float]]:
    result = []
    for robot in sorted({sample["robot"] for sample in samples}):
        order = sorted((i for i, sample in enumerate(samples) if sample["robot"] == robot), key=lambda i: samples[i]["stamp_ns"])
        for left, right in zip(order[:-1], order[1:]):
            dt = (samples[right]["stamp_ns"] - samples[left]["stamp_ns"]) / 1e9
            step = float(np.linalg.norm(samples[right]["pose"][:3] - samples[left]["pose"][:3]))
            if 0.0 < dt <= max_dt and step <= max_step:
                result.append((left, right, dt, step))
    return result


def stable_segments(edges: list[tuple[int, int, float, float]], stable: np.ndarray, samples: list[dict[str, Any]], minimum_edges: int) -> list[dict[str, Any]]:
    segments = []
    current: list[int] = []
    for edge_index, ((left, right, _, _), is_stable) in enumerate(zip(edges, stable)):
        continuous = current and edges[current[-1]][1] == left and samples[edges[current[-1]][1]]["robot"] == samples[left]["robot"]
        if is_stable:
            if not continuous:
                current = []
            current.append(edge_index)
        else:
            if len(current) >= minimum_edges:
                segments.append(current)
            current = []
    if len(current) >= minimum_edges:
        segments.append(current)
    return [
        {
            "robot": samples[edges[group[0]][0]]["robot"],
            "edge_count": len(group),
            "sample_count": len(group) + 1,
            "start_stamp_ns": samples[edges[group[0]][0]]["stamp_ns"],
            "end_stamp_ns": samples[edges[group[-1]][1]]["stamp_ns"],
            "keys": [samples[edges[group[0]][0]]["key"]] + [samples[edges[i][1]]["key"] for i in group],
        }
        for group in segments
    ]


def all_cross_pairs(samples: list[dict[str, Any]], minimum_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(samples), k=1)
    positions = np.stack([sample["pose"][:3] for sample in samples])
    distance = np.linalg.norm(positions[left] - positions[right], axis=1)
    keep = distance >= minimum_distance
    return left[keep], right[keep], distance[keep]


def retrieval_audit(latent: np.ndarray, teacher: np.ndarray, samples: list[dict[str, Any]], minimum_distance: float, k: int, similar_threshold: float) -> dict[str, Any]:
    positions = np.stack([sample["pose"][:3] for sample in samples])
    top_geometry = []
    all_geometry = []
    precision = []
    candidate_counts = []
    for query in range(len(samples)):
        candidates = np.flatnonzero(np.linalg.norm(positions - positions[query], axis=1) >= minimum_distance)
        if len(candidates) < k:
            continue
        latent_distance = cosine_distance(latent[candidates], latent[query][None])
        teacher_distance = cosine_distance(teacher[candidates], teacher[query][None])
        selected = np.argsort(latent_distance)[:k]
        top_geometry.extend(teacher_distance[selected])
        all_geometry.extend(teacher_distance)
        precision.append(float(np.mean(teacher_distance[selected] <= similar_threshold)))
        candidate_counts.append(len(candidates))
    prevalence = float(np.mean(np.asarray(all_geometry) <= similar_threshold))
    return {
        "queries": len(precision),
        "k": k,
        "candidate_count": describe(np.asarray(candidate_counts)),
        "top_k_teacher_geometry_distance": describe(np.asarray(top_geometry)),
        "all_candidate_teacher_geometry_distance": describe(np.asarray(all_geometry)),
        "top_k_similar_precision": float(np.mean(precision)),
        "random_candidate_similar_prevalence": prevalence,
        "precision_lift": float(np.mean(precision) / max(prevalence, 1e-8)),
    }


def save_trajectory_preview(path: Path, samples: list[dict[str, Any]], edges: list[tuple[int, int, float, float]], teacher_change: np.ndarray, latent_change: np.ndarray, stable: np.ndarray, changed: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for robot, marker in (("husky3", "o"), ("husky4", "s")):
        indices = [i for i, sample in enumerate(samples) if sample["robot"] == robot]
        xy = np.stack([samples[i]["pose"][:2] for i in indices])
        axes[0].scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.45, marker=marker, label=robot)
    for edge_index, (left, right, _, _) in enumerate(edges):
        xy = np.stack((samples[left]["pose"][:2], samples[right]["pose"][:2]))
        if changed[edge_index]:
            axes[0].plot(xy[:, 0], xy[:, 1], color="crimson", linewidth=2.2)
        elif stable[edge_index]:
            axes[0].plot(xy[:, 0], xy[:, 1], color="seagreen", linewidth=1.5)
    axes[0].set_title("Validation trajectory: stable (green), change (red)")
    axes[0].set_aspect("equal"); axes[0].legend(); axes[0].set_xlabel("world x [m]"); axes[0].set_ylabel("world y [m]")
    axes[1].scatter(teacher_change, latent_change, s=18, alpha=0.65)
    axes[1].set_xlabel("teacher multiscale geometry distance"); axes[1].set_ylabel("latent cosine distance")
    axes[1].set_title("Consecutive validation edges")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_map_examples(path: Path, samples: list[dict[str, Any]], edges: list[tuple[int, int, float, float]], values: np.ndarray, stable: np.ndarray, changed: np.ndarray, count: int) -> None:
    stable_indices = np.flatnonzero(stable)[np.argsort(values[stable])[:count]]
    changed_indices = np.flatnonzero(changed)[np.argsort(values[changed])[-count:]]
    rows = max(len(stable_indices), len(changed_indices))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    for row, (stable_edge, changed_edge) in enumerate(zip(stable_indices, changed_indices)):
        for column, (edge_index, title) in enumerate(((stable_edge, "stable"), (changed_edge, "change"))):
            sample = samples[edges[edge_index][1]]
            axes[row, column * 2].imshow(sample["input"][0], cmap="gray", origin="upper", vmin=0, vmax=1)
            axes[row, column * 2].set_title(f"{title} input, {sample['robot']}")
            axes[row, column * 2 + 1].imshow(sample["teacher"][0], cmap="gray", origin="upper", vmin=0, vmax=1)
            axes[row, column * 2 + 1].set_title(f"{title} teacher, d={values[edge_index]:.3f}")
        for axis in axes[row]:
            axis.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_cross_preview(path: Path, samples: list[dict[str, Any]], left: np.ndarray, right: np.ndarray, similar: np.ndarray, different: np.ndarray, latent_distance: np.ndarray, teacher_distance: np.ndarray, count: int) -> None:
    similar_indices = np.flatnonzero(similar)[np.argsort(latent_distance[similar])[:count]]
    different_indices = np.flatnonzero(different)[np.argsort(latent_distance[different])[-count:]]
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for row, (same_i, diff_i) in enumerate(zip(similar_indices, different_indices)):
        for column, (pair_i, label) in enumerate(((same_i, "similar"), (diff_i, "different"))):
            for side, sample_i in enumerate((left[pair_i], right[pair_i])):
                axes[row, column * 2 + side].imshow(samples[sample_i]["teacher"][0], cmap="gray", origin="upper", vmin=0, vmax=1)
                axes[row, column * 2 + side].set_title(f"{label}: L={latent_distance[pair_i]:.2f}, G={teacher_distance[pair_i]:.2f}")
                axes[row, column * 2 + side].set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


@torch.no_grad()
def perturbation_audit(model: ForcedGlobalBottleneckNet, inputs: np.ndarray, channels: list[int], device: torch.device, count: int, reference: np.ndarray) -> dict[str, Any]:
    indices = np.linspace(0, len(inputs) - 1, min(count, len(inputs)), dtype=int)
    batch = torch.from_numpy(inputs[indices][:, channels]).to(device)
    base = F.normalize(model.encode(batch), dim=1)
    result = {}
    for name, changed in make_perturbations(batch).items():
        altered = F.normalize(model.encode(changed), dim=1)
        distance = torch.clamp(1.0 - (base * altered).sum(1), min=0).cpu().numpy()
        result[name] = {
            "same_sample": describe(distance),
            "structural_change_reference": describe(np.resize(reference, len(distance))),
            "same_less_fraction": float(np.mean(distance < np.resize(reference, len(distance)))),
        }
    return result


@torch.no_grad()
def mtare_audit(model: ForcedGlobalBottleneckNet, root: Path, channels: list[int], device: torch.device) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            rows.append((int(data["stamp_ns"]), data["input_surface"].astype(np.float32), data["center_pose"].astype(np.float64), str(path)))
    rows.sort(key=lambda row: row[0])
    inputs = np.stack([row[1] for row in rows])
    batch = torch.from_numpy(inputs[:, channels]).to(device)
    latent = F.normalize(model.encode(batch), dim=1)
    finite = bool(torch.isfinite(latent).all())
    latent_np = latent.cpu().numpy()
    descriptor = surface_descriptors(inputs, device)
    adjacent_latent = cosine_distance(latent_np[:-1], latent_np[1:])
    adjacent_geometry = cosine_distance(descriptor[:-1], descriptor[1:])
    low = adjacent_geometry <= np.quantile(adjacent_geometry, 0.4)
    high = adjacent_geometry >= np.quantile(adjacent_geometry, 0.8)
    perturbations = {}
    for name, changed in make_perturbations(batch).items():
        altered = F.normalize(model.encode(changed), dim=1)
        distance = torch.clamp(1.0 - (latent * altered).sum(1), min=0).cpu().numpy()
        perturbations[name] = describe(distance)
    pair_left, pair_right = np.triu_indices(len(rows), k=1)
    pair_distance = cosine_distance(latent_np[pair_left], latent_np[pair_right])
    return {
        "samples": len(rows),
        "input_contract_shape": list(inputs.shape[1:]),
        "model_input_channels": channels,
        "forward": True,
        "finite": finite,
        "output_not_identical": bool(np.mean(np.std(latent_np, axis=0)) > 1e-6),
        "stamp_interval_s": describe(np.diff([row[0] for row in rows]) / 1e9),
        "adjacent_latent_distance": describe(adjacent_latent),
        "adjacent_input_geometry_distance": describe(adjacent_geometry),
        "stable_input_geometry_latent_distance": describe(adjacent_latent[low]),
        "changed_input_geometry_latent_distance": describe(adjacent_latent[high]),
        "input_geometry_latent_spearman": spearman(adjacent_latent, adjacent_geometry),
        "all_sample_pair_latent_distance": describe(pair_distance),
        "perturbations": perturbations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/structural_semantic_validation.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["experiment"]["output_root"]) / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    directories = (
        "baseline_metrics", "final_metrics", "stable_segment_audit", "structural_change_audit",
        "cross_location_similarity", "nuisance_factor_audit", "mtare_transfer_audit", "trajectory_previews", "config",
    )
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "config" / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if cfg["experiment"]["deterministic"]:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    channels = list(cfg["model"]["input_channels"])
    model = ForcedGlobalBottleneckNet(
        input_channels=len(channels), latent_dim=int(cfg["model"]["latent_dim"]), base_channels=int(cfg["model"]["base_channels"])
    ).to(device)
    checkpoint_path = Path(cfg["experiment"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()

    samples = load_split(Path(cfg["experiment"]["dataset_dir"]), "val")
    inputs = np.stack([sample["input"] for sample in samples])
    teachers = np.stack([sample["teacher"] for sample in samples])
    latent = encode(model, inputs, channels, device)
    input_descriptor = surface_descriptors(inputs, device)
    teacher_descriptor = surface_descriptors(teachers, device)
    cells = (inputs[:, 0] > 0.5).sum(axis=(1, 2)).astype(np.float64)
    density = np.asarray([float(sample["input"][1][sample["input"][0] > 0.5].mean()) if np.any(sample["input"][0] > 0.5) else 0.0 for sample in samples])

    audit = cfg["audit"]
    edges = adjacent_edges(samples, float(audit["trajectory_max_step_m"]), float(audit["trajectory_max_dt_s"]))
    edge_left = np.asarray([edge[0] for edge in edges]); edge_right = np.asarray([edge[1] for edge in edges])
    teacher_edge = cosine_distance(teacher_descriptor[edge_left], teacher_descriptor[edge_right])
    latent_edge = cosine_distance(latent[edge_left], latent[edge_right])
    coverage_edge = cosine_distance(input_descriptor[edge_left], input_descriptor[edge_right])
    cell_edge = np.abs(cells[edge_left] - cells[edge_right]) / np.maximum(np.maximum(cells[edge_left], cells[edge_right]), 1)
    stable_threshold = float(np.quantile(teacher_edge, float(audit["stable_quantile"])))
    change_threshold = float(np.quantile(teacher_edge, float(audit["change_quantile"])))
    stable = teacher_edge <= stable_threshold
    changed = teacher_edge >= change_threshold
    segments = stable_segments(edges, stable, samples, int(audit["minimum_stable_edges"]))
    stable_result = {
        "definition": f"consecutive same-robot edges with step <= {audit['trajectory_max_step_m']} m, dt <= {audit['trajectory_max_dt_s']} s, teacher geometry distance <= q{audit['stable_quantile']}",
        "valid_consecutive_edges": len(edges),
        "stable_edge_count": int(stable.sum()),
        "stable_segment_count": len(segments),
        "segments": segments,
        "latent_distance": describe(latent_edge[stable]),
        "teacher_geometry_distance": describe(teacher_edge[stable]),
        "input_coverage_shape_distance": describe(coverage_edge[stable]),
        "surface_cell_count_difference": describe(cell_edge[stable]),
    }
    change_test = distribution_test(latent_edge[stable], latent_edge[changed])
    change_result = {
        "reference": "multiscale dilated teacher surface layout; no free/unknown or manual category labels",
        "stable_threshold": stable_threshold,
        "change_threshold": change_threshold,
        "change_edge_count": int(changed.sum()),
        "latent_distance_test": change_test,
        "teacher_geometry_distance": {"stable": describe(teacher_edge[stable]), "changed": describe(teacher_edge[changed])},
        "input_coverage_shape_distance": {"stable": describe(coverage_edge[stable]), "changed": describe(coverage_edge[changed])},
        "surface_cell_count_difference": {"stable": describe(cell_edge[stable]), "changed": describe(cell_edge[changed])},
        "all_adjacent_teacher_latent_spearman": spearman(teacher_edge, latent_edge),
    }
    write_json(root / "stable_segment_audit" / "summary.json", stable_result)
    write_json(root / "structural_change_audit" / "summary.json", change_result)

    left, right, world_distance = all_cross_pairs(samples, float(audit["cross_location_min_distance_m"]))
    pair_latent = cosine_distance(latent[left], latent[right])
    pair_teacher = cosine_distance(teacher_descriptor[left], teacher_descriptor[right])
    pair_coverage = cosine_distance(input_descriptor[left], input_descriptor[right])
    pair_cells = np.abs(cells[left] - cells[right]) / np.maximum(np.maximum(cells[left], cells[right]), 1)
    pair_density = np.abs(density[left] - density[right])
    pair_robot = np.asarray([samples[l]["robot"] != samples[r]["robot"] for l, r in zip(left, right)], dtype=np.float64)
    pair_time = np.log1p(np.abs(np.asarray([samples[l]["stamp_ns"] - samples[r]["stamp_ns"] for l, r in zip(left, right)], dtype=np.float64)) / 1e9)
    similar_threshold = float(np.quantile(pair_teacher, float(audit["cross_similar_quantile"])))
    different_threshold = float(np.quantile(pair_teacher, float(audit["cross_different_quantile"])))
    similar = pair_teacher <= similar_threshold
    different = pair_teacher >= different_threshold
    cross_test = distribution_test(pair_latent[similar], pair_latent[different])
    retrieval = retrieval_audit(latent, teacher_descriptor, samples, float(audit["cross_location_min_distance_m"]), int(audit["retrieval_k"]), similar_threshold)
    cross_result = {
        "definition": f"world center distance >= {audit['cross_location_min_distance_m']} m",
        "pair_count": len(pair_latent),
        "similar_teacher_threshold": similar_threshold,
        "different_teacher_threshold": different_threshold,
        "similar_structure_pairs": describe(pair_latent[similar]),
        "different_structure_pairs": describe(pair_latent[different]),
        "distance_test": cross_test,
        "world_distance_m": {"similar": describe(world_distance[similar]), "different": describe(world_distance[different])},
        "retrieval": retrieval,
    }
    nuisance = {
        "pair_definition": f"all validation pairs at least {audit['cross_location_min_distance_m']} m apart",
        "pair_count": len(pair_latent),
        "spearman": {
            "teacher_geometry_difference": spearman(pair_latent, pair_teacher),
            "input_coverage_shape_difference": spearman(pair_latent, pair_coverage),
            "surface_cell_count_difference": spearman(pair_latent, pair_cells),
            "log_density_proxy_difference": spearman(pair_latent, pair_density),
            "robot_identity_difference": spearman(pair_latent, pair_robot),
            "log_time_interval": spearman(pair_latent, pair_time),
            "world_distance": spearman(pair_latent, world_distance),
        },
        "robot_conditioned": {
            "same_robot_latent_distance": describe(pair_latent[pair_robot == 0]),
            "different_robot_latent_distance": describe(pair_latent[pair_robot == 1]),
        },
    }
    write_json(root / "cross_location_similarity" / "summary.json", cross_result)
    write_json(root / "nuisance_factor_audit" / "summary.json", nuisance)

    perturb = perturbation_audit(model, inputs, channels, device, int(audit["perturbation_samples"]), latent_edge[changed])
    write_json(root / "final_metrics" / "perturbation.json", perturb)
    mtare = mtare_audit(model, Path(cfg["experiment"]["dataset_dir"]) / "mtare_samples", channels, device)
    write_json(root / "mtare_transfer_audit" / "summary.json", mtare)

    save_trajectory_preview(root / "trajectory_previews" / "validation_trajectory.png", samples, edges, teacher_edge, latent_edge, stable, changed)
    save_map_examples(root / "trajectory_previews" / "stable_and_change_examples.png", samples, edges, teacher_edge, stable, changed, int(audit["preview_examples"]))
    save_cross_preview(root / "trajectory_previews" / "cross_location_examples.png", samples, left, right, similar, different, pair_latent, pair_teacher, int(audit["preview_examples"]))

    correlations = nuisance["spearman"]
    nuisance_max = max(abs(correlations[name]) for name in ("input_coverage_shape_difference", "surface_cell_count_difference", "log_density_proxy_difference", "robot_identity_difference", "log_time_interval"))
    change_pass = change_test["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_change_to_stable_ratio"]) and change_test["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
    cross_pass = cross_test["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_cross_location_ratio"]) and cross_test["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
    geometry_pass = correlations["teacher_geometry_difference"] > nuisance_max
    perturb_pass = all(value["same_sample"]["median"] < value["structural_change_reference"]["median"] for value in perturb.values())
    mtare_pass = mtare["forward"] and mtare["finite"] and mtare["output_not_identical"] and mtare["changed_input_geometry_latent_distance"]["median"] > mtare["stable_input_geometry_latent_distance"]["median"]
    all_pass = change_pass and cross_pass and geometry_pass and perturb_pass and mtare_pass
    core_pass = change_pass and cross_pass and perturb_pass and mtare["forward"] and mtare["finite"]
    status = "STRUCTURAL_REPRESENTATION_READY" if all_pass else "STRUCTURAL_REPRESENTATION_READY_WITH_LIMITATIONS" if core_pass else "STRUCTURAL_REPRESENTATION_NOT_READY"
    baseline = {
        "method": "unchanged 128-dimensional ForcedGlobalBottleneckNet latent",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "new_training": False,
        "projection_head": False,
        "model_input_channels": ["surface_mask", "mean_height", "height_span"],
        "validation_samples": len(samples),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "structural_change": change_result,
        "cross_location": cross_result,
        "nuisance": nuisance,
        "perturbation": perturb,
        "mtare": mtare,
    }
    write_json(root / "baseline_metrics" / "summary.json", baseline)
    final = {
        "status": status,
        "representation": baseline["method"],
        "new_training": False,
        "acceptance": {
            "stable_vs_change": change_pass,
            "cross_location_similarity": cross_pass,
            "geometry_dominates_nuisance": geometry_pass,
            "observation_degradation_stability": perturb_pass,
            "mtare_transfer": mtare_pass,
        },
        "dominant_nuisance_abs_spearman": nuisance_max,
        "teacher_geometry_spearman": correlations["teacher_geometry_difference"],
        "environment": {
            "hostname": os.uname().nodename,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(root / "final_metrics" / "summary.json", final)
    print(json.dumps({"result_dir": str(root), "status": status, "acceptance": final["acceptance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
