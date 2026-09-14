from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.tools.run_structural_semantic_validation import (
    cosine_distance,
    describe,
    spearman,
    surface_descriptors,
    write_json,
)


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def load_world(dataset_root: Path, world: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted((dataset_root / world / "samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            samples.append(
                {
                    "path": str(path),
                    "input": data["input_surface"].astype(np.float32),
                    "teacher": data["teacher_surface"].astype(np.float32),
                    "pose": data["center_pose"].astype(np.float64),
                    "stamp_ns": int(data["stamp_ns"]),
                    "world": str(data["world"]),
                    "source_frame_index": int(data["source_frame_index"]),
                    "raw_point_count": int(data["raw_point_count"]),
                    "input_surface_cells": int(data["input_surface_cells"]),
                    "teacher_surface_cells": int(data["teacher_surface_cells"]),
                }
            )
    if not samples:
        raise RuntimeError(f"no samples found for {world}")
    return samples


def path_distance(samples: list[dict[str, Any]]) -> np.ndarray:
    distance = np.zeros(len(samples), dtype=np.float64)
    for index in range(1, len(samples)):
        distance[index] = distance[index - 1] + float(np.linalg.norm(samples[index]["pose"][:3] - samples[index - 1]["pose"][:3]))
    return distance


def causal_smooth(features: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.empty_like(features)
    smoothed[0] = features[0]
    for index in range(1, len(features)):
        smoothed[index] = alpha * features[index] + (1.0 - alpha) * smoothed[index - 1]
        smoothed[index] /= max(float(np.linalg.norm(smoothed[index])), 1e-8)
    return smoothed


@torch.no_grad()
def represent(
    base: ForcedGlobalBottleneckNet,
    projection: StructuralProjectionHead,
    inputs: np.ndarray,
    channels: list[int],
    device: torch.device,
) -> np.ndarray:
    output = []
    for start in range(0, len(inputs), 64):
        batch = torch.from_numpy(inputs[start : start + 64, channels]).to(device)
        output.append(projection(base.encode(batch)).cpu().numpy())
    return np.concatenate(output)


def teacher_change_reference(
    teacher_descriptor: np.ndarray,
    distances_m: np.ndarray,
    cfg: dict[str, Any],
    eval_start: int,
) -> dict[str, Any]:
    edge_distance = cosine_distance(teacher_descriptor[:-1], teacher_descriptor[1:])
    eval_edges = np.arange(eval_start, len(edge_distance))
    edge_eval = edge_distance[eval_edges]
    stable_threshold = float(np.quantile(edge_eval, float(cfg["change_reference"]["stable_quantile"])))
    change_threshold = float(np.quantile(edge_eval, float(cfg["change_reference"]["change_quantile"])))
    stable_edges = edge_distance <= stable_threshold
    candidate_edges = eval_edges[edge_eval >= change_threshold]
    order = candidate_edges[np.argsort(edge_distance[candidate_edges])[::-1]]
    selected: list[int] = []
    suppression = float(cfg["change_reference"]["nonmax_suppression_m"])
    for edge_index in order:
        event_distance = distances_m[edge_index + 1]
        if all(abs(event_distance - distances_m[other + 1]) >= suppression for other in selected):
            selected.append(int(edge_index))
    selected = sorted(selected, key=lambda edge: distances_m[edge + 1])
    events = [
        {
            "edge_index": int(edge),
            "sample_index": int(edge + 1),
            "path_m": float(distances_m[edge + 1]),
            "teacher_edge_distance": float(edge_distance[edge]),
        }
        for edge in selected
    ]
    stable_segments = []
    current_start: int | None = None
    min_length = float(cfg["change_reference"]["stable_segment_min_length_m"])
    for edge in eval_edges:
        if stable_edges[edge]:
            if current_start is None:
                current_start = int(edge)
        else:
            if current_start is not None and distances_m[edge] - distances_m[current_start] >= min_length:
                stable_segments.append({"start_sample": current_start, "end_sample": int(edge), "start_m": float(distances_m[current_start]), "end_m": float(distances_m[edge])})
            current_start = None
    if current_start is not None and distances_m[-1] - distances_m[current_start] >= min_length:
        stable_segments.append({"start_sample": current_start, "end_sample": len(distances_m) - 1, "start_m": float(distances_m[current_start]), "end_m": float(distances_m[-1])})
    return {
        "edge_distance": edge_distance,
        "stable_threshold": stable_threshold,
        "change_threshold": change_threshold,
        "stable_edges": stable_edges,
        "events": events,
        "stable_segments": stable_segments,
    }


def calibration_threshold(features: np.ndarray, calibration_count: int, quantile: float, alpha: float) -> dict[str, Any]:
    smoothed = causal_smooth(features, alpha)
    count = min(max(calibration_count, 3), len(features))
    edge_distance = cosine_distance(smoothed[: count - 1], smoothed[1:count])
    threshold = float(np.quantile(edge_distance, quantile))
    return {
        "threshold": threshold,
        "calibration_samples": int(count),
        "quantile": float(quantile),
        "edge_distance": describe(edge_distance),
    }


def make_node(sample_index: int, node_type: str, distances_m: np.ndarray, score: float) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "path_m": float(distances_m[sample_index]),
        "type": node_type,
        "trigger_score": float(score),
    }


def fixed_distance_nodes(distances_m: np.ndarray, eval_start: int, spacing_m: float) -> list[dict[str, Any]]:
    nodes = [make_node(eval_start, "start", distances_m, 0.0)]
    last = eval_start
    for index in range(eval_start + 1, len(distances_m)):
        if distances_m[index] - distances_m[last] >= spacing_m:
            nodes.append(make_node(index, "anchor", distances_m, distances_m[index] - distances_m[last]))
            last = index
    if nodes[-1]["sample_index"] != len(distances_m) - 1:
        nodes.append(make_node(len(distances_m) - 1, "end", distances_m, 0.0))
    return nodes


def triggered_nodes(
    features: np.ndarray,
    distances_m: np.ndarray,
    eval_start: int,
    threshold: float,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    online = cfg["online_node_generation"]
    smoothed = causal_smooth(features, float(online["smoothing_alpha"]))
    min_distance = float(online["min_node_distance_m"])
    max_anchor = float(online["max_anchor_distance_m"])
    cooldown = float(online["cooldown_distance_m"])
    persistence = int(online["trigger_persistence"])
    nodes = [make_node(eval_start, "start", distances_m, 0.0)]
    last_node = eval_start
    pending = 0
    pending_score = 0.0
    for index in range(eval_start + 1, len(distances_m)):
        since_last = distances_m[index] - distances_m[last_node]
        if since_last >= max_anchor:
            nodes.append(make_node(index, "anchor", distances_m, since_last))
            last_node = index
            pending = 0
            pending_score = 0.0
            continue
        if since_last < min_distance:
            pending = 0
            pending_score = 0.0
            continue
        score = float(cosine_distance(smoothed[index : index + 1], smoothed[last_node : last_node + 1])[0])
        if score >= threshold and since_last >= cooldown:
            pending += 1
            pending_score = max(pending_score, score)
        else:
            pending = 0
            pending_score = 0.0
        if pending >= persistence:
            nodes.append(make_node(index, "structure", distances_m, pending_score))
            last_node = index
            pending = 0
            pending_score = 0.0
    if nodes[-1]["sample_index"] != len(distances_m) - 1:
        nodes.append(make_node(len(distances_m) - 1, "end", distances_m, 0.0))
    return nodes


def edge_list(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "from_node": index,
            "to_node": index + 1,
            "from_sample": int(nodes[index]["sample_index"]),
            "to_sample": int(nodes[index + 1]["sample_index"]),
            "length_m": float(nodes[index + 1]["path_m"] - nodes[index]["path_m"]),
        }
        for index in range(len(nodes) - 1)
    ]


def nearest_distances(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(values) == 0 or len(targets) == 0:
        return np.asarray([], dtype=np.float64)
    return np.min(np.abs(values[:, None] - targets[None, :]), axis=1)


def evaluate_nodes(
    nodes: list[dict[str, Any]],
    distances_m: np.ndarray,
    change_ref: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    eval_cfg = cfg["evaluation"]
    node_positions = np.asarray([node["path_m"] for node in nodes], dtype=np.float64)
    structure_positions = np.asarray([node["path_m"] for node in nodes if node["type"] == "structure"], dtype=np.float64)
    anchor_positions = np.asarray([node["path_m"] for node in nodes if node["type"] == "anchor"], dtype=np.float64)
    event_positions = np.asarray([event["path_m"] for event in change_ref["events"]], dtype=np.float64)
    stable_segments = change_ref["stable_segments"]
    stable_length = float(sum(segment["end_m"] - segment["start_m"] for segment in stable_segments))
    stable_structure_count = 0
    stable_total_count = 0
    for segment in stable_segments:
        for node in nodes:
            if segment["start_m"] <= node["path_m"] <= segment["end_m"]:
                stable_total_count += 1
                if node["type"] == "structure":
                    stable_structure_count += 1
    recall_by_radius = {}
    structure_recall_by_radius = {}
    for radius in eval_cfg["change_match_radii_m"]:
        radius = float(radius)
        recall_by_radius[f"{radius:.1f}m"] = float(np.mean(nearest_distances(event_positions, node_positions) <= radius)) if len(event_positions) else 0.0
        structure_recall_by_radius[f"{radius:.1f}m"] = float(np.mean(nearest_distances(event_positions, structure_positions) <= radius)) if len(event_positions) and len(structure_positions) else 0.0
    precision_radius = float(eval_cfg["structure_precision_radius_m"])
    matched_structure = nearest_distances(structure_positions, event_positions) <= precision_radius if len(structure_positions) and len(event_positions) else np.zeros(len(structure_positions), dtype=bool)
    precision = float(np.mean(matched_structure)) if len(structure_positions) else 0.0
    recall = recall_by_radius[f"{precision_radius:.1f}m"]
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
    intervals = np.diff(node_positions)
    graph_edges = edge_list(nodes)
    max_anchor = float(cfg["online_node_generation"]["max_anchor_distance_m"])
    large_edge_limit = 1.25 * max_anchor
    return {
        "node_count": int(len(nodes)),
        "structure_node_count": int(len(structure_positions)),
        "anchor_node_count": int(len(anchor_positions)),
        "node_type_counts": {kind: int(sum(node["type"] == kind for node in nodes)) for kind in ("start", "structure", "anchor", "end")},
        "compression_ratio_samples_per_node": float(len(distances_m) / max(len(nodes), 1)),
        "node_spacing_m": describe(intervals),
        "stable_redundancy": {
            "stable_segment_count": int(len(stable_segments)),
            "stable_length_m": stable_length,
            "structure_nodes_in_stable_segments": int(stable_structure_count),
            "all_nodes_in_stable_segments": int(stable_total_count),
            "structure_nodes_per_100m_stable": float(100.0 * stable_structure_count / max(stable_length, 1e-8)),
            "consecutive_structure_triggers_under_3m": int(sum((nodes[i]["type"] == "structure" and nodes[i + 1]["type"] == "structure" and nodes[i + 1]["path_m"] - nodes[i]["path_m"] < 3.0) for i in range(len(nodes) - 1))),
        },
        "change_coverage": {
            "change_event_count": int(len(event_positions)),
            "recall_any_node_by_radius": recall_by_radius,
            "recall_structure_node_by_radius": structure_recall_by_radius,
            "nearest_any_node_error_m": describe(nearest_distances(event_positions, node_positions)),
            "nearest_structure_node_error_m": describe(nearest_distances(event_positions, structure_positions)),
        },
        "precision_recall": {
            "radius_m": precision_radius,
            "structure_precision": precision,
            "any_node_recall": recall,
            "f1_structure_precision_any_recall": f1,
            "anchor_nodes_reported_separately": True,
        },
        "graph_connectivity": {
            "single_connected_chain": len(nodes) >= 2 and all(edge["to_node"] == edge["from_node"] + 1 for edge in graph_edges),
            "edge_count": int(len(graph_edges)),
            "edge_length_m": describe(np.asarray([edge["length_m"] for edge in graph_edges], dtype=np.float64)),
            "edges_longer_than_max_anchor": int(sum(edge["length_m"] > max_anchor + 1e-6 for edge in graph_edges)),
            "edges_longer_than_1p25x_max_anchor": int(sum(edge["length_m"] > large_edge_limit + 1e-6 for edge in graph_edges)),
            "max_anchor_distance_m": max_anchor,
            "discrete_sampling_can_overshoot_anchor": True,
            "abnormal_cross_world_edges": 0,
        },
    }


def perturb_inputs(inputs: np.ndarray, name: str, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    changed = inputs.copy()
    mask = changed[:, 0] > 0.5
    height, width = mask.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    angle = np.arctan2(cy - yy, xx - cx)
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * 0.2
    if name == "random_drop":
        keep = rng.random(mask.shape) < float(cfg["robustness"]["random_drop_keep_probability"])
        drop = mask & ~keep
    elif name == "nonuniform_drop":
        keep_prob = float(cfg["robustness"]["nonuniform_keep_probability"])
        gradient = np.clip(0.25 + 0.75 * (xx / max(width - 1, 1)), 0.0, 1.0)
        keep = rng.random(mask.shape) < (keep_prob * gradient[None])
        drop = mask & ~keep
    elif name == "sector_occlusion":
        sector_width = np.deg2rad(float(cfg["robustness"]["sector_width_deg"]))
        centers = rng.uniform(-np.pi, np.pi, size=(len(inputs), 1, 1))
        wrapped = np.abs(np.angle(np.exp(1j * (angle[None] - centers))))
        drop = mask & (wrapped < sector_width / 2.0)
    elif name == "shorten_range":
        drop = mask & (radius[None] > float(cfg["robustness"]["shorten_range_m"]))
    else:
        raise ValueError(name)
    for channel in range(changed.shape[1]):
        changed[:, channel][drop] = 0.0
    return changed


def robustness_audit(
    method: str,
    features_original: np.ndarray,
    inputs: np.ndarray,
    distances_m: np.ndarray,
    eval_start: int,
    threshold: float,
    cfg: dict[str, Any],
    device: torch.device,
    base: ForcedGlobalBottleneckNet | None,
    projection: StructuralProjectionHead | None,
    channels: list[int],
    input_descriptor: bool,
) -> dict[str, Any]:
    original_nodes = triggered_nodes(features_original, distances_m, eval_start, threshold, cfg)
    original_positions = np.asarray([node["path_m"] for node in original_nodes], dtype=np.float64)
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + len(method))
    result = {}
    for name in cfg["robustness"]["perturbation_names"]:
        changed_inputs = perturb_inputs(inputs, name, cfg, rng)
        if input_descriptor:
            features = surface_descriptors(changed_inputs, device)
        else:
            assert base is not None and projection is not None
            features = represent(base, projection, changed_inputs, channels, device)
        nodes = triggered_nodes(features, distances_m, eval_start, threshold, cfg)
        positions = np.asarray([node["path_m"] for node in nodes], dtype=np.float64)
        nearest = nearest_distances(original_positions, positions)
        tolerance = float(cfg["online_node_generation"]["min_node_distance_m"])
        result[name] = {
            "original_node_count": int(len(original_nodes)),
            "perturbed_node_count": int(len(nodes)),
            "node_count_delta": int(len(nodes) - len(original_nodes)),
            "original_to_perturbed_nearest_shift_m": describe(nearest),
            "node_match_rate_within_min_distance": float(np.mean(nearest <= tolerance)) if len(nearest) else 0.0,
            "duplicate_triggers_under_3m": int(sum((positions[i + 1] - positions[i]) < 3.0 for i in range(len(positions) - 1))),
        }
    return result


def save_preview(
    path: Path,
    world: str,
    samples: list[dict[str, Any]],
    distances_m: np.ndarray,
    change_ref: dict[str, Any],
    method_nodes: dict[str, list[dict[str, Any]]],
) -> None:
    xy = np.stack([sample["pose"][:2] for sample in samples])
    fig, axes = plt.subplots(len(method_nodes), 1, figsize=(10, 4 * len(method_nodes)), squeeze=False)
    event_samples = [event["sample_index"] for event in change_ref["events"]]
    for row, (method, nodes) in enumerate(method_nodes.items()):
        ax = axes[row, 0]
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=1.0, label="trajectory")
        if event_samples:
            ax.scatter(xy[event_samples, 0], xy[event_samples, 1], marker="x", s=60, color="crimson", label="teacher change")
        for kind, color, marker in (("structure", "dodgerblue", "o"), ("anchor", "darkorange", "^"), ("start", "black", "s"), ("end", "black", "D")):
            indices = [node["sample_index"] for node in nodes if node["type"] == kind]
            if indices:
                ax.scatter(xy[indices, 0], xy[indices, 1], s=28, color=color, marker=marker, label=kind)
        ax.set_title(f"{world} {method}")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_world(
    world: str,
    samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    root: Path,
    base: ForcedGlobalBottleneckNet,
    projection: StructuralProjectionHead,
    device: torch.device,
) -> dict[str, Any]:
    channels = list(cfg["model"]["input_channels"])
    inputs = np.stack([sample["input"] for sample in samples])
    teachers = np.stack([sample["teacher"] for sample in samples])
    distances_m = path_distance(samples)
    eval_start = int(cfg["calibration"]["samples_per_world"])
    latent = represent(base, projection, inputs, channels, device)
    input_desc = surface_descriptors(inputs, device)
    teacher_desc = surface_descriptors(teachers, device)
    change_ref = teacher_change_reference(teacher_desc, distances_m, cfg, eval_start)
    latent_threshold = calibration_threshold(latent, eval_start, float(cfg["calibration"]["latent_trigger_quantile"]), float(cfg["online_node_generation"]["smoothing_alpha"]))
    input_threshold = calibration_threshold(input_desc, eval_start, float(cfg["calibration"]["input_trigger_quantile"]), float(cfg["online_node_generation"]["smoothing_alpha"]))
    methods: dict[str, list[dict[str, Any]]] = {
        "fixed_distance": fixed_distance_nodes(distances_m, eval_start, float(cfg["fixed_distance_baseline"]["primary_spacing_m"])),
        "input_geometry": triggered_nodes(input_desc, distances_m, eval_start, input_threshold["threshold"], cfg),
        "learned_representation": triggered_nodes(latent, distances_m, eval_start, latent_threshold["threshold"], cfg),
    }
    sensitivity = {
        f"fixed_{float(spacing):.1f}m": evaluate_nodes(fixed_distance_nodes(distances_m, eval_start, float(spacing)), distances_m, change_ref, cfg)
        for spacing in cfg["fixed_distance_baseline"]["sensitivity_spacing_m"]
    }
    metrics = {method: evaluate_nodes(nodes, distances_m, change_ref, cfg) for method, nodes in methods.items()}
    robustness = {
        "input_geometry": robustness_audit("input_geometry", input_desc, inputs, distances_m, eval_start, input_threshold["threshold"], cfg, device, base, projection, channels, True),
        "learned_representation": robustness_audit("learned_representation", latent, inputs, distances_m, eval_start, latent_threshold["threshold"], cfg, device, base, projection, channels, False),
    }
    edge_distance = change_ref["edge_distance"]
    eval_edges = np.arange(eval_start, len(edge_distance))
    latent_edge = cosine_distance(latent[:-1], latent[1:])
    input_edge = cosine_distance(input_desc[:-1], input_desc[1:])
    factor_audit = {
        "teacher_vs_latent_edge_spearman": spearman(edge_distance[eval_edges], latent_edge[eval_edges]),
        "teacher_vs_input_edge_spearman": spearman(edge_distance[eval_edges], input_edge[eval_edges]),
        "surface_cell_edge_spearman_latent": spearman(latent_edge[eval_edges], np.abs(np.diff([sample["input_surface_cells"] for sample in samples]))[eval_edges]),
        "raw_point_edge_spearman_latent": spearman(latent_edge[eval_edges], np.abs(np.diff([sample["raw_point_count"] for sample in samples]))[eval_edges]),
    }
    thresholds = {
        "world": world,
        "eval_start_sample": eval_start,
        "calibration_policy": "first fixed samples excluded from final metrics",
        "latent": latent_threshold,
        "input_geometry": input_threshold,
        "teacher_change_reference": {
            "stable_threshold": change_ref["stable_threshold"],
            "change_threshold": change_ref["change_threshold"],
            "event_count": len(change_ref["events"]),
            "stable_segment_count": len(change_ref["stable_segments"]),
        },
    }
    world_dir = root / "node_lists" / world
    edge_dir = root / "edge_lists" / world
    for method, nodes in methods.items():
        write_json(world_dir / f"{method}.json", nodes)
        write_json(edge_dir / f"{method}.json", edge_list(nodes))
    write_json(root / "change_reference" / f"{world}.json", {k: v for k, v in change_ref.items() if k not in ("edge_distance", "stable_edges")})
    np.savez_compressed(root / "change_reference" / f"{world}_arrays.npz", edge_distance=change_ref["edge_distance"], stable_edges=change_ref["stable_edges"])
    write_json(root / "threshold_provenance" / f"{world}.json", thresholds)
    write_json(root / "world_metrics" / f"{world}.json", {"world": world, "sample_count": len(samples), "evaluated_samples": len(samples) - eval_start, "path_length_m": float(distances_m[-1] - distances_m[eval_start]), "methods": metrics, "factor_audit": factor_audit})
    write_json(root / "baseline_comparison" / f"{world}.json", {"primary": metrics, "fixed_distance_sensitivity": sensitivity})
    write_json(root / "robustness_metrics" / f"{world}.json", robustness)
    save_preview(root / "trajectory_previews" / f"{world}_nodes.png", world, samples, distances_m, change_ref, methods)
    return {
        "world": world,
        "sample_count": len(samples),
        "evaluated_samples": len(samples) - eval_start,
        "path_length_m": float(distances_m[-1] - distances_m[eval_start]),
        "metrics": metrics,
        "robustness": robustness,
        "thresholds": thresholds,
        "factor_audit": factor_audit,
    }


def status_from_results(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    checks = {}
    for world, result in results.items():
        learned = result["metrics"]["learned_representation"]
        fixed = result["metrics"]["fixed_distance"]
        geom = result["metrics"]["input_geometry"]
        learned_recall = learned["change_coverage"]["recall_any_node_by_radius"]["2.0m"]
        fixed_recall = fixed["change_coverage"]["recall_any_node_by_radius"]["2.0m"]
        geom_recall = geom["change_coverage"]["recall_any_node_by_radius"]["2.0m"]
        learned_redundancy = learned["stable_redundancy"]["structure_nodes_per_100m_stable"]
        geom_redundancy = geom["stable_redundancy"]["structure_nodes_per_100m_stable"]
        learned_error = learned["change_coverage"]["nearest_any_node_error_m"].get("median", float("inf"))
        fixed_error = fixed["change_coverage"]["nearest_any_node_error_m"].get("median", float("inf"))
        graph_ok = learned["graph_connectivity"]["single_connected_chain"] and learned["graph_connectivity"]["edges_longer_than_1p25x_max_anchor"] == 0
        robust_values = result["robustness"]["learned_representation"].values()
        robust_ok = all(value["node_match_rate_within_min_distance"] >= 0.65 for value in robust_values)
        learned_better_than_input = (
            learned_recall + 1e-8 >= geom_recall
            and learned["precision_recall"]["structure_precision"] + 1e-8 >= geom["precision_recall"]["structure_precision"]
            and learned["node_count"] <= geom["node_count"] + 5
        )
        checks[world] = {
            "recall_not_worse_than_fixed": learned_recall + 1e-8 >= fixed_recall,
            "recall_not_worse_than_input_geometry": learned_recall + 1e-8 >= geom_recall,
            "stable_redundancy_below_input_geometry": learned_redundancy <= geom_redundancy + 1e-8,
            "node_error_below_fixed": learned_error <= fixed_error + 1e-8,
            "learned_improves_structural_metrics_over_input_geometry": learned_better_than_input,
            "robustness": robust_ok,
            "graph_connected": graph_ok,
        }
    pass_worlds = sum(all(values.values()) for values in checks.values())
    if pass_worlds == len(checks):
        return "TOPOLOGY_NODE_READY", checks
    partial_worlds = sum(
        values["graph_connected"]
        and values["recall_not_worse_than_fixed"]
        and values["recall_not_worse_than_input_geometry"]
        and values["learned_improves_structural_metrics_over_input_geometry"]
        for values in checks.values()
    )
    if pass_worlds >= 1 or partial_worlds >= 1:
        return "TOPOLOGY_NODE_READY_WITH_LIMITATIONS", checks
    return "TOPOLOGY_NODE_NOT_READY", checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/offline_topology_node_validation.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["experiment"]["output_root"]) / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    for name in ("config", "threshold_provenance", "world_metrics", "baseline_comparison", "change_reference", "robustness_metrics", "trajectory_previews", "node_lists", "edge_lists"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, root / "config" / "config.yaml")
    source_paths = [
        Path(__file__),
        Path("learning/structural_learning/bottleneck_model.py"),
        Path("learning/structural_learning/semantic_projection.py"),
        Path("learning/structural_learning/tools/run_structural_semantic_validation.py"),
    ]
    provenance = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "config" / "provenance.json", provenance)
    (root / "config" / "package_list.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"])["stdout"], encoding="utf-8")
    random.seed(int(cfg["experiment"]["seed"]))
    np.random.seed(int(cfg["experiment"]["seed"]))
    torch.manual_seed(int(cfg["experiment"]["seed"]))
    torch.cuda.manual_seed_all(int(cfg["experiment"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = list(cfg["model"]["input_channels"])
    base = ForcedGlobalBottleneckNet(input_channels=len(channels), latent_dim=int(cfg["model"]["encoder_latent_dim"]), base_channels=int(cfg["model"]["encoder_base_channels"])).to(device)
    encoder_path = Path(cfg["experiment"]["encoder_checkpoint"])
    encoder_checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    base.load_state_dict(encoder_checkpoint["model"])
    base.eval()
    projection = StructuralProjectionHead(int(cfg["model"]["encoder_latent_dim"]), int(cfg["model"]["projection_hidden_dim"]), int(cfg["model"]["representation_dim"])).to(device)
    projection_path = Path(cfg["experiment"]["projection_checkpoint"])
    projection_checkpoint = torch.load(projection_path, map_location=device, weights_only=False)
    projection.load_state_dict(projection_checkpoint["projection"])
    projection.eval()
    for parameter in list(base.parameters()) + list(projection.parameters()):
        parameter.requires_grad_(False)
    model_audit = {
        "frozen": True,
        "supervised_mtare_training": False,
        "encoder_checkpoint": str(encoder_path),
        "projection_checkpoint": str(projection_path),
        "encoder_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        "projection_sha256": hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(root / "config" / "model_audit.json", model_audit)
    dataset_root = Path(cfg["experiment"]["dataset_root"])
    input_contract = json.loads((dataset_root / "input_contract.json").read_text())
    write_json(root / "config" / "input_contract_audit.json", {"contract": input_contract, "model_channel_indices": channels, "model_channel_names": [input_contract["channels"][index]["name"] for index in channels], "same_builder": True})
    results = {}
    for world in cfg["worlds"]:
        samples = load_world(dataset_root, world)
        results[world] = evaluate_world(world, samples, cfg, root, base, projection, device)
    status, checks = status_from_results(results)
    summary = {
        "status": status,
        "acceptance_checks": checks,
        "worlds": {world: {"sample_count": result["sample_count"], "evaluated_samples": result["evaluated_samples"], "path_length_m": result["path_length_m"], "learned": result["metrics"]["learned_representation"], "fixed": result["metrics"]["fixed_distance"], "input_geometry": result["metrics"]["input_geometry"]} for world, result in results.items()},
        "model_audit": model_audit,
    }
    write_json(root / "world_metrics" / "summary.json", summary)
    print(json.dumps({"result_dir": str(root), "status": status, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
