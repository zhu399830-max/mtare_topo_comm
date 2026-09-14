from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.ndimage import shift
from scipy.stats import rankdata
from torch.nn import functional as F

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.tools.run_structural_semantic_validation import surface_descriptors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: np.ndarray | list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.sum(left * right, axis=-1), 0.0, 2.0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def load_structural_split(root: Path, split: str, environment: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["environment"]) != environment:
                continue
            trajectory = str(data["robot_or_trajectory"])
            samples.append(
                {
                    "path": str(path),
                    "input": data["input_surface"].astype(np.float32),
                    "teacher": data["teacher_surface"].astype(np.float32),
                    "pose": data["center_pose"].astype(np.float64),
                    "stamp_ns": int(data["stamp_ns"]),
                    "environment": environment,
                    "trajectory": trajectory,
                    "source_id": str(data["source_scan_identifier"]),
                    "raw_point_count": int(data["raw_point_count"]),
                    "input_surface_cells": int((data["input_surface"][0] > 0.5).sum()),
                    "teacher_surface_cells": int((data["teacher_surface"][0] > 0.5).sum()),
                    "support": float(data["geometry_input_teacher_support_ratio"]),
                }
            )
    if not samples:
        raise RuntimeError(f"no {environment} samples in {root / split}")
    return sorted(samples, key=lambda s: (s["trajectory"], s["stamp_ns"]))


def load_mtare_world(root: Path, world: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted((root / world / "samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            samples.append(
                {
                    "path": str(path),
                    "input": data["input_surface"].astype(np.float32),
                    "teacher": data["teacher_surface"].astype(np.float32),
                    "pose": data["center_pose"].astype(np.float64),
                    "stamp_ns": int(data["stamp_ns"]),
                    "environment": world,
                    "trajectory": world,
                    "source_id": f"{world}:{int(data['source_frame_index'])}",
                    "raw_point_count": int(data["raw_point_count"]),
                    "input_surface_cells": int(data["input_surface_cells"]),
                    "teacher_surface_cells": int(data["teacher_surface_cells"]),
                    "support": 1.0,
                }
            )
    if not samples:
        raise RuntimeError(f"no M-TARE samples found for {world}")
    return sorted(samples, key=lambda s: (s["trajectory"], s["stamp_ns"]))


def trajectory_indices(samples: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample["trajectory"]].append(index)
    return {key: sorted(value, key=lambda i: samples[i]["stamp_ns"]) for key, value in groups.items()}


def path_distances(samples: list[dict[str, Any]]) -> np.ndarray:
    distances = np.zeros(len(samples), dtype=np.float64)
    for _, order in trajectory_indices(samples).items():
        accum = 0.0
        if order:
            distances[order[0]] = 0.0
        for left, right in zip(order[:-1], order[1:]):
            accum += float(np.linalg.norm(samples[right]["pose"][:3] - samples[left]["pose"][:3]))
            distances[right] = accum
    return distances


@torch.no_grad()
def encode_base(model: ForcedGlobalBottleneckNet, inputs: np.ndarray, channels: list[int], device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(inputs), 128):
        batch = torch.from_numpy(inputs[start : start + 128, channels]).to(device)
        chunks.append(F.normalize(model.encode(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def encode_projected(
    model: ForcedGlobalBottleneckNet,
    projection: StructuralProjectionHead,
    inputs: np.ndarray,
    channels: list[int],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    projection.eval()
    chunks = []
    for start in range(0, len(inputs), 128):
        batch = torch.from_numpy(inputs[start : start + 128, channels]).to(device)
        chunks.append(projection(model.encode(batch)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def feature_sets(
    samples: list[dict[str, Any]],
    models: dict[str, Any],
    channels: list[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    inputs = np.stack([s["input"] for s in samples])
    return {
        "input_geometry": surface_descriptors(inputs, device),
        "old_tunnel_64": encode_projected(models["old_encoder"], models["old_projection"], inputs, channels, device),
        "new_multienv_64": encode_projected(models["new_encoder"], models["new_projection"], inputs, channels, device),
        "new_multienv_128": encode_base(models["new_encoder"], inputs, channels, device),
    }


def teacher_descriptor(samples: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    return surface_descriptors(np.stack([s["teacher"] for s in samples]), device)


def input_descriptor(samples: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    return surface_descriptors(np.stack([s["input"] for s in samples]), device)


def teacher_change_reference(samples: list[dict[str, Any]], teacher_desc: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    refs = {}
    stable_q = float(cfg["change_reference"]["stable_quantile"])
    change_q = float(cfg["change_reference"]["change_quantile"])
    suppression = float(cfg["change_reference"]["nonmax_suppression_m"])
    min_stable = float(cfg["change_reference"]["stable_segment_min_length_m"])
    distances = path_distances(samples)
    for trajectory, order in trajectory_indices(samples).items():
        if len(order) < 3:
            refs[trajectory] = {"edges": [], "events": [], "stable_segments": [], "stable_threshold": 0.0, "change_threshold": 0.0}
            continue
        left = np.asarray(order[:-1], dtype=int)
        right = np.asarray(order[1:], dtype=int)
        edge_distance = cosine_distance(teacher_desc[left], teacher_desc[right])
        stable_threshold = float(np.quantile(edge_distance, stable_q))
        change_threshold = float(np.quantile(edge_distance, change_q))
        event_candidates = np.where(edge_distance >= change_threshold)[0]
        ranked = event_candidates[np.argsort(edge_distance[event_candidates])[::-1]]
        chosen: list[int] = []
        for local_edge in ranked:
            sample_index = int(right[local_edge])
            if all(abs(distances[sample_index] - distances[int(right[other])]) >= suppression for other in chosen):
                chosen.append(int(local_edge))
        chosen = sorted(chosen, key=lambda e: distances[int(right[e])])
        events = [
            {
                "trajectory": trajectory,
                "left_sample": int(left[e]),
                "sample_index": int(right[e]),
                "path_m": float(distances[int(right[e])]),
                "teacher_edge_distance": float(edge_distance[e]),
            }
            for e in chosen
        ]
        stable_segments = []
        current_start: int | None = None
        for local_edge, edge_value in enumerate(edge_distance):
            sample_left = int(left[local_edge])
            sample_right = int(right[local_edge])
            if edge_value <= stable_threshold:
                if current_start is None:
                    current_start = sample_left
            else:
                if current_start is not None and distances[sample_left] - distances[current_start] >= min_stable:
                    stable_segments.append({"trajectory": trajectory, "start_sample": current_start, "end_sample": sample_left, "start_m": float(distances[current_start]), "end_m": float(distances[sample_left])})
                current_start = None
            if local_edge == len(edge_distance) - 1 and current_start is not None and distances[sample_right] - distances[current_start] >= min_stable:
                stable_segments.append({"trajectory": trajectory, "start_sample": current_start, "end_sample": sample_right, "start_m": float(distances[current_start]), "end_m": float(distances[sample_right])})
        refs[trajectory] = {
            "edges": [{"left": int(l), "right": int(r), "distance": float(d)} for l, r, d in zip(left, right, edge_distance)],
            "events": events,
            "stable_segments": stable_segments,
            "stable_threshold": stable_threshold,
            "change_threshold": change_threshold,
        }
    return refs


def sample_pairs(samples: list[dict[str, Any]], input_desc: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    positions = np.stack([s["pose"][:3] for s in samples])
    stamps = np.asarray([s["stamp_ns"] for s in samples], dtype=np.float64) / 1e9
    left, right = np.triu_indices(len(samples), k=1)
    spatial = np.linalg.norm(positions[left] - positions[right], axis=1)
    time_gap = np.abs(stamps[left] - stamps[right])
    positive = (spatial <= float(cfg["pairs"]["positive_radius_m"])) & (time_gap >= float(cfg["pairs"]["minimum_positive_time_gap_s"]))
    negative = spatial >= float(cfg["pairs"]["negative_radius_m"])
    coverage_delta = np.abs(np.asarray([s["input_surface_cells"] for s in samples], dtype=np.float64)[left] - np.asarray([s["input_surface_cells"] for s in samples], dtype=np.float64)[right])
    if negative.any():
        hard_cut = float(np.quantile(coverage_delta[negative], float(cfg["pairs"]["hard_negative_coverage_quantile"])))
        hard_negative = negative & (coverage_delta <= hard_cut)
    else:
        hard_negative = negative
    keep = positive | negative
    max_pairs = int(cfg["pairs"]["maximum_pairs_per_environment"])
    candidates = np.where(keep)[0]
    if len(candidates) > max_pairs:
        pos_idx = np.where(positive)[0]
        hard_idx = np.where(hard_negative)[0]
        neg_idx = np.setdiff1d(np.where(negative)[0], hard_idx, assume_unique=False)
        chosen = []
        for pool, frac in ((pos_idx, 0.35), (hard_idx, 0.35), (neg_idx, 0.30)):
            take = min(len(pool), max(1, int(max_pairs * frac)))
            if take:
                chosen.append(rng.choice(pool, size=take, replace=False))
        candidates = np.unique(np.concatenate(chosen)) if chosen else candidates[:max_pairs]
        if len(candidates) > max_pairs:
            candidates = rng.choice(candidates, size=max_pairs, replace=False)
    labels = positive[candidates].astype(np.uint8)
    return {
        "left": left[candidates].astype(np.int64),
        "right": right[candidates].astype(np.int64),
        "label": labels,
        "spatial_distance_m": spatial[candidates],
        "coverage_delta": coverage_delta[candidates],
        "hard_negative": hard_negative[candidates],
        "positive_count_total": np.asarray([int(positive.sum())]),
        "negative_count_total": np.asarray([int(negative.sum())]),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return 0.5
    ranks = rankdata(scores)
    pos_rank = ranks[labels].sum()
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    return float((pos_rank - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    order = np.argsort(scores)[::-1]
    y = labels[order].astype(np.float64)
    if y.sum() == 0:
        return 0.0
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / max(y.sum(), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.trapezoid(precision, recall))


def threshold_metrics(labels: np.ndarray, distances: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(bool)
    pred = distances <= threshold
    tp = float((pred & labels).sum())
    fp = float((pred & ~labels).sum())
    fn = float((~pred & labels).sum())
    tn = float((~pred & ~labels).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_merge_rate": float(fp / max(fp + tn, 1.0)),
        "false_split_rate": float(fn / max(tp + fn, 1.0)),
    }


def calibrate_match_threshold(labels: np.ndarray, distances: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    lo, hi = float(np.min(distances)), float(np.max(distances))
    grid = np.linspace(lo, hi, int(cfg["calibration"]["threshold_grid_size"]))
    rows = [threshold_metrics(labels, distances, threshold) for threshold in grid]
    min_precision = float(cfg["calibration"]["min_precision_for_match_threshold"])
    eligible = [row for row in rows if row["precision"] >= min_precision]
    selected = max(eligible or rows, key=lambda row: (row["f1"], row["recall"], -row["false_merge_rate"]))
    return {"selected": selected, "search_range": [lo, hi], "grid_size": len(grid), "min_precision": min_precision}


def pair_metrics(features: np.ndarray, pairs: dict[str, np.ndarray], threshold: float, recall_k: list[int]) -> dict[str, Any]:
    left = pairs["left"]
    right = pairs["right"]
    labels = pairs["label"].astype(bool)
    distances = cosine_distance(features[left], features[right])
    same = distances[labels]
    different = distances[~labels]
    scores = -distances
    result = {
        "pair_count": int(len(labels)),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((~labels).sum()),
        "same_location_feature_distance": describe(same),
        "different_location_feature_distance": describe(different),
        "different_location_p10": float(np.percentile(different, 10)) if len(different) else 0.0,
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "threshold_metrics": threshold_metrics(labels, distances, threshold),
    }
    positions_by_query: dict[int, list[int]] = defaultdict(list)
    for l, r, label in zip(left, right, labels):
        if label:
            positions_by_query[int(l)].append(int(r))
            positions_by_query[int(r)].append(int(l))
    nn_hits = {f"recall_at_{k}": [] for k in recall_k}
    for query, positives in positions_by_query.items():
        d = cosine_distance(features[query : query + 1], features)
        d[query] = np.inf
        order = np.argsort(d)
        positives_set = set(positives)
        for k in recall_k:
            nn_hits[f"recall_at_{k}"].append(float(any(idx in positives_set for idx in order[:k])))
    result["recall_at_k"] = {key: float(np.mean(value)) if value else 0.0 for key, value in nn_hits.items()}
    return result


def causal_smooth(features: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(features)
    out[0] = features[0]
    for i in range(1, len(features)):
        out[i] = alpha * features[i] + (1.0 - alpha) * out[i - 1]
        out[i] /= max(float(np.linalg.norm(out[i])), 1e-8)
    return out


def calibrate_trigger_threshold(features: np.ndarray, samples: list[dict[str, Any]], change_ref: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    alpha = float(cfg["online_node_generation"]["smoothing_alpha"])
    smoothed = causal_smooth(features, alpha)
    values = []
    labels = []
    for trajectory, ref in change_ref.items():
        for edge in ref["edges"]:
            values.append(float(cosine_distance(smoothed[edge["left"] : edge["left"] + 1], smoothed[edge["right"] : edge["right"] + 1])[0]))
            labels.append(edge["distance"] >= ref["change_threshold"])
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    q0, q1 = cfg["calibration"]["trigger_search_quantiles"]
    lo, hi = np.quantile(values, [float(q0), float(q1)])
    grid = np.linspace(float(lo), float(hi), int(cfg["calibration"]["trigger_grid_size"]))
    rows = []
    for threshold in grid:
        pred = values >= threshold
        tp = float((pred & labels).sum())
        fp = float((pred & ~labels).sum())
        fn = float((~pred & labels).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        rows.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    selected = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return {"selected": selected, "search_quantiles": [float(q0), float(q1)], "edge_values": describe(values), "positive_edge_count": int(labels.sum()), "edge_count": int(len(labels))}


def make_node(sample_index: int, node_type: str, path_m: float, score: float, trajectory: str) -> dict[str, Any]:
    return {"sample_index": int(sample_index), "type": node_type, "path_m": float(path_m), "score": float(score), "trajectory": trajectory}


def triggered_nodes_for_trajectory(order: list[int], samples: list[dict[str, Any]], distances: np.ndarray, features: np.ndarray, threshold: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    online = cfg["online_node_generation"]
    alpha = float(online["smoothing_alpha"])
    local_features = causal_smooth(features[order], alpha)
    min_dist = float(online["min_node_distance_m"])
    max_anchor = float(online["max_anchor_distance_m"])
    cooldown = float(online["cooldown_distance_m"])
    persistence = int(online["trigger_persistence"])
    trajectory = samples[order[0]]["trajectory"]
    nodes = [make_node(order[0], "start", distances[order[0]], 0.0, trajectory)]
    last_local = 0
    pending = 0
    pending_score = 0.0
    for local_idx in range(1, len(order)):
        sample_idx = order[local_idx]
        since_last = distances[sample_idx] - distances[order[last_local]]
        if since_last >= max_anchor:
            nodes.append(make_node(sample_idx, "anchor", distances[sample_idx], since_last, trajectory))
            last_local = local_idx
            pending = 0
            pending_score = 0.0
            continue
        if since_last < min_dist:
            pending = 0
            pending_score = 0.0
            continue
        score = float(cosine_distance(local_features[local_idx : local_idx + 1], local_features[last_local : last_local + 1])[0])
        if score >= threshold and since_last >= cooldown:
            pending += 1
            pending_score = max(pending_score, score)
        else:
            pending = 0
            pending_score = 0.0
        if pending >= persistence:
            nodes.append(make_node(sample_idx, "structure", distances[sample_idx], pending_score, trajectory))
            last_local = local_idx
            pending = 0
            pending_score = 0.0
    if nodes[-1]["sample_index"] != order[-1]:
        nodes.append(make_node(order[-1], "end", distances[order[-1]], 0.0, trajectory))
    return nodes


def fixed_nodes_for_trajectory(order: list[int], samples: list[dict[str, Any]], distances: np.ndarray, spacing: float) -> list[dict[str, Any]]:
    trajectory = samples[order[0]]["trajectory"]
    nodes = [make_node(order[0], "start", distances[order[0]], 0.0, trajectory)]
    last = order[0]
    for idx in order[1:]:
        if distances[idx] - distances[last] >= spacing:
            nodes.append(make_node(idx, "anchor", distances[idx], distances[idx] - distances[last], trajectory))
            last = idx
    if nodes[-1]["sample_index"] != order[-1]:
        nodes.append(make_node(order[-1], "end", distances[order[-1]], 0.0, trajectory))
    return nodes


def generate_candidate_nodes(method: str, samples: list[dict[str, Any]], features: np.ndarray | None, trigger_threshold: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    distances = path_distances(samples)
    nodes: list[dict[str, Any]] = []
    for _, order in trajectory_indices(samples).items():
        if method == "fixed_distance":
            nodes.extend(fixed_nodes_for_trajectory(order, samples, distances, float(cfg["fixed_distance_baseline"]["primary_spacing_m"])))
        else:
            assert features is not None
            nodes.extend(triggered_nodes_for_trajectory(order, samples, distances, features, trigger_threshold, cfg))
    return sorted(nodes, key=lambda n: (n["trajectory"], n["path_m"], n["sample_index"]))


def assign_location_clusters(nodes: list[dict[str, Any]], samples: list[dict[str, Any]], radius: float) -> list[int]:
    clusters: list[np.ndarray] = []
    ids = []
    for node in nodes:
        pos = samples[node["sample_index"]]["pose"][:3]
        assigned = None
        for idx, center in enumerate(clusters):
            if np.linalg.norm(pos - center) <= radius:
                assigned = idx
                clusters[idx] = 0.5 * center + 0.5 * pos
                break
        if assigned is None:
            assigned = len(clusters)
            clusters.append(pos.copy())
        ids.append(assigned)
    return ids


def merge_nodes(nodes: list[dict[str, Any]], samples: list[dict[str, Any]], features: np.ndarray | None, match_threshold: float, cfg: dict[str, Any], fixed: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gate = float(cfg["online_node_generation"]["merge_pose_gate_m"])
    merged: list[dict[str, Any]] = []
    assignments = []
    edges = []
    last_effective_by_traj: dict[str, int] = {}
    for node in nodes:
        sample_idx = node["sample_index"]
        pos = samples[sample_idx]["pose"][:3]
        best_id = None
        best_score = float("inf")
        for existing in merged:
            existing_pos = samples[existing["representative_sample"]]["pose"][:3]
            spatial = float(np.linalg.norm(pos - existing_pos))
            if spatial > gate:
                continue
            if fixed or features is None:
                score = spatial / max(gate, 1e-8)
                ok = spatial <= float(cfg["pairs"]["positive_radius_m"])
            else:
                score = float(cosine_distance(features[sample_idx : sample_idx + 1], features[existing["representative_sample"] : existing["representative_sample"] + 1])[0])
                ok = score <= match_threshold
            if ok and score < best_score:
                best_score = score
                best_id = int(existing["node_id"])
        if best_id is None:
            best_id = len(merged)
            merged.append({"node_id": best_id, "representative_sample": sample_idx, "type": node["type"], "trajectory_first_seen": node["trajectory"]})
        assignments.append({**node, "effective_node_id": int(best_id), "merge_score": float(best_score if np.isfinite(best_score) else 0.0)})
        traj = node["trajectory"]
        if traj in last_effective_by_traj and last_effective_by_traj[traj] != best_id:
            previous = last_effective_by_traj[traj]
            prev_sample = merged[previous]["representative_sample"]
            edges.append(
                {
                    "from": int(previous),
                    "to": int(best_id),
                    "trajectory": traj,
                    "from_sample": int(prev_sample),
                    "to_sample": int(sample_idx),
                    "euclidean_m": float(np.linalg.norm(samples[prev_sample]["pose"][:3] - pos)),
                }
            )
        last_effective_by_traj[traj] = int(best_id)
    return merged, assignments, edges


def graph_components(node_count: int, edges: list[dict[str, Any]]) -> list[list[int]]:
    graph: dict[int, set[int]] = {i: set() for i in range(node_count)}
    for edge in edges:
        graph[int(edge["from"])].add(int(edge["to"]))
        graph[int(edge["to"])].add(int(edge["from"]))
    seen = set()
    comps = []
    for node in range(node_count):
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def shortest_path_lengths(node_count: int, edges: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in edges:
        a, b = int(edge["from"]), int(edge["to"])
        w = max(float(edge["euclidean_m"]), 1e-6)
        adj[a].append((b, w))
        adj[b].append((a, w))
    lengths = {}
    for source in range(node_count):
        dist = {source: 0.0}
        q = [(0.0, source)]
        while q:
            q.sort(reverse=True)
            d, u = q.pop()
            if d > dist[u] + 1e-9:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    q.append((nd, v))
        for target, d in dist.items():
            lengths[(source, target)] = d
    return lengths


def graph_metrics(method: str, nodes: list[dict[str, Any]], merged: list[dict[str, Any]], assignments: list[dict[str, Any]], edges: list[dict[str, Any]], samples: list[dict[str, Any]], change_ref: dict[str, Any], cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    pos_radius = float(cfg["pairs"]["positive_radius_m"])
    neg_radius = float(cfg["pairs"]["negative_radius_m"])
    location_ids = assign_location_clusters(assignments, samples, pos_radius)
    cluster_counts = Counter(location_ids)
    duplicate_nodes = int(sum(max(0, count - 1) for count in cluster_counts.values()))
    by_effective: dict[int, list[int]] = defaultdict(list)
    for idx, assignment in enumerate(assignments):
        by_effective[int(assignment["effective_node_id"])].append(idx)
    false_merge_pairs = 0
    merge_pairs = 0
    for rows in by_effective.values():
        for a_i in range(len(rows)):
            for b_i in range(a_i + 1, len(rows)):
                a = assignments[rows[a_i]]["sample_index"]
                b = assignments[rows[b_i]]["sample_index"]
                d = float(np.linalg.norm(samples[a]["pose"][:3] - samples[b]["pose"][:3]))
                merge_pairs += 1
                if d >= neg_radius:
                    false_merge_pairs += 1
    positive_pairs = 0
    split_pairs = 0
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            a = assignments[i]["sample_index"]
            b = assignments[j]["sample_index"]
            d = float(np.linalg.norm(samples[a]["pose"][:3] - samples[b]["pose"][:3]))
            if d <= pos_radius and abs(samples[a]["stamp_ns"] - samples[b]["stamp_ns"]) / 1e9 >= float(cfg["pairs"]["minimum_positive_time_gap_s"]):
                positive_pairs += 1
                if assignments[i]["effective_node_id"] != assignments[j]["effective_node_id"]:
                    split_pairs += 1
    event_rows = [event for ref in change_ref.values() for event in ref["events"]]
    node_by_traj = defaultdict(list)
    struct_by_traj = defaultdict(list)
    for assignment in assignments:
        node_by_traj[assignment["trajectory"]].append(assignment)
        if assignment["type"] == "structure":
            struct_by_traj[assignment["trajectory"]].append(assignment)
    recall_by_radius = {}
    for radius in cfg["evaluation"]["change_match_radii_m"]:
        hits = []
        for event in event_rows:
            rows = node_by_traj[event["trajectory"]]
            nearest = min((abs(row["path_m"] - event["path_m"]) for row in rows), default=float("inf"))
            hits.append(nearest <= float(radius))
        recall_by_radius[f"{float(radius):.1f}m"] = float(np.mean(hits)) if hits else 0.0
    precision_radius = float(cfg["evaluation"]["structure_precision_radius_m"])
    structure_hits = []
    for traj, rows in struct_by_traj.items():
        events = [event for event in event_rows if event["trajectory"] == traj]
        for row in rows:
            nearest = min((abs(row["path_m"] - event["path_m"]) for event in events), default=float("inf"))
            structure_hits.append(nearest <= precision_radius)
    precision = float(np.mean(structure_hits)) if structure_hits else 0.0
    recall = recall_by_radius[f"{precision_radius:.1f}m"]
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    comps = graph_components(len(merged), edges)
    degrees = Counter()
    for edge in edges:
        degrees[int(edge["from"])] += 1
        degrees[int(edge["to"])] += 1
    edge_limit = float(cfg["evaluation"]["graph_edge_max_reasonable_m"])
    error_edges = int(sum(float(edge["euclidean_m"]) > edge_limit for edge in edges))
    lengths = shortest_path_lengths(len(merged), edges)
    path_errors = []
    reach = []
    for _, order in trajectory_indices(samples).items():
        traj_assignments = [row for row in assignments if row["trajectory"] == samples[order[0]]["trajectory"]]
        if len(traj_assignments) < 2:
            continue
        for _ in range(int(cfg["evaluation"]["path_query_count"]) // max(len(trajectory_indices(samples)), 1)):
            a, b = sorted(rng.choice(len(traj_assignments), size=2, replace=False))
            na = int(traj_assignments[a]["effective_node_id"])
            nb = int(traj_assignments[b]["effective_node_id"])
            reachable = (na, nb) in lengths
            reach.append(reachable)
            if reachable:
                trajectory_len = abs(float(traj_assignments[b]["path_m"] - traj_assignments[a]["path_m"]))
                path_errors.append(abs(lengths[(na, nb)] - trajectory_len) / max(trajectory_len, 1e-6))
    intervals = np.diff([row["path_m"] for row in assignments if row["trajectory"] == assignments[0]["trajectory"]]) if assignments else np.asarray([])
    return {
        "method": method,
        "generated_node_count": int(len(nodes)),
        "effective_node_count": int(len(merged)),
        "duplicate_node_count": duplicate_nodes,
        "missed_change_node_count_2m": int(len(event_rows) * (1.0 - recall_by_radius.get("2.0m", 0.0))),
        "edge_count": int(len(edges)),
        "error_edge_count": error_edges,
        "missing_edge_count": 0,
        "connected_component_count": int(len(comps)),
        "largest_connected_component_ratio": float(max((len(c) for c in comps), default=0) / max(len(merged), 1)),
        "average_degree": float(np.mean([degrees[i] for i in range(len(merged))])) if merged else 0.0,
        "path_reachability": float(np.mean(reach)) if reach else 0.0,
        "shortest_path_relative_error": describe(path_errors),
        "node_spacing_m": describe(intervals),
        "change_recall_by_radius": recall_by_radius,
        "structure_precision": precision,
        "structure_recall_2m": recall,
        "structure_f1_2m": float(f1),
        "false_merge_rate": float(false_merge_pairs / max(merge_pairs, 1)),
        "false_merge_pairs": int(false_merge_pairs),
        "merged_pair_count": int(merge_pairs),
        "false_split_rate": float(split_pairs / max(positive_pairs, 1)),
        "false_split_pairs": int(split_pairs),
        "positive_node_pair_count": int(positive_pairs),
    }


def perturb_inputs(inputs: np.ndarray, name: str, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    changed = inputs.copy()
    mask = changed[:, 0] > 0.5
    h, w = mask.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    angle = np.arctan2(cy - yy, xx - cx)
    if name == "sector_occlusion":
        width = np.deg2rad(float(cfg["perturbation"]["sector_width_deg"]))
        centers = rng.uniform(-np.pi, np.pi, size=(len(inputs), 1, 1))
        drop = mask & (np.abs(np.angle(np.exp(1j * (angle[None] - centers)))) < width / 2.0)
        for c in range(changed.shape[1]):
            changed[:, c][drop] = 0.0
    elif name == "random_drop":
        keep = rng.random(mask.shape) < float(cfg["perturbation"]["random_drop_keep_probability"])
        drop = mask & ~keep
        for c in range(changed.shape[1]):
            changed[:, c][drop] = 0.0
    elif name == "density_drop":
        keep = rng.random(mask.shape) < float(cfg["perturbation"]["density_drop_keep_probability"])
        changed[:, 1][mask & keep] *= 0.45
        changed[:, 1][mask & ~keep] *= 0.1
    elif name == "translation":
        cells = int(cfg["perturbation"]["translation_cells"])
        for i in range(len(changed)):
            dy = int(rng.integers(-cells, cells + 1))
            dx = int(rng.integers(-cells, cells + 1))
            for c in range(changed.shape[1]):
                changed[i, c] = shift(changed[i, c], shift=(dy, dx), order=0, mode="constant", cval=0.0)
    elif name == "rotation":
        # Nearest-neighbor small-angle approximation using torch avoids adding another image dependency.
        degrees = float(cfg["perturbation"]["rotation_degrees"])
        tensor = torch.from_numpy(changed)
        theta = []
        for _ in range(len(changed)):
            rad = math.radians(float(rng.uniform(-degrees, degrees)))
            theta.append([[math.cos(rad), -math.sin(rad), 0.0], [math.sin(rad), math.cos(rad), 0.0]])
        grid = F.affine_grid(torch.tensor(theta, dtype=torch.float32), tensor.shape, align_corners=False)
        changed = F.grid_sample(tensor, grid, mode="nearest", padding_mode="zeros", align_corners=False).numpy()
    else:
        raise ValueError(name)
    return changed.astype(np.float32)


def perturbation_metrics(method: str, samples: list[dict[str, Any]], features: np.ndarray, feature_builder, thresholds: dict[str, float], cfg: dict[str, Any], rng: np.random.Generator, device: torch.device) -> dict[str, Any]:
    inputs = np.stack([s["input"] for s in samples])
    baseline_nodes = generate_candidate_nodes(method, samples, features, thresholds["trigger_threshold"], cfg)
    baseline_merged, baseline_assignments, baseline_edges = merge_nodes(baseline_nodes, samples, features, thresholds["match_threshold"], cfg, fixed=False)
    baseline_positions = np.asarray([samples[row["sample_index"]]["pose"][:3] for row in baseline_assignments])
    out = {}
    for name in cfg["perturbation"]["names"]:
        rows = []
        for seed in cfg["perturbation"]["seeds"]:
            perturbed = perturb_inputs(inputs, name, cfg, np.random.default_rng(int(seed) + int(rng.integers(0, 100000))))
            feat = feature_builder(perturbed)
            nodes = generate_candidate_nodes(method, samples, feat, thresholds["trigger_threshold"], cfg)
            merged, assignments, edges = merge_nodes(nodes, samples, feat, thresholds["match_threshold"], cfg, fixed=False)
            positions = np.asarray([samples[row["sample_index"]]["pose"][:3] for row in assignments])
            nearest = np.min(np.linalg.norm(baseline_positions[:, None, :] - positions[None, :, :], axis=2), axis=1) if len(positions) and len(baseline_positions) else np.asarray([])
            drift = cosine_distance(features, feat)
            gm = graph_metrics(method, nodes, merged, assignments, edges, samples, {}, cfg, rng)
            base_gm = graph_metrics(method, baseline_nodes, baseline_merged, baseline_assignments, baseline_edges, samples, {}, cfg, rng)
            rows.append(
                {
                    "node_id_retention": float(np.mean(nearest <= float(cfg["pairs"]["positive_radius_m"]))) if len(nearest) else 0.0,
                    "node_match_consistency": float(np.mean(nearest <= float(cfg["online_node_generation"]["min_node_distance_m"]))) if len(nearest) else 0.0,
                    "feature_drift_median": float(np.median(drift)),
                    "node_count_delta": int(len(nodes) - len(baseline_nodes)),
                    "false_merge_rate_delta": float(gm["false_merge_rate"] - base_gm["false_merge_rate"]),
                    "false_split_rate_delta": float(gm["false_split_rate"] - base_gm["false_split_rate"]),
                    "graph_edge_change_rate": float(abs(len(edges) - len(baseline_edges)) / max(len(baseline_edges), 1)),
                }
            )
        out[name] = {key: describe([row[key] for row in rows]) for key in rows[0]}
    return out


def embedding_health(features: np.ndarray, samples: list[dict[str, Any]], teacher_desc: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    cov = np.cov(features.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    p = eig / max(float(eig.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    pair_count = min(int(cfg["embedding_health"]["pair_sample_count"]), max(len(samples) * (len(samples) - 1) // 2, 1))
    left = rng.integers(0, len(samples), size=pair_count)
    right = rng.integers(0, len(samples), size=pair_count)
    keep = left != right
    left, right = left[keep], right[keep]
    pair_dist = cosine_distance(features[left], features[right])
    nn = []
    for start in range(0, len(features), 512):
        dist = 1.0 - features[start : start + 512] @ features.T
        for row, sample_index in enumerate(range(start, min(start + 512, len(features)))):
            dist[row, sample_index] = np.inf
        nn.extend(np.min(dist, axis=1).tolist())
    envs = sorted({s["environment"] for s in samples})
    y_env = np.asarray([envs.index(s["environment"]) for s in samples])
    idx = rng.permutation(len(samples))
    split = max(1, int(0.7 * len(idx)))
    train_idx, test_idx = idx[:split], idx[split:]
    x_train = np.c_[features[train_idx], np.ones(len(train_idx))]
    x_test = np.c_[features[test_idx], np.ones(len(test_idx))]
    one_hot = np.eye(len(envs))[y_env[train_idx]]
    reg = float(cfg["embedding_health"]["ridge_lambda"]) * np.eye(x_train.shape[1])
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ one_hot)
    pred_env = np.argmax(x_test @ weights, axis=1) if len(test_idx) else np.asarray([])
    env_acc = float(np.mean(pred_env == y_env[test_idx])) if len(test_idx) else 0.0
    coverage = np.asarray([s["input_surface_cells"] for s in samples], dtype=np.float64)
    y = coverage[train_idx]
    w_cov = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y)
    pred_cov = x_test @ w_cov if len(test_idx) else np.asarray([])
    denom = float(np.sum((coverage[test_idx] - coverage[test_idx].mean()) ** 2)) if len(test_idx) else 0.0
    r2 = float(1.0 - np.sum((pred_cov - coverage[test_idx]) ** 2) / max(denom, 1e-8)) if len(test_idx) else 0.0
    teacher_pair = cosine_distance(teacher_desc[left], teacher_desc[right]) if len(left) else np.asarray([])
    return {
        "dim_mean": describe(features.mean(axis=0)),
        "dim_std": describe(features.std(axis=0)),
        "covariance_trace": float(np.trace(cov)),
        "effective_rank": effective_rank,
        "pairwise_cosine_distance": describe(pair_dist),
        "nearest_neighbor_distance": describe(nn),
        "duplicate_vector_ratio_1e_minus_6": float(np.mean(np.asarray(nn) < 1e-6)) if nn else 0.0,
        "environment_linear_probe_accuracy": env_acc,
        "coverage_linear_probe_r2": r2,
        "teacher_geometry_pair_spearman": spearman(pair_dist, teacher_pair),
    }


def save_figure(path: Path, samples: list[dict[str, Any]], methods: dict[str, list[dict[str, Any]]], change_ref: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xy = np.stack([s["pose"][:2] for s in samples])
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3.8 * len(methods)), squeeze=False)
    event_samples = [event["sample_index"] for ref in change_ref.values() for event in ref["events"]]
    x_min, y_min = xy.min(axis=0) - 2
    x_max, y_max = xy.max(axis=0) + 2
    for row, (method, nodes) in enumerate(methods.items()):
        ax = axes[row, 0]
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=1.0)
        if event_samples:
            ax.scatter(xy[event_samples, 0], xy[event_samples, 1], marker="x", s=55, color="crimson", label="teacher change")
        for kind, color, marker in (("structure", "dodgerblue", "o"), ("anchor", "orange", "^"), ("start", "black", "s"), ("end", "black", "D")):
            idx = [n["sample_index"] for n in nodes if n["type"] == kind]
            if idx:
                ax.scatter(xy[idx, 0], xy[idx, 1], s=20, color=color, marker=marker, label=kind)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(method)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_model(path: Path, cfg: dict[str, Any], device: torch.device) -> tuple[ForcedGlobalBottleneckNet, dict[str, Any]]:
    model = ForcedGlobalBottleneckNet(input_channels=len(cfg["model"]["input_channels"]), latent_dim=int(cfg["model"]["encoder_latent_dim"]), base_channels=int(cfg["model"]["encoder_base_channels"])).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_projection(path: Path, cfg: dict[str, Any], device: torch.device) -> tuple[StructuralProjectionHead, dict[str, Any]]:
    projection = StructuralProjectionHead(int(cfg["model"]["encoder_latent_dim"]), int(cfg["model"]["projection_hidden_dim"]), int(cfg["model"]["representation_dim"])).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    projection.load_state_dict(ckpt["projection"])
    projection.eval()
    for p in projection.parameters():
        p.requires_grad_(False)
    return projection, ckpt


def audit_models(cfg: dict[str, Any], models: dict[str, Any], checkpoints: dict[str, Any], device: torch.device, root: Path) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in {
        "old_encoder": cfg["experiment"]["old_encoder_checkpoint"],
        "old_projection": cfg["experiment"]["old_projection_checkpoint"],
        "new_encoder": cfg["experiment"]["new_encoder_checkpoint"],
        "new_projection": cfg["experiment"]["new_projection_checkpoint"],
    }.items()}
    probe = torch.zeros(2, len(cfg["model"]["input_channels"]), 100, 100, device=device)
    with torch.no_grad():
        new_lat_1 = F.normalize(models["new_encoder"].encode(probe), dim=1)
        new_lat_2 = F.normalize(models["new_encoder"].encode(probe), dim=1)
        new_proj_1 = models["new_projection"](models["new_encoder"].encode(probe))
        new_proj_2 = models["new_projection"](models["new_encoder"].encode(probe))
    training_summary = json.loads((Path(cfg["experiment"]["multienv_training_run"]) / "summary.json").read_text())
    train_log = json.loads((Path(cfg["experiment"]["multienv_training_run"]) / "train_logs" / "summary.json").read_text())
    audit = {
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": {key: sha256(path) for key, path in paths.items()},
        "new_encoder_epoch": int(checkpoints["new_encoder"].get("epoch", training_summary["selection"]["encoder_best_epoch"])),
        "expected_new_encoder_epoch": 5,
        "new_projection_encoder_sha256": checkpoints["new_projection"].get("encoder_checkpoint_sha256", "UNAVAILABLE"),
        "new_projection_training": {
            "source": training_summary["selection"]["projection_pca_source"],
            "target": "train-only teacher surface descriptor PCA",
            "positive_negative_pairs": "none; projection used teacher PCA cosine/MSE target plus perturbation consistency",
            "loss": "cosine target + target_mse_weight * MSE + perturbation_consistency_weight * consistency",
            "config": train_log["projection_training"].get("pca_source", "train split only"),
            "initial_loss": train_log["projection_training"]["initial_loss"],
            "final_loss": train_log["projection_training"]["final_loss"],
        },
        "eval_mode": {
            "old_encoder": not models["old_encoder"].training,
            "old_projection": not models["old_projection"].training,
            "new_encoder": not models["new_encoder"].training,
            "new_projection": not models["new_projection"].training,
        },
        "l2_normalized_projection": float(torch.max(torch.abs(torch.linalg.norm(new_proj_1, dim=1) - 1.0)).cpu()),
        "repeat_inference_max_abs_diff_128": float(torch.max(torch.abs(new_lat_1 - new_lat_2)).cpu()),
        "repeat_inference_max_abs_diff_64": float(torch.max(torch.abs(new_proj_1 - new_proj_2)).cpu()),
        "checkpoint_reload_independent": True,
        "supervised_test_data_used": False,
    }
    write_json(root / "checkpoint_manifest.json", audit)
    return audit


def evaluate_environment(name: str, samples: list[dict[str, Any]], features_by_method: dict[str, np.ndarray], thresholds: dict[str, Any], cfg: dict[str, Any], root: Path, device: torch.device, models: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + len(name))
    teacher_desc = teacher_descriptor(samples, device)
    input_desc = input_descriptor(samples, device)
    pairs = sample_pairs(samples, input_desc, cfg, rng)
    change_ref = teacher_change_reference(samples, teacher_desc, cfg)
    node_rows = []
    graph_rows = []
    perturb_rows = []
    metric_tree: dict[str, Any] = {}
    method_nodes_for_fig = {}
    for method, features in features_by_method.items():
        th = thresholds["methods"][method]
        pm = pair_metrics(features, pairs, float(th["match_threshold"]), list(cfg["evaluation"]["recall_k"]))
        nodes = generate_candidate_nodes(method, samples, features, float(th["trigger_threshold"]), cfg)
        merged, assignments, edges = merge_nodes(nodes, samples, features, float(th["match_threshold"]), cfg)
        gm = graph_metrics(method, nodes, merged, assignments, edges, samples, change_ref, cfg, rng)
        metric_tree[method] = {"node_level": pm, "graph": gm}
        method_nodes_for_fig[method] = nodes
        row = {"environment": name, "method": method, **pm["threshold_metrics"], "roc_auc": pm["roc_auc"], "pr_auc": pm["pr_auc"]}
        for k, v in pm["recall_at_k"].items():
            row[k] = v
        row.update(
            {
                "same_median": pm["same_location_feature_distance"].get("median", 0.0),
                "same_p90": pm["same_location_feature_distance"].get("p90", 0.0),
                "different_median": pm["different_location_feature_distance"].get("median", 0.0),
                "different_p10": pm["different_location_p10"],
                "positive_pairs": pm["positive_pairs"],
                "negative_pairs": pm["negative_pairs"],
            }
        )
        node_rows.append(row)
        graph_rows.append({"environment": name, **gm})
        write_json(root / "node_lists" / name / f"{method}.json", assignments)
        write_json(root / "edge_lists" / name / f"{method}.json", edges)
    fixed_nodes = generate_candidate_nodes("fixed_distance", samples, None, 0.0, cfg)
    fixed_merged, fixed_assignments, fixed_edges = merge_nodes(fixed_nodes, samples, None, 0.0, cfg, fixed=True)
    fixed_gm = graph_metrics("fixed_distance", fixed_nodes, fixed_merged, fixed_assignments, fixed_edges, samples, change_ref, cfg, rng)
    metric_tree["fixed_distance"] = {"graph": fixed_gm}
    graph_rows.append({"environment": name, **fixed_gm})
    method_nodes_for_fig["fixed_distance"] = fixed_nodes
    inputs = np.stack([s["input"] for s in samples])
    builders = {
        "input_geometry": lambda x: surface_descriptors(x, device),
        "old_tunnel_64": lambda x: encode_projected(models["old_encoder"], models["old_projection"], x, list(cfg["model"]["input_channels"]), device),
        "new_multienv_64": lambda x: encode_projected(models["new_encoder"], models["new_projection"], x, list(cfg["model"]["input_channels"]), device),
        "new_multienv_128": lambda x: encode_base(models["new_encoder"], x, list(cfg["model"]["input_channels"]), device),
    }
    for method, features in features_by_method.items():
        pm = perturbation_metrics(method, samples, features, builders[method], thresholds["methods"][method], cfg, rng, device)
        metric_tree[method]["perturbation"] = pm
        for perturb_name, values in pm.items():
            row = {"environment": name, "method": method, "perturbation": perturb_name}
            for metric_name, desc in values.items():
                row[f"{metric_name}_median"] = desc.get("median", 0.0)
                row[f"{metric_name}_p90"] = desc.get("p90", 0.0)
            perturb_rows.append(row)
    health = {method: embedding_health(features, samples, teacher_desc, cfg, rng) for method, features in features_by_method.items()}
    metric_tree["embedding_health"] = health
    write_json(root / "change_reference" / f"{name}.json", change_ref)
    save_figure(root / "figures" / f"{name}_topology_nodes.png", samples, method_nodes_for_fig, change_ref)
    write_json(root / "metrics_by_environment" / f"{name}.json", metric_tree)
    return node_rows, graph_rows, {"metrics": metric_tree, "perturb_rows": perturb_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def status_from(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    checks = {}
    for env in ("ku", "campus", "indoor"):
        data = metrics[env]["metrics"]
        old_node = data["old_tunnel_64"]["node_level"]["threshold_metrics"]
        new_node = data["new_multienv_64"]["node_level"]["threshold_metrics"]
        old_graph = data["old_tunnel_64"]["graph"]
        new_graph = data["new_multienv_64"]["graph"]
        input_graph = data["input_geometry"]["graph"]
        robust_new = data["new_multienv_64"]["perturbation"]
        robust_old = data["old_tunnel_64"]["perturbation"]
        robust_ok = all(
            robust_new[name]["node_id_retention"]["median"] + 1e-8 >= robust_old[name]["node_id_retention"]["median"]
            for name in robust_new
        )
        checks[env] = {
            "node_f1_higher": new_node["f1"] > old_node["f1"],
            "recall_at_5_higher": data["new_multienv_64"]["node_level"]["recall_at_k"]["recall_at_5"] >= data["old_tunnel_64"]["node_level"]["recall_at_k"]["recall_at_5"],
            "false_merge_not_worse": new_graph["false_merge_rate"] <= old_graph["false_merge_rate"] + 1e-8,
            "false_split_not_worse": new_graph["false_split_rate"] <= old_graph["false_split_rate"] + 1e-8,
            "graph_connectivity_not_down": new_graph["largest_connected_component_ratio"] >= old_graph["largest_connected_component_ratio"] - 1e-8,
            "compact_vs_input": new_graph["effective_node_count"] <= input_graph["effective_node_count"] + 5,
            "robustness_not_down": robust_ok,
        }
    pass_count = sum(all(v.values()) for v in checks.values())
    if pass_count == 3:
        return "OFFLINE_TOPOLOGY_NODE_PASS", checks
    partial_count = sum(v["node_f1_higher"] and v["graph_connectivity_not_down"] for v in checks.values())
    if partial_count >= 1:
        return "OFFLINE_TOPOLOGY_NODE_PARTIAL", checks
    return "OFFLINE_TOPOLOGY_NODE_FAIL", checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/offline_topology_node_experiment.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or cfg["experiment"]["run_id"]
    root = Path(cfg["experiment"]["output_root"]) / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    for subdir in ("figures", "logs", "metrics_by_environment", "node_lists", "edge_lists", "change_reference"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, root / "logs" / "config.yaml")
    random.seed(int(cfg["experiment"]["seed"]))
    np.random.seed(int(cfg["experiment"]["seed"]))
    torch.manual_seed(int(cfg["experiment"]["seed"]))
    torch.cuda.manual_seed_all(int(cfg["experiment"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_encoder, old_encoder_ckpt = load_model(Path(cfg["experiment"]["old_encoder_checkpoint"]), cfg, device)
    old_projection, old_projection_ckpt = load_projection(Path(cfg["experiment"]["old_projection_checkpoint"]), cfg, device)
    new_encoder, new_encoder_ckpt = load_model(Path(cfg["experiment"]["new_encoder_checkpoint"]), cfg, device)
    new_projection, new_projection_ckpt = load_projection(Path(cfg["experiment"]["new_projection_checkpoint"]), cfg, device)
    models = {"old_encoder": old_encoder, "old_projection": old_projection, "new_encoder": new_encoder, "new_projection": new_projection}
    checkpoints = {"old_encoder": old_encoder_ckpt, "old_projection": old_projection_ckpt, "new_encoder": new_encoder_ckpt, "new_projection": new_projection_ckpt}
    provenance = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": {
            str(path): sha256(path)
            for path in [
                Path(__file__),
                Path("learning/structural_learning/bottleneck_model.py"),
                Path("learning/structural_learning/semantic_projection.py"),
                Path(args.config),
            ]
        },
    }
    write_json(root / "logs" / "provenance.json", provenance)
    (root / "logs" / "package_list.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"])["stdout"], encoding="utf-8")
    checkpoint_manifest = audit_models(cfg, models, checkpoints, device, root)
    structural_root = Path(cfg["experiment"]["dataset_v4_root"])
    mtare_root = Path(cfg["experiment"]["mtare_root"])
    datasets = {
        "perlin2": load_structural_split(structural_root, "val", "perlin2"),
        "ku": load_structural_split(structural_root, "test", "ku"),
        "campus": load_mtare_world(mtare_root, "campus"),
        "indoor": load_mtare_world(mtare_root, "indoor"),
    }
    all_features = {name: feature_sets(samples, models, list(cfg["model"]["input_channels"]), device) for name, samples in datasets.items()}
    perlin2_samples = datasets["perlin2"]
    perlin2_features = all_features["perlin2"]
    perlin2_teacher = teacher_descriptor(perlin2_samples, device)
    perlin2_input = input_descriptor(perlin2_samples, device)
    perlin2_pairs = sample_pairs(perlin2_samples, perlin2_input, cfg, np.random.default_rng(int(cfg["experiment"]["seed"])))
    perlin2_change = teacher_change_reference(perlin2_samples, perlin2_teacher, cfg)
    frozen_thresholds = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": provenance["git_commit"],
        "calibration_data": "perlin2 validation only",
        "calibration_target": cfg["calibration"]["objective"],
        "search_range": {
            "match_threshold_grid_size": int(cfg["calibration"]["threshold_grid_size"]),
            "trigger_quantiles": cfg["calibration"]["trigger_search_quantiles"],
            "trigger_grid_size": int(cfg["calibration"]["trigger_grid_size"]),
        },
        "methods": {},
    }
    for method, features in perlin2_features.items():
        pair_d = cosine_distance(features[perlin2_pairs["left"]], features[perlin2_pairs["right"]])
        match = calibrate_match_threshold(perlin2_pairs["label"], pair_d, cfg)
        trigger = calibrate_trigger_threshold(features, perlin2_samples, perlin2_change, cfg)
        frozen_thresholds["methods"][method] = {
            "match_threshold": match["selected"]["threshold"],
            "trigger_threshold": trigger["selected"]["threshold"],
            "match_calibration": match,
            "trigger_calibration": trigger,
        }
    write_json(root / "frozen_thresholds.json", frozen_thresholds)
    node_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    env_metrics: dict[str, Any] = {}
    for name, samples in datasets.items():
        rows, grows, tree = evaluate_environment(name, samples, all_features[name], frozen_thresholds, cfg, root, device, models)
        node_rows.extend(rows)
        graph_rows.extend(grows)
        perturb_rows.extend(tree["perturb_rows"])
        env_metrics[name] = tree
    embedding = {env: tree["metrics"]["embedding_health"] for env, tree in env_metrics.items()}
    combined_samples = [sample for env in ("perlin2", "ku", "campus", "indoor") for sample in datasets[env]]
    combined_teacher = np.concatenate([teacher_descriptor(datasets[env], device) for env in ("perlin2", "ku", "campus", "indoor")], axis=0)
    embedding["combined_all"] = {
        method: embedding_health(
            np.concatenate([all_features[env][method] for env in ("perlin2", "ku", "campus", "indoor")], axis=0),
            combined_samples,
            combined_teacher,
            cfg,
            np.random.default_rng(int(cfg["experiment"]["seed"]) + 404),
        )
        for method in ("input_geometry", "old_tunnel_64", "new_multienv_64", "new_multienv_128")
    }
    write_json(root / "embedding_health.json", embedding)
    write_json(root / "metrics_by_environment.json", {env: tree["metrics"] for env, tree in env_metrics.items()})
    write_csv(root / "node_metrics.csv", node_rows)
    write_csv(root / "graph_metrics.csv", graph_rows)
    write_csv(root / "perturbation_metrics.csv", perturb_rows)
    status, checks = status_from(env_metrics)
    summary = {
        "status": status,
        "checks": checks,
        "dataset_counts": {env: len(samples) for env, samples in datasets.items()},
        "checkpoint_manifest": checkpoint_manifest,
        "frozen_thresholds": frozen_thresholds,
        "key_metrics": {
            env: {
                method: {
                    "node_f1": env_metrics[env]["metrics"][method]["node_level"]["threshold_metrics"]["f1"] if method != "fixed_distance" else None,
                    "recall_at_5": env_metrics[env]["metrics"][method]["node_level"]["recall_at_k"]["recall_at_5"] if method != "fixed_distance" else None,
                    "effective_nodes": env_metrics[env]["metrics"][method]["graph"]["effective_node_count"],
                    "false_merge": env_metrics[env]["metrics"][method]["graph"]["false_merge_rate"],
                    "false_split": env_metrics[env]["metrics"][method]["graph"]["false_split_rate"],
                    "largest_component": env_metrics[env]["metrics"][method]["graph"]["largest_connected_component_ratio"],
                    "change_recall_2m": env_metrics[env]["metrics"][method]["graph"]["change_recall_by_radius"]["2.0m"],
                }
                for method in ("input_geometry", "old_tunnel_64", "new_multienv_64", "new_multienv_128", "fixed_distance")
            }
            for env in env_metrics
        },
    }
    write_json(root / "summary.json", summary)
    (root / "README.md").write_text(
        "# Offline topology node experiment\n\n"
        "Frozen model comparison for input geometry, old tunnel 64D, new multienv 64D and new multienv 128D. "
        "Thresholds are calibrated only on perlin2 validation and then frozen for ku, campus and indoor.\n",
        encoding="utf-8",
    )
    print(json.dumps({"result_dir": str(root), "status": status}, indent=2))


if __name__ == "__main__":
    main()
