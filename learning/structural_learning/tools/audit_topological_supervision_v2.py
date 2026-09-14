from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.structural_learning.tools.train_topological_semantic_model import (  # noqa: E402
    TopologicalSemanticDataset,
    TopologicalSemanticNet,
    collate,
    target_tensors,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): clean(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(val) for val in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def desc(values: Any) -> dict[str, Any]:
    arr = finite(values)
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


def safe_spearman(a: Any, b: Any) -> float:
    x, y = finite(a), finite(b)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def binary(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    truth = y >= 0.5
    pred = p >= threshold
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "threshold": float(threshold), "precision": precision, "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1.0) + tn / max(tn + fp, 1.0)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def binary_summary(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    result = {"positive_rate": float((y >= 0.5).mean()), "prediction_positive_rate": float((p >= threshold).mean())}
    result["at_threshold"] = binary(y, p, threshold)
    if len(np.unique(y >= 0.5)) > 1:
        result["pr_auc"] = float(average_precision_score(y >= 0.5, p))
        result["roc_auc"] = float(roc_auc_score(y >= 0.5, p))
    else:
        result["pr_auc"] = float("nan")
        result["roc_auc"] = float("nan")
    return result


def load_arrays(dataset: TopologicalSemanticDataset) -> dict[str, Any]:
    rows = [dataset[i] for i in range(len(dataset))]
    keys = ["direction", "distance", "area", "exit", "scores", "role", "canonical_role", "current", "history", "history_valid_mask", "relative_poses", "local_connectivity"]
    result = {}
    for key in keys:
        if key == "canonical_role":
            result[key] = np.stack([np.load(row["path"], allow_pickle=False)["rotation_canonical_role_teacher"].astype(np.float32) for row in rows])
        else:
            result[key] = np.stack([row[key] for row in rows])
    result["world"] = np.asarray([row["world"] for row in rows])
    result["trajectory"] = np.asarray([row["trajectory"] for row in rows])
    result["path"] = np.asarray([row["path"] for row in rows])
    result["timestamp"] = np.asarray([row["timestamp"] for row in rows], dtype=np.int64)
    result["raw_point_count"] = np.asarray([row["raw_point_count"] for row in rows], dtype=np.float64)
    return result


def score_baselines(arr: dict[str, Any], train: dict[str, Any], score_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target_name in ["direction", "exit"]:
        y = arr[target_name]
        train_y = train[target_name]
        priors = train_y.mean(axis=0)
        majority = (priors >= 0.5).astype(np.float32)
        candidates = {
            "all_negative": np.zeros_like(y),
            "all_positive": np.ones_like(y),
            "train_directional_probability": np.broadcast_to(priors, y.shape),
            "train_majority": np.broadcast_to(majority, y.shape),
        }
        out[target_name] = {name: {"per_direction": [binary_summary(y[:, i], pred[:, i]) for i in range(y.shape[1])],
                                   "macro_f1": float(np.mean([binary(y[:, i], pred[:, i], 0.5)["f1"] for i in range(y.shape[1])]))}
                            for name, pred in candidates.items()}
    y_dist = arr["distance"]
    mean_dir = train["distance"].mean(axis=0)
    global_mean = float(train["distance"].mean())
    dist_pred = {
        "all_zero": np.zeros_like(y_dist),
        "train_global_mean": np.full_like(y_dist, global_mean),
        "train_directional_mean": np.broadcast_to(mean_dir, y_dist.shape),
        "direction_positive_fixed_0_5": (arr["direction"] >= 0.5).astype(np.float32) * 0.5,
    }
    valid = (arr["direction"] >= 0.5) | (arr["distance"] > 0.01)
    out["distance"] = {}
    for name, pred in dist_pred.items():
        err = np.abs(pred - y_dist)
        valid_err = err[valid]
        out["distance"][name] = {
            "mae": float(err.mean()), "median_ae": float(np.median(err)),
            "valid_mae": float(valid_err.mean()) if len(valid_err) else float("nan"),
            "near_mae": float(err[(y_dist > 0) & (y_dist < 0.35)].mean()) if np.any((y_dist > 0) & (y_dist < 0.35)) else float("nan"),
            "far_mae": float(err[y_dist >= 0.65].mean()) if np.any(y_dist >= 0.65) else float("nan"),
        }
    out["scores"] = {}
    for i, name in enumerate(score_names):
        out["scores"][name] = {}
        for pred_name, pred in {"train_mean": np.full(len(arr["scores"]), train["scores"][:, i].mean()),
                                "train_median": np.full(len(arr["scores"]), np.median(train["scores"][:, i]))}.items():
            err = np.abs(pred - arr["scores"][:, i])
            out["scores"][name][pred_name] = {"mae": float(err.mean()), "spearman": safe_spearman(pred, arr["scores"][:, i])}
    return out


def circular_runs(x: np.ndarray) -> list[int]:
    x = np.asarray(x, dtype=bool)
    if not x.any():
        return []
    if x.all():
        return [len(x)]
    start = int(np.where(~x)[0][0])
    y = np.roll(x, -start)
    runs = []
    i = 0
    while i < len(y):
        if y[i]:
            j = i
            while j < len(y) and y[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs


def exit_audit(arrays: dict[str, Any], split: str) -> dict[str, Any]:
    y = arrays["exit"] >= 0.5
    direction = arrays["direction"] >= 0.5
    distance = arrays["distance"]
    by_world = {}
    for world in sorted(set(arrays["world"].tolist())):
        idx = arrays["world"] == world
        yw, dw, lw = y[idx], direction[idx], distance[idx]
        runs = [run for row in yw for run in circular_runs(row)]
        positive_counts = yw.sum(1)
        logical = {
            "exit_implies_direction_violation_rate": float(np.logical_and(yw, ~dw).mean()),
            "exit_without_distance_violation_rate": float(np.logical_and(yw, lw <= 0.05).mean()),
            "exit_positive_sample_rate": float((positive_counts > 0).mean()),
            "all_zero_sample_rate": float((positive_counts == 0).mean()),
            "multi_exit_sample_rate": float((positive_counts > 1).mean()),
        }
        by_world[world] = {
            "samples": int(idx.sum()),
            "positive_rate": float(yw.mean()),
            "direction_positive_rate": float(dw.mean()),
            "positive_count_distribution": desc(positive_counts),
            "direction_positive_rates": yw.mean(0).tolist(),
            "continuous_positive_bin_width": desc(runs),
            "mean_positive_bins_per_sample": float(positive_counts.mean()),
            "logical_consistency": logical,
        }
    return {"split": split, "overall": {
        "samples": int(len(y)), "positive_rate": float(y.mean()),
        "positive_sample_rate": float((y.sum(1) > 0).mean()), "all_zero_rate": float((y.sum(1) == 0).mean()),
        "multi_exit_rate": float((y.sum(1) > 1).mean()),
        "positive_bins_per_sample": desc(y.sum(1)), "run_width": desc([r for row in y for r in circular_runs(row)]),
        "logical_consistency": {
            "exit_implies_direction_violation_rate": float(np.logical_and(y, ~direction).mean()),
            "exit_without_distance_violation_rate": float(np.logical_and(y, distance <= 0.05).mean()),
        },
    }, "by_world": by_world}


@torch.no_grad()
def collect_model_outputs(model: nn.Module, dataset: TopologicalSemanticDataset, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    out = defaultdict(list)
    for raw in loader:
        batch = target_tensors(raw, device)
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        out["exit_prob"].append(torch.sigmoid(pred["exit_logits"]).cpu().numpy())
        out["direction_prob"].append(torch.sigmoid(pred["direction_logits"]).cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}


def exit_threshold_audit(model: nn.Module, datasets: dict[str, TopologicalSemanticDataset], cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for name, ds in datasets.items():
        arrays = load_arrays(ds)
        pred = collect_model_outputs(model, ds, cfg, device)
        y, p = arrays["exit"], pred["exit_prob"]
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s = [binary(y.reshape(-1), p.reshape(-1), float(t))["f1"] for t in thresholds]
        best = int(np.argmax(f1s))
        result[name] = {
            "sample_count": int(len(y)), "logit_proxy_probability": {
                "positive": desc(p[y >= 0.5]), "negative": desc(p[y < 0.5])},
            "default_threshold": binary_summary(y, p),
            "best_threshold_on_split": {"threshold": float(thresholds[best]), "f1": float(f1s[best]), "precision": binary(y.reshape(-1), p.reshape(-1), thresholds[best])["precision"], "recall": binary(y.reshape(-1), p.reshape(-1), thresholds[best])["recall"]},
            "threshold_curve": [{"threshold": float(t), "f1": float(f)} for t, f in zip(thresholds, f1s)],
            "predicted_positive_count_default": int((p >= 0.5).sum()),
            "pr_auc": binary_summary(y, p)["pr_auc"], "roc_auc": binary_summary(y, p)["roc_auc"],
        }
    return result


def small_exit_fit(model: nn.Module, train_ds: TopologicalSemanticDataset, cfg: dict[str, Any], device: torch.device, seed: int) -> dict[str, Any]:
    arrays = load_arrays(train_ds)
    positive = np.where((arrays["exit"].sum(1) > 0))[0]
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(positive, min(32, len(positive)), replace=False))
    subset = Subset(train_ds, selected.tolist())
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate)
    raw = next(iter(loader))
    batch = target_tensors(raw, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.exit_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.exit_head.parameters(), lr=0.02)
    initial = None
    records = []
    for epoch in range(121):
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        loss = F.binary_cross_entropy_with_logits(pred["exit_logits"], batch["exit"])
        if initial is None:
            initial = float(loss.detach())
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in {0, 1, 10, 40, 80, 120}:
            with torch.no_grad():
                prob = torch.sigmoid(pred["exit_logits"]).cpu().numpy()
            records.append({"epoch": epoch, "loss": float(loss.detach()), "f1_0.5": binary(arrays["exit"][selected].reshape(-1), prob.reshape(-1), 0.5)["f1"]})
    with torch.no_grad():
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        p = torch.sigmoid(pred["exit_logits"]).cpu().numpy()
    y = arrays["exit"][selected]
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = thresholds[int(np.argmax([binary(y.reshape(-1), p.reshape(-1), float(t))["f1"] for t in thresholds]))]
    return {"selected_count": int(len(selected)), "selected_indices": selected.tolist(), "positive_sample_count": int(len(positive)), "records": records, "initial_loss": initial, "final_loss": float(records[-1]["loss"]), "best_f1": binary(y.reshape(-1), p.reshape(-1), float(best_t))["f1"], "best_threshold": float(best_t), "fit_success": bool(records[-1]["loss"] < 0.35 * initial and binary(y.reshape(-1), p.reshape(-1), float(best_t))["f1"] > 0.85)}


def history_audit(arrays: dict[str, Any]) -> dict[str, Any]:
    current = arrays["current"][:, 0] >= 0.5
    history = arrays["history"][:, :, 0] >= 0.5
    valid = arrays["history_valid_mask"] >= 0.5
    union = current.copy()
    evidence_counts = np.zeros_like(current, dtype=np.float32)
    for t in range(history.shape[1]):
        evidence_counts += history[:, t] * valid[:, t, None, None]
        union |= history[:, t] & valid[:, t, None, None]
    current_count = current.reshape(len(current), -1).sum(1)
    union_count = union.reshape(len(union), -1).sum(1)
    new_cells = np.maximum(union_count - current_count, 0)
    age = np.where(evidence_counts > 0, evidence_counts, 0)
    sectors = 16
    h, w = current.shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    angles = np.arctan2(xx - (w - 1) / 2, (h - 1) / 2 - yy)
    sector = ((angles % (2 * np.pi)) / (2 * np.pi) * sectors).astype(int) % sectors
    sector_gain = []
    for row in range(len(current)):
        gains = []
        for s in range(sectors):
            region = sector == s
            gains.append(float(np.maximum(union[row][region].sum() - current[row][region].sum(), 0)))
        sector_gain.append(gains)
    return {"current_cells": desc(current_count), "history_union_cells": desc(union_count), "new_history_cells": desc(new_cells), "relative_gain": desc(new_cells / np.maximum(current_count, 1)), "valid_history_length": desc(valid.sum(1)), "history_evidence_count": desc(age), "sector_gain": desc(np.asarray(sector_gain)), "samples_with_new_cells": float((new_cells > 0).mean()), "teacher_direction_vs_current_cells": safe_spearman(current_count, arrays["direction"].mean(1)), "teacher_direction_vs_union_cells": safe_spearman(union_count, arrays["direction"].mean(1)), "teacher_exit_vs_current_cells": safe_spearman(current_count, arrays["exit"].sum(1)), "teacher_exit_vs_union_cells": safe_spearman(union_count, arrays["exit"].sum(1)), "note": "History evidence is measured only from stored past frames already expressed in the current frame; no future or mesh points are added."}


def role_audit(arrays: dict[str, Any]) -> dict[str, Any]:
    role = arrays["role"]
    canonical = arrays["canonical_role"]
    role_n = role / np.maximum(np.linalg.norm(role, axis=1, keepdims=True), 1e-8)
    canonical_n = canonical / np.maximum(np.linalg.norm(canonical, axis=1, keepdims=True), 1e-8)
    return {
        "topological_role_teacher_shape": list(role.shape[1:]),
        "rotation_canonical_role_shape": list(canonical.shape[1:]),
        "canonical_role_field_available_in_loader": True,
        "directional_vs_canonical_cosine": desc(np.sum(role_n * canonical_n, axis=1)),
        "directional_role_must_not_be_reused_as_canonical": True,
        "note": "Both stored fields are audited separately; the existing training loader still uses only topological_role_teacher.",
    }


def make_pairs(datasets: dict[str, TopologicalSemanticDataset], out: Path, seed: int, max_pairs: int = 500) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = []
    for split, ds in datasets.items():
        arrays = load_arrays(ds)
        n = len(ds)
        teacher_signature = np.concatenate([arrays["direction"], arrays["distance"], arrays["area"], arrays["exit"], arrays["scores"]], axis=1)
        teacher_signature = teacher_signature / np.maximum(np.linalg.norm(teacher_signature, axis=1, keepdims=True), 1e-8)
        mask = arrays["current"][:, 0] >= 0.5
        flat = mask.reshape(n, -1).astype(np.float32)
        cells = flat.sum(1)
        sample_pairs = []
        candidates = []
        candidate_metrics = []
        for _ in range(min(n * 8, max_pairs * 30)):
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            inter = float(flat[i] @ flat[j])
            coverage_dist = float(1.0 - inter / max(float(cells[i] + cells[j] - inter), 1.0))
            teacher_dist = float(1.0 - teacher_signature[i] @ teacher_signature[j])
            candidates.append((i, j))
            candidate_metrics.append((coverage_dist, teacher_dist))
        if not candidate_metrics:
            continue
        coverage_values = np.asarray([row[0] for row in candidate_metrics])
        teacher_values = np.asarray([row[1] for row in candidate_metrics])
        coverage_low, coverage_high = np.percentile(coverage_values, [25, 75])
        teacher_low, teacher_high = np.percentile(teacher_values, [25, 75])
        for (i, j), (coverage_dist, teacher_dist) in zip(candidates, candidate_metrics):
            if coverage_dist >= coverage_high and teacher_dist <= teacher_low:
                sample_pairs.append({"pair_type": "same_topology_different_coverage", "i": int(i), "j": int(j), "coverage_distance": coverage_dist, "teacher_topology_distance": teacher_dist})
            elif coverage_dist <= coverage_low and teacher_dist >= teacher_high:
                sample_pairs.append({"pair_type": "hard_negative_coverage_similar_topology_different", "i": int(i), "j": int(j), "coverage_distance": coverage_dist, "teacher_topology_distance": teacher_dist})
            if len(sample_pairs) >= max_pairs:
                break
        for k, row in enumerate(sample_pairs):
            row.update({"split": split, "pair_id": f"{split}_{k:05d}", "sample_i": str(arrays["path"][row.pop("i")]), "sample_j": str(arrays["path"][row.pop("j")])})
            rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return {"count": len(rows), "by_type": dict(Counter(row["pair_type"] for row in rows)), "path": str(out / "pairs.jsonl"), "selection": "same split; current surface-mask IoU and independent teacher connection signature cosine-distance quartiles; deterministic seed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["dataset_root"])
    out = Path(cfg["output_root"]) / cfg["run_id"]
    for name in ["metrics", "diagnostics", "exit_label_previews", "history_evidence", "coverage_pairs", "contracts"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    datasets = {split: TopologicalSemanticDataset(root, split, cfg["data"]["input_channels"], cfg["data"]["score_names"]) for split in ["train", "val", "test"]}
    arrays = {split: load_arrays(ds) for split, ds in datasets.items()}
    baseline = {split: score_baselines(arrays[split], arrays["train"], cfg["data"]["score_names"]) for split in arrays}
    write_json(out / "metrics" / "simple_baselines.json", baseline)
    exits = {split: exit_audit(arrays[split], split) for split in arrays}
    write_json(out / "diagnostics" / "exit_label_audit.json", exits)
    # Temporal continuity is reported without imposing a learned threshold.
    temporal = {}
    for split, arr in arrays.items():
        transitions = []
        for trajectory in sorted(set(arr["trajectory"].tolist())):
            idx = np.where(arr["trajectory"] == trajectory)[0]
            idx = idx[np.argsort(arr["timestamp"][idx])]
            if len(idx) > 1:
                transitions.extend(np.abs(arr["exit"][idx[1:]] - arr["exit"][idx[:-1]]).mean(1).tolist())
        temporal[split] = {"adjacent_exit_bin_change": desc(transitions)}
    write_json(out / "diagnostics" / "exit_temporal_audit.json", temporal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg["checkpoint"])
    model_cfg = yaml.safe_load((Path(cfg["model_config"])).read_text(encoding="utf-8"))
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TopologicalSemanticNet(model_cfg, use_history=True).to(device)
    model.load_state_dict(checkpoint["model"])
    threshold_audit = exit_threshold_audit(model, datasets, model_cfg, device)
    write_json(out / "diagnostics" / "exit_threshold_audit.json", threshold_audit)
    small_fit = small_exit_fit(model, datasets["train"], model_cfg, device, int(cfg["seed"]))
    write_json(out / "diagnostics" / "exit_small_overfit.json", small_fit)
    write_json(out / "diagnostics" / "history_information_audit.json", {split: history_audit(arrays[split]) for split in arrays})
    write_json(out / "diagnostics" / "role_audit.json", {split: role_audit(arrays[split]) for split in arrays})
    pair_summary = make_pairs(datasets, out / "coverage_pairs", int(cfg["seed"]), int(cfg["max_pairs_per_split"]))
    pair_types = pair_summary["by_type"]
    if not small_fit["fit_success"]:
        conclusion = "TOPOLOGICAL_SUPERVISION_NOT_READY"
        gate_reason = "the fixed-feature exit head did not fit the restricted positive-sample audit"
    elif pair_types.get("same_topology_different_coverage", 0) < 10 or pair_types.get("hard_negative_coverage_similar_topology_different", 0) < 10:
        conclusion = "TOPOLOGICAL_SUPERVISION_NOT_READY"
        gate_reason = "coverage-invariance audit lacks both positive and hard-negative populations"
    else:
        conclusion = "TOPOLOGICAL_SUPERVISION_READY_WITH_LIMITATIONS"
        gate_reason = "basic supervision learnability and pair construction gates passed"
    contracts = {
        "exit_supervision_v2": {"version": "exit_supervision_v2", "representation": "circular_soft_exit_sectors", "source": "existing independent_exit_distribution and traversability/distance consistency audit", "recommended_use": "derive contiguous sectors from traversability and reachable distance, then represent each sector by center angle, angular width and normalized length; do not use 32 independent hard Bernoulli labels as the sole target", "circular_resolution_degrees": 360.0 / 32.0, "required_fields": ["exit_sector_center_sin_cos", "exit_sector_width", "exit_sector_length", "exit_sector_valid_mask"], "legacy_field": "independent_exit_distribution", "reason": "The audit must determine whether hard bins are stable; adjacent bins should be grouped into connected sectors and soft labels should span the measured sector width."},
        "directional_teacher_v2": {"version": "directional_teacher_v2", "frame": "robot_frame", "direction_zero": "robot_forward", "fields": ["traversable_direction_distribution", "reachable_distance_distribution", "reachable_area_distribution", "exit_sector_distribution"], "rotation_rule": "a robot yaw change circularly shifts the robot-frame direction bins; left/right information is retained", "online_constraint": "teacher only; student sees current and causal history surface evidence"},
        "canonical_role_teacher_v2": {"version": "canonical_role_teacher_v2", "purpose": "rotation-canonical local connection role for cross-location matching", "forbidden_inputs": ["world_absolute_heading", "world_id", "trajectory_id", "absolute_pose"], "recommended_construction": "canonicalize the local connectivity/exit direction field by its principal main-path axis or minimum circular-shift equivalence, and retain a separate unoriented multiset of exit count, widths, lengths, bottleneck and openness", "required_audit": ["same role different robot heading is close", "surface-similar topology-different is separated", "surface-different topology-similar is close"], "legacy_field": "rotation_canonical_role_teacher"},
        "topological_task_decomposition_v2": {"version": "topological_task_decomposition_v2", "static_tasks": ["direction", "distance", "exit_sector", "openness_score", "bottleneck_score", "canonical_role"], "ordered_dynamic_tasks": ["turn_strength", "transition_score", "new_branch_appearance", "topological_node_score"], "history_requirement": {"static": "causal accumulated evidence is sufficient; strict order is not required", "dynamic": "ordered causal history and relative poses are required"}, "invalid_target_policy": "use explicit masks; never convert unavailable labels to zero"},
    }
    for name, value in contracts.items():
        write_json(out / "contracts" / f"{name}.json", value)
    summary = {
        "conclusion": conclusion,
        "gate_reason": gate_reason,
        "status_policy": "provisional until audit metrics are reviewed; no full second model was trained",
        "dataset_root": str(root), "checkpoint": str(ckpt_path), "checkpoint_epoch": checkpoint.get("epoch"), "device": str(device),
        "simple_baselines": str(out / "metrics" / "simple_baselines.json"), "exit_audit": exits, "history": str(out / "diagnostics" / "history_information_audit.json"), "coverage_pairs": pair_summary,
        "limitations": ["legacy hard exit bins remain an audit finding until sector regeneration", "stored role fields require explicit directional/canonical separation", "history union adds evidence but existing model pooling cannot establish order dependence", "no formal second model training was performed"],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"output_dir": str(out), "conclusion": summary["conclusion"], "device": str(device), "pairs": pair_summary["count"]}, indent=2))


if __name__ == "__main__":
    main()
