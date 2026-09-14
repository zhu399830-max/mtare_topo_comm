from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class TopologicalSemanticDataset(Dataset):
    def __init__(self, root: Path, split: str, input_channels: list[int], score_names: list[str]) -> None:
        self.root = root
        self.split = split
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no npz samples under {root / split}")
        self.input_channels = input_channels
        self.score_names = score_names

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            current4 = data["student_input_current"].astype(np.float32)
            history4 = data["student_input_history"].astype(np.float32)
            current = current4[self.input_channels]
            history = history4[:, self.input_channels]
            scores = data["continuous_structural_scores"].astype(np.float32)
            # The contract stores extra diagnostic scores in JSON; the vector is
            # already ordered by continuous_structural_score_names.
            item = {
                "current": current,
                "history": history,
                "history_valid_mask": data["history_valid_mask"].astype(np.float32),
                "relative_poses": data["relative_poses"].astype(np.float32),
                "direction": data["traversable_direction_distribution"].astype(np.float32),
                "distance": data["reachable_distance_distribution"].astype(np.float32),
                "area": data["reachable_area_distribution"].astype(np.float32),
                "exit": data["independent_exit_distribution"].astype(np.float32),
                "scores": scores[: len(self.score_names)],
                "role": data["topological_role_teacher"].astype(np.float32),
                "local_connectivity": data["local_connectivity_data"].astype(np.float32),
                "path": str(path),
                "world": str(data["world"]),
                "trajectory": str(data["trajectory"]),
                "timestamp": int(data["timestamp"]),
                "raw_point_count": int(data["raw_point_count"]),
                "center_pose": data["center_pose"].astype(np.float32),
            }
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "current",
        "history",
        "history_valid_mask",
        "relative_poses",
        "direction",
        "distance",
        "area",
        "exit",
        "scores",
        "role",
        "local_connectivity",
        "center_pose",
    ]
    out = {key: torch.from_numpy(np.stack([sample[key] for sample in batch])) for key in tensor_keys}
    for key in ["path", "world", "trajectory", "timestamp", "raw_point_count"]:
        out[key] = [sample[key] for sample in batch]
    return out


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 5, stride=2, padding=2),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(base * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class TopologicalSemanticNet(nn.Module):
    def __init__(self, cfg: dict[str, Any], use_history: bool) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self.use_history = use_history
        self.frame_dim = int(model_cfg["frame_embedding_dim"])
        self.encoder = FrameEncoder(len(data_cfg["input_channels"]), int(model_cfg["base_channels"]), self.frame_dim)
        if use_history:
            pose_dim = int(model_cfg["pose_embedding_dim"])
            self.pose_mlp = nn.Sequential(nn.Linear(5, pose_dim), nn.SiLU(), nn.Linear(pose_dim, pose_dim), nn.SiLU())
            self.fuse = nn.Sequential(
                nn.Linear(self.frame_dim * 2 + pose_dim, int(model_cfg["fused_dim"])),
                nn.SiLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(int(model_cfg["fused_dim"]), self.frame_dim),
                nn.SiLU(),
            )
        else:
            self.pose_mlp = None
            self.fuse = nn.Sequential(
                nn.Linear(self.frame_dim, self.frame_dim),
                nn.SiLU(),
                nn.Dropout(float(model_cfg["dropout"])),
                nn.Linear(self.frame_dim, self.frame_dim),
                nn.SiLU(),
            )
        self.semantic = nn.Sequential(nn.Linear(self.frame_dim, 128), nn.SiLU(), nn.Linear(128, int(model_cfg["semantic_dim"])))
        d = int(data_cfg["direction_count"])
        s = len(data_cfg["score_names"])
        r = int(data_cfg["role_dim"])
        self.direction_head = nn.Linear(self.frame_dim, d)
        self.distance_head = nn.Linear(self.frame_dim, d)
        self.area_head = nn.Linear(self.frame_dim, d)
        self.exit_head = nn.Linear(self.frame_dim, d)
        self.score_head = nn.Linear(self.frame_dim, s)
        self.role_head = nn.Linear(self.frame_dim, r)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        history_valid_mask: torch.Tensor,
        relative_poses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        current_z = self.encoder(current)
        if self.use_history:
            b, t, c, h, w = history.shape
            hist_z = self.encoder(history.reshape(b * t, c, h, w)).reshape(b, t, -1)
            pose_z = self.pose_mlp(relative_poses)
            mask = history_valid_mask.unsqueeze(-1)
            hist_joint = torch.cat([hist_z, pose_z], dim=-1) * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            hist_pool = hist_joint.sum(dim=1) / denom
            fused = self.fuse(torch.cat([current_z, hist_pool], dim=1))
        else:
            fused = self.fuse(current_z)
        semantic = F.normalize(self.semantic(fused), dim=1)
        role = F.normalize(self.role_head(fused), dim=1)
        return {
            "direction_logits": self.direction_head(fused),
            "distance": torch.sigmoid(self.distance_head(fused)),
            "area": torch.sigmoid(self.area_head(fused)),
            "exit_logits": self.exit_head(fused),
            "scores": torch.sigmoid(self.score_head(fused)),
            "role": role,
            "semantic": semantic,
        }


def target_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ["current", "history", "history_valid_mask", "relative_poses", "direction", "distance", "area", "exit", "scores", "role"]
    }


def compute_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    w = cfg["loss"]
    direction = F.binary_cross_entropy_with_logits(pred["direction_logits"], batch["direction"])
    exit_loss = F.binary_cross_entropy_with_logits(pred["exit_logits"], batch["exit"])
    valid_dir = (batch["direction"] > 0.5) | (batch["distance"] > 0.01)
    if valid_dir.any():
        distance = F.smooth_l1_loss(pred["distance"][valid_dir], batch["distance"][valid_dir])
    else:
        distance = pred["distance"].sum() * 0.0
    area = F.smooth_l1_loss(pred["area"], batch["area"])
    scores = F.smooth_l1_loss(pred["scores"], batch["scores"])
    teacher_role = F.normalize(batch["role"], dim=1)
    role_cos = (1.0 - F.cosine_similarity(pred["role"], teacher_role, dim=1)).mean()
    role_mse = F.mse_loss(pred["role"], teacher_role)
    total = (
        float(w["direction_bce"]) * direction
        + float(w["distance"]) * distance
        + float(w["area"]) * area
        + float(w["exit_bce"]) * exit_loss
        + float(w["scores"]) * scores
        + float(w["role_cosine"]) * role_cos
        + float(w["role_mse"]) * role_mse
    )
    parts = {
        "total": float(total.detach()),
        "direction": float(direction.detach()),
        "distance": float(distance.detach()),
        "area": float(area.detach()),
        "exit": float(exit_loss.detach()),
        "scores": float(scores.detach()),
        "role_cosine": float(role_cos.detach()),
        "role_mse": float(role_mse.detach()),
    }
    return total, parts


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, cfg: dict[str, Any]) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for raw in loader:
        batch = target_tensors(raw, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        loss, parts = compute_loss(pred, batch, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        optimizer.step()
        rows.append(parts)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = float((pred & truth).sum().item())
    fp = float((pred & ~truth).sum().item())
    fn = float((~pred & truth).sum().item())
    return {
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "false_rate": fp / max(float(pred.sum().item()), 1.0),
        "missed_rate": fn / max(float(truth.sum().item()), 1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(spearmanr(a, b).correlation)


@torch.no_grad()
def collect_predictions(model: nn.Module, dataset: Dataset, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), collate_fn=collate)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, list[Any]] = defaultdict(list)
    losses: list[dict[str, float]] = []
    for raw in loader:
        batch = target_tensors(raw, device)
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        _, parts = compute_loss(pred, batch, cfg)
        losses.append(parts)
        arrays["direction_prob"].append(torch.sigmoid(pred["direction_logits"]).cpu().numpy())
        arrays["distance_pred"].append(pred["distance"].cpu().numpy())
        arrays["area_pred"].append(pred["area"].cpu().numpy())
        arrays["exit_prob"].append(torch.sigmoid(pred["exit_logits"]).cpu().numpy())
        arrays["scores_pred"].append(pred["scores"].cpu().numpy())
        arrays["role_pred"].append(pred["role"].cpu().numpy())
        arrays["semantic"].append(pred["semantic"].cpu().numpy())
        for key in ["current", "direction", "distance", "area", "exit", "scores", "role"]:
            arrays[key].append(batch[key].cpu().numpy())
        for key in ["world", "trajectory", "path", "raw_point_count"]:
            meta[key].extend(raw[key])
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    merged["meta"] = dict(meta)
    merged["loss"] = {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}
    return merged


def retrieval_metrics(pred_role: np.ndarray, teacher_role: np.ndarray, rng: np.random.Generator, max_samples: int) -> dict[str, Any]:
    n = len(pred_role)
    if n > max_samples:
        idx = np.sort(rng.choice(n, max_samples, replace=False))
        pred_role = pred_role[idx]
        teacher_role = teacher_role[idx]
        n = len(idx)
    pred = pred_role / np.maximum(np.linalg.norm(pred_role, axis=1, keepdims=True), 1e-8)
    teacher = teacher_role / np.maximum(np.linalg.norm(teacher_role, axis=1, keepdims=True), 1e-8)
    pred_dist = 1.0 - pred @ pred.T
    teacher_dist = 1.0 - teacher @ teacher.T
    np.fill_diagonal(pred_dist, np.inf)
    np.fill_diagonal(teacher_dist, np.inf)
    target_nn = np.argmin(teacher_dist, axis=1)
    order = np.argsort(pred_dist, axis=1)
    metrics = {"count": int(n), "random_top1": float(1.0 / max(n - 1, 1)), "random_top5": float(min(5, n - 1) / max(n - 1, 1)), "random_top10": float(min(10, n - 1) / max(n - 1, 1))}
    for k in [1, 5, 10]:
        kk = min(k, max(n - 1, 1))
        metrics[f"top{k}"] = float(np.mean([target_nn[i] in order[i, :kk] for i in range(n)]))
    upper = np.triu_indices(n, 1)
    metrics["pred_vs_teacher_distance_spearman"] = spearman(pred_dist[upper], teacher_dist[upper])
    metrics["role_cosine_distance"] = describe(np.sum(pred * teacher, axis=1) * -1.0 + 1.0)
    return metrics


def evaluate_prediction(preds: dict[str, Any], cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    dthr = float(cfg["eval"]["direction_threshold"])
    ethr = float(cfg["eval"]["exit_threshold"])
    direction = binary_metrics(torch.from_numpy(np.log(preds["direction_prob"] / np.maximum(1.0 - preds["direction_prob"], 1e-6))), torch.from_numpy(preds["direction"]), dthr)
    exits = binary_metrics(torch.from_numpy(np.log(preds["exit_prob"] / np.maximum(1.0 - preds["exit_prob"], 1e-6))), torch.from_numpy(preds["exit"]), ethr)
    valid = (preds["direction"] > 0.5) | (preds["distance"] > 0.01)
    dist_abs = np.abs(preds["distance_pred"] - preds["distance"])
    score_names = cfg["data"]["score_names"]
    score_metrics = {}
    for i, name in enumerate(score_names):
        err = np.abs(preds["scores_pred"][:, i] - preds["scores"][:, i])
        score_metrics[name] = {"mae": float(np.mean(err)), "spearman": spearman(preds["scores_pred"][:, i], preds["scores"][:, i])}
    return {
        "loss": preds["loss"],
        "direction": direction,
        "exit": {
            **exits,
            "predicted_exit_count_error_mean": float(np.mean(np.abs((preds["exit_prob"] >= ethr).sum(1) - (preds["exit"] >= 0.5).sum(1)))),
            "false_branch_rate": exits["false_rate"],
            "missed_branch_rate": exits["missed_rate"],
        },
        "distance": {
            "mae_valid": float(dist_abs[valid].mean()) if valid.any() else float("nan"),
            "median_abs_valid": float(np.median(dist_abs[valid])) if valid.any() else float("nan"),
            "near_mae": float(dist_abs[(preds["distance"] > 0.0) & (preds["distance"] < 0.35)].mean()) if np.any((preds["distance"] > 0.0) & (preds["distance"] < 0.35)) else float("nan"),
            "far_mae": float(dist_abs[preds["distance"] >= 0.65].mean()) if np.any(preds["distance"] >= 0.65) else float("nan"),
        },
        "scores": score_metrics,
        "role": retrieval_metrics(preds["role_pred"], preds["role"], rng, int(cfg["eval"]["retrieval_max_samples"])),
    }


def audit_dataset(dataset: TopologicalSemanticDataset, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    counts = Counter()
    nan_count = 0
    hist_len = []
    surface_cells = []
    reach_dirs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        counts[sample["world"]] += 1
        hist_len.append(float(sample["history_valid_mask"].sum()))
        surface_cells.append(float(sample["current"][0].sum()))
        reach_dirs.append(float((sample["direction"] >= 0.5).sum()))
        for key in ["current", "history", "history_valid_mask", "relative_poses", "direction", "distance", "area", "exit", "scores", "role"]:
            arr = np.asarray(sample[key])
            if not np.isfinite(arr).all():
                nan_count += 1
                break
    audit = {
        "count": len(dataset),
        "world_counts": dict(counts),
        "nan_or_inf_samples": nan_count,
        "history_valid_length": describe(hist_len),
        "surface_cells": describe(surface_cells),
        "reachable_direction_count": describe(reach_dirs),
        "current_shape": list(dataset[0]["current"].shape),
        "history_shape": list(dataset[0]["history"].shape),
        "relative_pose_shape": list(dataset[0]["relative_poses"].shape),
        "teacher_direction_0_convention": "robot-frame forward direction, inherited from v3 dataset contract",
    }
    write_json(out, audit)
    return audit


def save_previews(dataset: TopologicalSemanticDataset, cfg: dict[str, Any], out_dir: Path, rng: np.random.Generator) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(cfg["data"]["preview_count"]), len(dataset))
    indices = np.sort(rng.choice(len(dataset), count, replace=False))
    for slot, idx in enumerate(indices):
        s = dataset[int(idx)]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes[0, 0].imshow(s["current"][0], origin="lower", cmap="gray")
        axes[0, 0].set_title("current surface")
        hist_mask = s["history_valid_mask"][:, None, None, None]
        hist = (s["history"][:, 0:1] * hist_mask).sum(0)[0] / max(float(hist_mask.sum()), 1.0)
        axes[0, 1].imshow(hist, origin="lower", cmap="magma")
        axes[0, 1].set_title("causal history surface")
        axes[0, 2].imshow(s["local_connectivity"], origin="lower", cmap="Greens")
        axes[0, 2].set_title("teacher traversable")
        angles = np.linspace(0, 2 * np.pi, len(s["direction"]), endpoint=False)
        ax = plt.subplot(2, 3, 4, projection="polar")
        ax.bar(angles, s["direction"], width=2 * np.pi / len(angles))
        ax.set_title("reachable dirs")
        ax = plt.subplot(2, 3, 5, projection="polar")
        ax.bar(angles, s["distance"], width=2 * np.pi / len(angles))
        ax.set_title("distance")
        axes[1, 2].bar(np.arange(len(s["scores"])), s["scores"])
        axes[1, 2].set_ylim(0, 1.05)
        axes[1, 2].set_title("structural scores")
        for ax2 in axes.flatten()[:3].tolist() + [axes[1, 2]]:
            ax2.set_xticks([])
            ax2.set_yticks([])
        fig.suptitle(f"{s['world']} {Path(s['path']).name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"preview_{slot:03d}.png", dpi=140)
        plt.close(fig)


def train_model(name: str, use_history: bool, train_ds: Dataset, val_ds: Dataset, cfg: dict[str, Any], out_dir: Path, device: torch.device) -> dict[str, Any]:
    model = TopologicalSemanticNet(cfg, use_history=use_history).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]), collate_fn=collate, drop_last=False)
    best = {"epoch": -1, "val_total": float("inf")}
    log = []
    patience = int(cfg["train"]["patience"])
    stale = 0
    ckpt_dir = out_dir / f"{name}_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        val_preds = collect_predictions(model, val_ds, device, cfg)
        val_loss = val_preds["loss"]
        row = {"epoch": epoch, "train": train_loss, "val": val_loss}
        log.append(row)
        if val_loss["total"] < best["val_total"]:
            best = {"epoch": epoch, "val_total": float(val_loss["total"])}
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "use_history": use_history}, ckpt_dir / "best.pt")
            stale = 0
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "use_history": use_history}, ckpt_dir / "last.pt")
        if stale >= patience:
            break
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    write_json(out_dir / "training_logs" / f"{name}.json", {"best": best, "epochs": log})
    return {"model": model, "best": best, "param_count": sum(p.numel() for p in model.parameters())}


def overfit_check(train_ds: Dataset, cfg: dict[str, Any], out_dir: Path, device: torch.device) -> dict[str, Any]:
    subset = Subset(train_ds, list(range(min(int(cfg["train"]["overfit_samples"]), len(train_ds)))))
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate)
    batch_raw = next(iter(loader))
    model = TopologicalSemanticNet(cfg, use_history=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=0.0)
    first_param = next(model.parameters()).detach().clone()
    records = []
    for epoch in range(int(cfg["train"]["overfit_epochs"]) + 1):
        batch = target_tensors(batch_raw, device)
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
                _, parts = compute_loss(pred, batch, cfg)
            records.append({"epoch": epoch, **parts})
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        loss, parts = compute_loss(pred, batch, cfg)
        loss.backward()
        optimizer.step()
        if epoch in {1, 5, 10, 20, 40, int(cfg["train"]["overfit_epochs"])}:
            records.append({"epoch": epoch, **parts})
    param_delta = float((next(model.parameters()).detach() - first_param).abs().max().cpu())
    result = {"sample_count": len(subset), "records": records, "parameter_max_delta": param_delta, "success": records[-1]["total"] < 0.65 * records[0]["total"] and param_delta > 0.0}
    write_json(out_dir / "training_logs" / "overfit.json", result)
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": int(cfg["train"]["overfit_epochs"]), "use_history": True}, out_dir / "history_model_checkpoint" / "overfit.pt")
    return result


@torch.no_grad()
def history_dependency(model: nn.Module, dataset: Dataset, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate)
    rows: dict[str, list[float]] = defaultdict(list)
    model.eval()
    for i, raw in enumerate(loader):
        if i >= int(cfg["eval"]["dependency_batches"]):
            break
        batch = target_tensors(raw, device)
        base = model(batch["current"], batch["history"], batch["history_valid_mask"], batch["relative_poses"])
        variants = {
            "zero_history": (torch.zeros_like(batch["history"]), torch.zeros_like(batch["history_valid_mask"]), batch["relative_poses"]),
            "shuffle_history": (batch["history"].flip(1), batch["history_valid_mask"].flip(1), batch["relative_poses"].flip(1)),
            "zero_relative_pose": (batch["history"], batch["history_valid_mask"], torch.zeros_like(batch["relative_poses"])),
            "current_only_mask": (batch["history"], torch.zeros_like(batch["history_valid_mask"]), batch["relative_poses"]),
        }
        for name, (hist, mask, poses) in variants.items():
            pred = model(batch["current"], hist, mask, poses)
            drift = 1.0 - F.cosine_similarity(base["semantic"], pred["semantic"], dim=1)
            rows[name].extend(drift.detach().cpu().numpy().tolist())
    return {name: describe(values) for name, values in rows.items()}


def counterexample_audit(preds: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    n = min(len(preds["role"]), int(cfg["eval"]["pair_audit_max_samples"]))
    pred_role = preds["role_pred"][:n]
    teacher_role = preds["role"][:n]
    surface = preds["current"][:n, 0].reshape(n, -1)
    pred_role = pred_role / np.maximum(np.linalg.norm(pred_role, axis=1, keepdims=True), 1e-8)
    teacher_role = teacher_role / np.maximum(np.linalg.norm(teacher_role, axis=1, keepdims=True), 1e-8)
    surf_inter = surface @ surface.T
    surf_sum = surface.sum(1)
    surf_iou = surf_inter / np.maximum(surf_sum[:, None] + surf_sum[None, :] - surf_inter, 1.0)
    surface_dist = 1.0 - surf_iou
    topo_dist = 1.0 - teacher_role @ teacher_role.T
    pred_dist = 1.0 - pred_role @ pred_role.T
    upper = np.triu_indices(n, 1)
    sd = surface_dist[upper]
    td = topo_dist[upper]
    pd = pred_dist[upper]
    a = (sd >= np.percentile(sd, 75)) & (td <= np.percentile(td, 25))
    b = (sd <= np.percentile(sd, 25)) & (td >= np.percentile(td, 75))
    return {
        "surface_different_topology_similar": {"pairs": int(a.sum()), "predicted_role_distance": describe(pd[a])},
        "surface_similar_topology_different": {"pairs": int(b.sum()), "predicted_role_distance": describe(pd[b])},
        "success_rate_separation": float(np.mean(pd[b][: min(a.sum(), b.sum())] > pd[a][: min(a.sum(), b.sum())])) if min(a.sum(), b.sum()) > 0 else float("nan"),
    }


def nuisance_probe(preds: dict[str, Any]) -> dict[str, Any]:
    role = preds["role_pred"]
    role = role / np.maximum(np.linalg.norm(role, axis=1, keepdims=True), 1e-8)
    n = len(role)
    pair_n = min(n, 512)
    role = role[:pair_n]
    teacher = preds["role"][:pair_n]
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-8)
    surface = preds["current"][:pair_n, 0].reshape(pair_n, -1)
    cells = surface.sum(1)
    raw = np.asarray(preds["meta"]["raw_point_count"][:pair_n], dtype=np.float64)
    worlds = np.asarray(preds["meta"]["world"][:pair_n])
    upper = np.triu_indices(pair_n, 1)
    role_d = (1.0 - role @ role.T)[upper]
    teacher_d = (1.0 - teacher @ teacher.T)[upper]
    cell_d = np.abs(cells[:, None] - cells[None, :])[upper]
    raw_d = np.abs(raw[:, None] - raw[None, :])[upper]
    world_same = (worlds[:, None] == worlds[None, :])[upper].astype(np.float64)
    return {
        "role_vs_teacher_connection_spearman": spearman(role_d, teacher_d),
        "role_vs_surface_cell_count_spearman": spearman(role_d, cell_d),
        "role_vs_raw_point_count_spearman": spearman(role_d, raw_d),
        "role_vs_same_world_spearman": spearman(role_d, world_same),
        "world_counts": dict(Counter(preds["meta"]["world"])),
    }


def subset_by_world(dataset: TopologicalSemanticDataset, world: str) -> Subset:
    indices = [i for i in range(len(dataset)) if dataset[i]["world"] == world]
    return Subset(dataset, indices)


def confidence_split(preds: dict[str, Any], cfg: dict[str, Any]) -> dict[str, list[int]]:
    hc = cfg["data"]["high_confidence"]
    reach_count = (preds["direction"] >= 0.5).sum(1)
    surface_cells = preds["current"][:, 0].reshape(len(preds["current"]), -1).sum(1)
    high = (reach_count >= int(hc["min_reachable_directions"])) & (reach_count <= int(hc["max_reachable_directions"])) & (surface_cells >= float(hc["min_surface_cells"]))
    return {"high": np.where(high)[0].tolist(), "low_or_boundary": np.where(~high)[0].tolist()}


def select_rows(preds: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    out = {}
    for key, value in preds.items():
        if key == "meta":
            out[key] = {mk: [mv[i] for i in indices] for mk, mv in value.items()}
        elif key == "loss":
            out[key] = value
        else:
            out[key] = value[indices]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_all(int(cfg["seed"]), bool(cfg["deterministic"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    dataset_root = Path(cfg["dataset_root"])
    out_dir = Path(cfg["output_root"]) / str(cfg["run_id"])
    for rel in [
        "config",
        "data_audit",
        "current_only_checkpoint",
        "history_model_checkpoint",
        "training_logs",
        "validation_metrics",
        "campus_metrics",
        "indoor_metrics",
        "direction_metrics",
        "exit_metrics",
        "structural_score_metrics",
        "role_retrieval_metrics",
        "history_dependency_audit",
        "counterexample_metrics",
        "nuisance_probe",
        "previews",
    ]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config" / args.config.name)
    write_json(out_dir / "config" / "provenance.json", {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_status": command_output(["git", "status", "--short"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "python": command_output(["python3", "--version"]),
        "package_list": command_output(["python3", "-m", "pip", "freeze"]),
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TopologicalSemanticDataset(dataset_root, cfg["splits"]["train"], cfg["data"]["input_channels"], cfg["data"]["score_names"])
    val_ds = TopologicalSemanticDataset(dataset_root, cfg["splits"]["val"], cfg["data"]["input_channels"], cfg["data"]["score_names"])
    test_ds = TopologicalSemanticDataset(dataset_root, cfg["splits"]["test"], cfg["data"]["input_channels"], cfg["data"]["score_names"])
    audits = {
        "train": audit_dataset(train_ds, cfg, out_dir / "data_audit" / "train.json"),
        "val": audit_dataset(val_ds, cfg, out_dir / "data_audit" / "val.json"),
        "test": audit_dataset(test_ds, cfg, out_dir / "data_audit" / "test.json"),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    write_json(out_dir / "data_audit" / "summary.json", audits)
    save_previews(train_ds, cfg, out_dir / "previews" / "train_audit", rng)
    overfit = overfit_check(train_ds, cfg, out_dir, device)
    current = train_model("current_only", False, train_ds, val_ds, cfg, out_dir, device)
    history = train_model("history_model", True, train_ds, val_ds, cfg, out_dir, device)
    models = {"current_only": current["model"], "history_model": history["model"]}
    all_metrics: dict[str, Any] = {
        "overfit": overfit,
        "param_count": {"current_only": current["param_count"], "history_model": history["param_count"]},
        "best_checkpoint": {"current_only": current["best"], "history_model": history["best"]},
    }
    for model_name, model in models.items():
        val_preds = collect_predictions(model, val_ds, device, cfg)
        val_metrics = evaluate_prediction(val_preds, cfg, rng)
        val_metrics["nuisance_probe"] = nuisance_probe(val_preds)
        val_metrics["counterexamples"] = counterexample_audit(val_preds, cfg)
        write_json(out_dir / "validation_metrics" / f"{model_name}_forest.json", val_metrics)
        write_json(out_dir / "direction_metrics" / f"{model_name}_forest.json", val_metrics["direction"])
        write_json(out_dir / "exit_metrics" / f"{model_name}_forest.json", val_metrics["exit"])
        write_json(out_dir / "structural_score_metrics" / f"{model_name}_forest.json", val_metrics["scores"])
        write_json(out_dir / "role_retrieval_metrics" / f"{model_name}_forest.json", val_metrics["role"])
        if model_name == "history_model":
            write_json(out_dir / "history_dependency_audit" / "forest.json", history_dependency(model, val_ds, device, cfg))
        all_metrics[f"{model_name}_forest"] = val_metrics
    # Test worlds are evaluated only after both checkpoints are fixed.
    for world in cfg["splits"]["test_worlds"]:
        world_ds = subset_by_world(test_ds, world)
        for model_name, model in models.items():
            preds = collect_predictions(model, world_ds, device, cfg)
            metrics = evaluate_prediction(preds, cfg, rng)
            groups = confidence_split(preds, cfg)
            metrics["teacher_confidence_groups"] = {
                name: evaluate_prediction(select_rows(preds, idx), cfg, rng) if idx else {"count": 0}
                for name, idx in groups.items()
            }
            metrics["nuisance_probe"] = nuisance_probe(preds)
            metrics["counterexamples"] = counterexample_audit(preds, cfg)
            write_json(out_dir / f"{world}_metrics" / f"{model_name}.json", metrics)
            if model_name == "history_model":
                write_json(out_dir / "history_dependency_audit" / f"{world}.json", history_dependency(model, world_ds, device, cfg))
            all_metrics[f"{model_name}_{world}"] = metrics
    hist_better = all_metrics["history_model_forest"]["direction"]["f1"] >= all_metrics["current_only_forest"]["direction"]["f1"]
    overfit_ok = bool(overfit["success"])
    forest_ok = all_metrics["history_model_forest"]["direction"]["f1"] > 0.55 and all_metrics["history_model_forest"]["exit"]["f1"] > 0.20
    test_ok = all(all_metrics[f"history_model_{w}"]["direction"]["f1"] > 0.45 for w in cfg["splits"]["test_worlds"])
    if overfit_ok and forest_ok and test_ok and hist_better:
        conclusion = "TOPOLOGICAL_SEMANTIC_MODEL_PASS"
    elif overfit_ok and forest_ok:
        conclusion = "TOPOLOGICAL_SEMANTIC_MODEL_MIXED"
    else:
        conclusion = "TOPOLOGICAL_SEMANTIC_MODEL_FAIL"
    summary = {
        "conclusion": conclusion,
        "output_dir": str(out_dir),
        "dataset_root": str(dataset_root),
        "data_audit": audits,
        "metrics": all_metrics,
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"conclusion": conclusion, "output_dir": str(out_dir), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
