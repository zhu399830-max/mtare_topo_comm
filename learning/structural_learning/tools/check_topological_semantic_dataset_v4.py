from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from learning.structural_learning.dataset import TopologicalSemanticDatasetV4, collate_topological_semantic_v4_torch
from learning.structural_learning.topological_metrics import direction_baselines, direction_metric_report
from learning.structural_learning.topological_supervision import canonical_role_descriptor
from learning.structural_learning.tools.train_topological_semantic_model import TopologicalSemanticNet


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


class TinySmokeHead(nn.Module):
    def __init__(self, role_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(3 * 10 * 10, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU())
        self.exit = nn.Linear(64, 32)
        self.count = nn.Linear(64, 1)
        self.role = nn.Linear(64, role_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = torch.nn.functional.adaptive_avg_pool2d(x[:, [0, 2, 3]], (10, 10)).flatten(1)
        z = self.body(pooled)
        return {"exit": self.exit(z), "count": self.count(z).squeeze(1), "role": self.role(z)}


def smoke_fit(dataset: TopologicalSemanticDatasetV4, device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    selected = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if int(sample["exit_count"]) > 0:
            selected.append(index)
        if len(selected) >= 64:
            break
    loader = DataLoader(dataset, batch_size=len(selected), sampler=selected, collate_fn=collate_topological_semantic_v4_torch)
    batch = next(iter(loader))
    model = TinySmokeHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    records = []
    for epoch in range(201):
        pred = model(batch["student_input_current"].to(device))
        exit_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["exit"], batch["exit_sector_soft"].to(device))
        count_loss = torch.nn.functional.smooth_l1_loss(pred["count"], batch["exit_count"].float().to(device))
        role_target = batch["canonical_topological_role"].to(device)
        role_target = role_target / role_target.norm(dim=1, keepdim=True).clamp_min(1e-8)
        role_pred = torch.nn.functional.normalize(pred["role"], dim=1)
        role_loss = (1.0 - torch.nn.functional.cosine_similarity(role_pred, role_target, dim=1)).mean()
        loss = exit_loss + 0.25 * count_loss + 0.5 * role_loss
        if epoch == 0 or epoch == 200:
            records.append({"epoch": epoch, "loss": float(loss.detach()), "exit_loss": float(exit_loss.detach()), "count_loss": float(count_loss.detach()), "role_loss": float(role_loss.detach())})
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pred = model(batch["student_input_current"].to(device))
        exit_prob = torch.sigmoid(pred["exit"]).cpu().numpy()
        count_pred = pred["count"].cpu().numpy()
        role_pred = torch.nn.functional.normalize(pred["role"], dim=1).cpu().numpy()
    exit_true = batch["exit_sector_binary"].numpy()
    exit_soft = batch["exit_sector_soft"].numpy()
    threshold_candidates = np.linspace(0.05, 0.95, 19)
    f1 = [direction_metric_report(exit_true, exit_prob, threshold=float(t))["positive_class_f1"] for t in threshold_candidates]
    role_true = batch["canonical_topological_role"].numpy()
    role_true /= np.maximum(np.linalg.norm(role_true, axis=1, keepdims=True), 1e-8)
    return {
        "selected_count": len(selected), "records": records,
        "exit_soft_pr_auc": direction_metric_report(exit_soft, exit_prob)["pr_auc"],
        "exit_binary_best_uniform_threshold": {"threshold": float(threshold_candidates[int(np.argmax(f1))]), "f1": float(max(f1))},
        "exit_count_mae": float(np.mean(np.abs(count_pred - batch["exit_count"].numpy()))),
        "canonical_role_cosine_distance": float(np.mean(1.0 - np.sum(role_pred * role_true, axis=1))),
        "fit_success": bool(records[-1]["loss"] < 0.6 * records[0]["loss"] and max(f1) > 0.7),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    args = parser.parse_args()
    datasets = {split: TopologicalSemanticDatasetV4(args.dataset, split) for split in ["train", "val", "test"]}
    batch_checks = {}
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False, collate_fn=collate_topological_semantic_v4_torch)
        batch = next(iter(loader))
        finite = all(torch.isfinite(value).all().item() for value in batch.values() if torch.is_tensor(value) and value.dtype.is_floating_point)
        dt = batch["history_time_deltas"].numpy()
        valid = batch["history_valid_mask"].numpy() > 0.5
        order_ok = bool(np.all(np.diff(dt, axis=1)[valid[:, 1:]] > 0))
        no_future = bool(np.all(dt[valid] < 0))
        relative_alignment = list(batch["relative_poses"].shape[1:]) == [8, 5]
        batch_checks[split] = {"samples": len(dataset), "current_shape": list(batch["student_input_current"].shape), "history_shape": list(batch["student_input_history"].shape), "history_time_deltas_shape": list(batch["history_time_deltas"].shape), "exit_shape": list(batch["exit_sector_soft"].shape), "role_shape": list(batch["canonical_topological_role"].shape), "finite": bool(finite), "history_order_old_to_new": order_ok, "history_no_future": no_future, "relative_pose_alignment": relative_alignment}
    train_labels = np.stack([datasets["train"][i]["direction_traversable"] for i in range(len(datasets["train"]))])
    direction_metrics = {}
    for split, dataset in datasets.items():
        labels = np.stack([dataset[i]["direction_traversable"] for i in range(len(dataset))])
        direction_metrics[split] = direction_baselines(labels, train_labels)
    smoke = smoke_fit(datasets["train"], torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.seed)
    rotation_role = []
    temporal_iou = []
    temporal_count_delta = []
    for index in range(min(64, len(datasets["train"]))):
        sample = datasets["train"][index]
        shifted = canonical_role_descriptor(
            np.roll(sample["exit_sector_soft"], 5),
            np.roll(sample["direction_reachable_distance"], 5),
            sample["exit_widths"], sample["exit_lengths"], sample["exit_valid_mask"], 64,
        )
        role = sample["canonical_topological_role"] / max(float(np.linalg.norm(sample["canonical_topological_role"])), 1e-8)
        rotation_role.append(float(1.0 - role @ shifted))
    for split, dataset in datasets.items():
        groups = {}
        for index in range(len(dataset)):
            sample = dataset[index]
            groups.setdefault(str(sample["trajectory"]), []).append((int(sample["timestamp"]), sample))
        for rows in groups.values():
            rows.sort(key=lambda row: row[0])
            for (_, previous), (_, current) in zip(rows, rows[1:]):
                a = previous["exit_sector_binary"] > 0.5
                b = current["exit_sector_binary"] > 0.5
                temporal_iou.append(float((a & b).sum() / max((a | b).sum(), 1)))
                temporal_count_delta.append(abs(int(current["exit_count"]) - int(previous["exit_count"])))
    audits = {"canonical_rotation_cosine_distance": {"mean": float(np.mean(rotation_role)), "p90": float(np.percentile(rotation_role, 90))}, "adjacent_exit_iou": {"mean": float(np.mean(temporal_iou)), "median": float(np.median(temporal_iou)), "p10": float(np.percentile(temporal_iou, 10))}, "adjacent_exit_count_delta": {"mean": float(np.mean(temporal_count_delta)), "p90": float(np.percentile(temporal_count_delta, 90))}}
    corrected_model_metrics = {}
    if args.checkpoint and args.model_config and args.checkpoint.exists():
        cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
        checkpoint = torch.load(args.checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TopologicalSemanticNet(cfg, use_history=True).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            for split, dataset in datasets.items():
                loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_topological_semantic_v4_torch)
                probabilities, labels = [], []
                for batch in loader:
                    current = batch["student_input_current"][:, [0, 2, 3]].to(device)
                    history = batch["student_input_history"][:, :, [0, 2, 3]].to(device)
                    pred = model(current, history, batch["history_valid_mask"].to(device), batch["relative_poses"].to(device))
                    probabilities.append(torch.sigmoid(pred["direction_logits"]).cpu().numpy())
                    labels.append(batch["direction_traversable"].numpy())
                corrected_model_metrics[split] = direction_metric_report(np.concatenate(labels), np.concatenate(probabilities))
    result = {"batch_checks": batch_checks, "direction_metrics": direction_metrics, "corrected_old_model_direction_metrics": corrected_model_metrics, "smoke": smoke, "temporal_rotation_audit": audits, "conclusion": "DATASET_READABLE" if all(row["finite"] and row["history_order_old_to_new"] and row["history_no_future"] and row["relative_pose_alignment"] for row in batch_checks.values()) else "DATASET_INVALID"}
    write_json(args.output / "diagnostics" / "v4_read_and_smoke_check.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
