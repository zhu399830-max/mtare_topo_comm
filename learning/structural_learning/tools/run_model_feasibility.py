from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from learning.structural_learning.dataset import StructuralSurfaceDataset
from learning.structural_learning.feasibility_model import TinySurfaceCompletionNet, completion_loss


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate(samples):
    return {
        "input_surface": torch.from_numpy(np.stack([s["input_surface"] for s in samples])),
        "teacher_surface": torch.from_numpy(np.stack([s["teacher_surface"] for s in samples])),
        "path": [s["path"] for s in samples],
    }


def parameter_checksum(model: torch.nn.Module) -> float:
    return float(sum(parameter.detach().double().sum().cpu() for parameter in model.parameters()))


def surface_counts(prediction: torch.Tensor, teacher: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = prediction[:, 0] >= threshold
    target = teacher[:, 0] >= 0.5
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def finish_metrics(counts: dict[str, float]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "surface_iou": tp / max(tp + fp + fn, 1.0),
        "surface_dice": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "surface_precision": tp / max(tp + fp, 1.0),
        "surface_recall": tp / max(tp + fn, 1.0),
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold, positive_weight) -> dict[str, object]:
    model.eval()
    losses, model_counts, baseline_counts = [], {"tp": 0.0, "fp": 0.0, "fn": 0.0}, {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    attr_abs, attr_n = 0.0, 0.0
    for batch in loader:
        x, teacher = batch["input_surface"].to(device), batch["teacher_surface"].to(device)
        pred = model(x)
        loss, _ = completion_loss(pred, teacher, positive_weight)
        losses.append(float(loss))
        for dst, src in ((model_counts, surface_counts(pred, teacher, threshold)), (baseline_counts, surface_counts(x, teacher, threshold))):
            for key in dst: dst[key] += src[key]
        mask = teacher[:, :1]
        attr_abs += float((torch.abs(pred[:, 1:] - teacher[:, 1:]) * mask).sum())
        attr_n += float(mask.sum() * 3.0)
    return {"loss": float(np.mean(losses)), "model": {**finish_metrics(model_counts), "teacher_attribute_mae": attr_abs / max(attr_n, 1.0)}, "input_baseline": finish_metrics(baseline_counts)}


def train_step(model, batch, optimizer, device, positive_weight):
    model.train()
    x, teacher = batch["input_surface"].to(device), batch["teacher_surface"].to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss, parts = completion_loss(prediction, teacher, positive_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    loss.backward()
    optimizer.step()
    return {key: float(value) for key, value in parts.items()}


def save_prediction_preview(path: Path, x: torch.Tensor, initial: torch.Tensor | None, prediction: torch.Tensor, teacher: torch.Tensor, count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4 if initial is not None else 3
    fig, axes = plt.subplots(min(count, len(x)), columns, figsize=(3.4 * columns, 3.2 * min(count, len(x))), squeeze=False)
    for row in range(min(count, len(x))):
        items = [(x[row, 0], "input")]
        if initial is not None: items.append((initial[row, 0], "initial"))
        items.extend([(prediction[row, 0], "prediction"), (teacher[row, 0], "teacher")])
        for ax, (image, title) in zip(axes[row], items):
            ax.imshow(image.detach().cpu(), origin="upper", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_curve(path: Path, series: dict[str, Iterable[float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in series.items(): ax.plot(list(values), label=name)
    ax.set_title(title); ax.set_xlabel("step/epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def perturb(x: torch.Tensor, cfg: dict, generator: torch.Generator) -> torch.Tensor:
    out = x.clone()
    batch, _, height, width = out.shape
    mask = out[:, :1]
    keep = torch.rand(mask.shape, generator=generator, device=out.device) >= float(cfg["drop_probability"])
    out = out * keep
    scale = torch.empty((batch, 1, 1), device=out.device).uniform_(float(cfg["density_scale_min"]), float(cfg["density_scale_max"]), generator=generator)
    out[:, 1] = torch.clamp(out[:, 1] * scale, 0, 1)
    size = int(cfg["occlusion_size_cells"])
    for index in range(batch):
        row = int(torch.randint(0, height - size + 1, (1,), generator=generator, device=out.device))
        col = int(torch.randint(0, width - size + 1, (1,), generator=generator, device=out.device))
        out[index, :, row:row + size, col:col + size] = 0
    noise = torch.randn(out[:, 1:].shape, generator=generator, device=out.device) * float(cfg["noise_std"])
    out[:, 1:] = torch.clamp(out[:, 1:] + noise * out[:, :1], 0, 1)
    return out


def describe(values: np.ndarray) -> dict[str, float]:
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)), "max": float(values.max())}


@torch.no_grad()
def robustness_test(model, dataset, device, cfg):
    model.eval()
    count = min(int(cfg["samples"]), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    x = torch.from_numpy(np.stack([dataset[int(i)]["input_surface"] for i in indices])).to(device)
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    changed = perturb(x, cfg, generator)
    _, f = model(x, return_features=True)
    pred_changed, fc = model(changed, return_features=True)
    f = torch.nn.functional.normalize(f, dim=1); fc = torch.nn.functional.normalize(fc, dim=1)
    same = (1.0 - (f * fc).sum(dim=1)).cpu().numpy()
    perm = torch.roll(torch.arange(count, device=device), 1)
    different = (1.0 - (f * f[perm]).sum(dim=1)).cpu().numpy()
    output_delta = torch.mean(torch.abs(model(x) - pred_changed), dim=(1, 2, 3)).cpu().numpy()
    return {"same_sample_cosine_distance": describe(same), "different_sample_cosine_distance": describe(different), "same_less_than_different_fraction": float((same < different).mean()), "output_mean_absolute_delta": describe(output_delta), "pass": bool(np.median(same) < np.median(different))}


class MTAREInputs(Dataset):
    def __init__(self, root: Path): self.files = sorted(root.glob("*.npz"))
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with np.load(self.files[index], allow_pickle=False) as data: return data["input_surface"].astype(np.float32)


@torch.no_grad()
def mtare_inference(model, root, device, robustness_cfg):
    dataset = MTAREInputs(root)
    x = torch.from_numpy(np.stack([dataset[i] for i in range(len(dataset))])).to(device)
    generator = torch.Generator(device=device).manual_seed(int(robustness_cfg["seed"]) + 1)
    changed = perturb(x, robustness_cfg, generator)
    pred, features = model(x, return_features=True)
    changed_pred, changed_features = model(changed, return_features=True)
    norm = torch.nn.functional.normalize(features, dim=1); changed_norm = torch.nn.functional.normalize(changed_features, dim=1)
    distances = (1.0 - (norm * changed_norm).sum(dim=1)).cpu().numpy()
    pairwise = torch.pdist(norm).cpu().numpy()
    return {"sample_count": len(dataset), "forward_success": True, "finite": bool(torch.isfinite(pred).all()), "output_range": [float(pred.min()), float(pred.max())], "output_sample_std_mean": float(pred.flatten(1).std(dim=0).mean()), "feature_pairwise_distance": describe(pairwise), "perturbed_feature_distance": describe(distances), "perturbed_output_mae": float(torch.mean(torch.abs(pred - changed_pred))), "outputs_not_all_identical": bool(float(pred.flatten(1).std(dim=0).mean()) > 1e-6)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/structural_model_feasibility.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["experiment"]["output_root"]) / run_id
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    for name in ("config", "checkpoint", "training_log", "metrics", "overfit_previews", "validation_previews", "robustness_results", "mtare_inference_results"): (output / name).mkdir(parents=True, exist_ok=True)
    (output / "config" / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    seed_all(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("GPU experiment requested but CUDA is unavailable")
    train_data = StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"], "train")
    val_data = StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"], "val")
    common = {"base_channels": int(cfg["model"]["base_channels"]), "latent_dim": int(cfg["model"]["latent_dim"])}
    positive_weight = float(cfg["loss"]["positive_weight"])
    threshold = float(cfg["metrics"]["surface_threshold"])

    fixed_indices = np.linspace(0, len(train_data) - 1, int(cfg["overfit"]["samples"]), dtype=int).tolist()
    fixed_loader = DataLoader(Subset(train_data, fixed_indices), batch_size=int(cfg["overfit"]["batch_size"]), shuffle=False, collate_fn=collate)
    fixed_batch = next(iter(fixed_loader))
    overfit_model = TinySurfaceCompletionNet(**common).to(device)
    initial_checksum = parameter_checksum(overfit_model)
    with torch.no_grad(): initial_prediction = overfit_model(fixed_batch["input_surface"].to(device)).cpu()
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=float(cfg["overfit"]["learning_rate"]))
    overfit_history = []
    for step in range(int(cfg["overfit"]["steps"])):
        overfit_history.append(train_step(overfit_model, fixed_batch, optimizer, device, positive_weight)["total"])
    with torch.no_grad(): final_prediction = overfit_model(fixed_batch["input_surface"].to(device)).cpu()
    final_checksum = parameter_checksum(overfit_model)
    overfit = {"samples": len(fixed_indices), "steps": len(overfit_history), "initial_loss": overfit_history[0], "middle_loss": overfit_history[len(overfit_history)//2], "final_loss": overfit_history[-1], "minimum_loss": min(overfit_history), "parameter_checksum_before": initial_checksum, "parameter_checksum_after": final_checksum, "parameters_updated": not math.isclose(initial_checksum, final_checksum), "finite": bool(np.isfinite(overfit_history).all()), "pass": bool(overfit_history[-1] < overfit_history[0] * 0.35)}
    write_json(output / "metrics" / "overfit.json", overfit)
    save_curve(output / "overfit_previews" / "loss_curve.png", {"overfit": overfit_history}, "Fixed-sample overfit")
    save_prediction_preview(output / "overfit_previews" / "input_initial_prediction_teacher.png", fixed_batch["input_surface"], initial_prediction, final_prediction, fixed_batch["teacher_surface"])

    model = TinySurfaceCompletionNet(**common).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"]["weight_decay"]))
    train_loader = DataLoader(train_data, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, collate_fn=collate, generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])))
    val_loader = DataLoader(val_data, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, collate_fn=collate)
    history, best_loss = [], float("inf")
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        losses = [train_step(model, batch, optimizer, device, positive_weight)["total"] for batch in train_loader]
        val_metrics = evaluate(model, val_loader, device, threshold, positive_weight)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_metrics["loss"], "val_model_iou": val_metrics["model"]["surface_iou"], "val_baseline_iou": val_metrics["input_baseline"]["surface_iou"]}
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg, "metrics": val_metrics}, output / "checkpoint" / "best.pt")
        print(json.dumps(row), flush=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": len(history), "config": cfg}, output / "checkpoint" / "last.pt")
    best = torch.load(output / "checkpoint" / "best.pt", map_location=device, weights_only=False)
    restored = TinySurfaceCompletionNet(**common).to(device); restored.load_state_dict(best["model"])
    model = restored
    final_val = evaluate(model, val_loader, device, threshold, positive_weight)
    final_val["best_epoch"] = int(best["epoch"]); final_val["checkpoint_restore_success"] = True
    write_json(output / "metrics" / "train_val.json", {"history": history, "best_validation": final_val})
    save_curve(output / "training_log" / "train_val_curve.png", {"train": [x["train_loss"] for x in history], "val": [x["val_loss"] for x in history]}, "Small train/validation run")
    preview_batch = next(iter(val_loader)); preview_x = preview_batch["input_surface"].to(device)
    with torch.no_grad(): preview_pred = model(preview_x).cpu()
    save_prediction_preview(output / "validation_previews" / "input_prediction_teacher.png", preview_batch["input_surface"], None, preview_pred, preview_batch["teacher_surface"], int(cfg["metrics"]["preview_samples"]))

    robustness = robustness_test(model, val_data, device, cfg["robustness"])
    write_json(output / "robustness_results" / "summary.json", robustness)
    mtare = mtare_inference(model, Path(cfg["experiment"]["dataset_dir"]) / "mtare_samples", device, cfg["robustness"])
    write_json(output / "mtare_inference_results" / "summary.json", mtare)
    environment = {"hostname": os.uname().nodename, "python": os.sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "dataset_dir": str(Path(cfg["experiment"]["dataset_dir"]).resolve())}
    write_json(output / "metrics" / "environment.json", environment)
    summary = {"overfit": overfit, "train_val": final_val, "robustness": robustness, "mtare": mtare, "environment": environment, "model_beats_input_baseline": final_val["model"]["surface_iou"] > final_val["input_baseline"]["surface_iou"], "status": "MODEL_FEASIBLE" if overfit["pass"] and final_val["model"]["surface_iou"] > final_val["input_baseline"]["surface_iou"] and robustness["pass"] and mtare["forward_success"] and mtare["finite"] else "MODEL_NOT_FEASIBLE"}
    write_json(output / "metrics" / "summary.json", summary)
    print(json.dumps({"result_dir": str(output), "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
