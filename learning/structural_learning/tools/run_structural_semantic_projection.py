from __future__ import annotations

import argparse
import hashlib
import json
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
from torch.nn import functional as F

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.tools.run_bottleneck_feasibility import make_perturbations
from learning.structural_learning.tools.run_structural_semantic_validation import (
    adjacent_edges,
    all_cross_pairs,
    cosine_distance,
    describe,
    distribution_test,
    encode,
    load_split,
    retrieval_audit,
    save_cross_preview,
    save_map_examples,
    save_trajectory_preview,
    spearman,
    stable_segments,
    surface_descriptors,
    write_json,
)


def train_projection(
    base_latent: np.ndarray,
    perturbed_latent: np.ndarray,
    teacher_descriptor: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[StructuralProjectionHead, dict[str, Any], np.ndarray, np.ndarray]:
    projection_cfg = cfg["projection"]
    centered = teacher_descriptor - teacher_descriptor.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[: int(projection_cfg["teacher_pca_dim"])]
    target = centered @ components.T
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)

    model = StructuralProjectionHead(
        input_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(projection_cfg["hidden_dim"]),
        output_dim=int(projection_cfg["output_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(projection_cfg["learning_rate"]), weight_decay=float(projection_cfg["weight_decay"])
    )
    base_tensor = torch.from_numpy(base_latent).to(device)
    perturbed_tensor = torch.from_numpy(perturbed_latent).to(device)
    target_tensor = torch.from_numpy(target.astype(np.float32)).to(device)
    history = []
    steps = int(projection_cfg["steps"])
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(base_tensor)
        altered = model(perturbed_tensor)
        cosine_loss = (1.0 - (prediction * target_tensor).sum(1)).mean()
        mse_loss = F.mse_loss(prediction, target_tensor)
        consistency = (1.0 - (prediction * altered).sum(1)).mean()
        total = cosine_loss + float(projection_cfg["target_mse_weight"]) * mse_loss + float(projection_cfg["perturbation_consistency_weight"]) * consistency
        if not torch.isfinite(total):
            raise RuntimeError("non-finite projection loss")
        total.backward(); optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "total": float(total.detach()),
                    "teacher_cosine": float(cosine_loss.detach()),
                    "target_mse": float(mse_loss.detach()),
                    "perturbation_consistency": float(consistency.detach()),
                }
            )
    return model, {"history": history, "initial_loss": history[0], "final_loss": history[-1]}, teacher_descriptor.mean(axis=0), components


@torch.no_grad()
def encode_perturbed_base(
    base: ForcedGlobalBottleneckNet, inputs: np.ndarray, channels: list[int], device: torch.device
) -> np.ndarray:
    output = []
    for start in range(0, len(inputs), 64):
        batch = torch.from_numpy(inputs[start : start + 64, channels]).to(device)
        changed = make_perturbations(batch)["random_surface_drop"]
        output.append(F.normalize(base.encode(changed), dim=1).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def project(projection: StructuralProjectionHead, latent: np.ndarray, device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(latent), 128):
        output.append(projection(torch.from_numpy(latent[start : start + 128]).to(device)).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def projected_perturbation_audit(
    base: ForcedGlobalBottleneckNet,
    projection: StructuralProjectionHead,
    inputs: np.ndarray,
    channels: list[int],
    device: torch.device,
    count: int,
    reference: np.ndarray,
) -> dict[str, Any]:
    indices = np.linspace(0, len(inputs) - 1, min(count, len(inputs)), dtype=int)
    batch = torch.from_numpy(inputs[indices][:, channels]).to(device)
    original = projection(base.encode(batch))
    result = {}
    for name, changed in make_perturbations(batch).items():
        altered = projection(base.encode(changed))
        distance = torch.clamp(1.0 - (original * altered).sum(1), min=0).cpu().numpy()
        reference_resized = np.resize(reference, len(distance))
        result[name] = {
            "same_sample": describe(distance),
            "structural_change_reference": describe(reference_resized),
            "same_less_fraction": float(np.mean(distance < reference_resized)),
        }
    return result


@torch.no_grad()
def projected_mtare_audit(
    base: ForcedGlobalBottleneckNet,
    projection: StructuralProjectionHead,
    root: Path,
    channels: list[int],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            rows.append((int(data["stamp_ns"]), data["input_surface"].astype(np.float32)))
    rows.sort(key=lambda row: row[0])
    inputs = np.stack([row[1] for row in rows])
    batch = torch.from_numpy(inputs[:, channels]).to(device)
    latent = projection(base.encode(batch))
    latent_np = latent.cpu().numpy()
    descriptors = surface_descriptors(inputs, device)
    adjacent_latent = cosine_distance(latent_np[:-1], latent_np[1:])
    adjacent_geometry = cosine_distance(descriptors[:-1], descriptors[1:])
    stable = adjacent_geometry <= np.quantile(adjacent_geometry, 0.4)
    changed = adjacent_geometry >= np.quantile(adjacent_geometry, 0.8)
    perturbations = {}
    for name, altered_input in make_perturbations(batch).items():
        altered = projection(base.encode(altered_input))
        perturbations[name] = describe(torch.clamp(1.0 - (latent * altered).sum(1), min=0).cpu().numpy())
    left, right = np.triu_indices(len(rows), k=1)
    return {
        "samples": len(rows),
        "input_contract_shape": list(inputs.shape[1:]),
        "model_input_channels": channels,
        "forward": True,
        "finite": bool(torch.isfinite(latent).all()),
        "output_not_identical": bool(np.mean(np.std(latent_np, axis=0)) > 1e-6),
        "stamp_interval_s": describe(np.diff([row[0] for row in rows]) / 1e9),
        "adjacent_latent_distance": describe(adjacent_latent),
        "adjacent_input_geometry_distance": describe(adjacent_geometry),
        "stable_input_geometry_latent_distance": describe(adjacent_latent[stable]),
        "changed_input_geometry_latent_distance": describe(adjacent_latent[changed]),
        "input_geometry_latent_spearman": spearman(adjacent_latent, adjacent_geometry),
        "all_sample_pair_latent_distance": describe(cosine_distance(latent_np[left], latent_np[right])),
        "perturbations": perturbations,
    }


def run_validation(
    samples: list[dict[str, Any]],
    latent: np.ndarray,
    input_descriptor: np.ndarray,
    teacher_descriptor: np.ndarray,
    cells: np.ndarray,
    density: np.ndarray,
    cfg: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    audit = cfg["audit"]
    edges = adjacent_edges(samples, float(audit["trajectory_max_step_m"]), float(audit["trajectory_max_dt_s"]))
    edge_left = np.asarray([edge[0] for edge in edges]); edge_right = np.asarray([edge[1] for edge in edges])
    teacher_edge = cosine_distance(teacher_descriptor[edge_left], teacher_descriptor[edge_right])
    latent_edge = cosine_distance(latent[edge_left], latent[edge_right])
    coverage_edge = cosine_distance(input_descriptor[edge_left], input_descriptor[edge_right])
    cell_edge = np.abs(cells[edge_left] - cells[edge_right]) / np.maximum(np.maximum(cells[edge_left], cells[edge_right]), 1)
    stable_threshold = float(np.quantile(teacher_edge, float(audit["stable_quantile"])))
    change_threshold = float(np.quantile(teacher_edge, float(audit["change_quantile"])))
    stable = teacher_edge <= stable_threshold; changed = teacher_edge >= change_threshold
    segments = stable_segments(edges, stable, samples, int(audit["minimum_stable_edges"]))
    stable_result = {
        "definition": f"same-robot consecutive edges, teacher multiscale geometry <= q{audit['stable_quantile']}",
        "valid_consecutive_edges": len(edges), "stable_edge_count": int(stable.sum()), "stable_segment_count": len(segments), "segments": segments,
        "latent_distance": describe(latent_edge[stable]), "teacher_geometry_distance": describe(teacher_edge[stable]),
        "input_coverage_shape_distance": describe(coverage_edge[stable]), "surface_cell_count_difference": describe(cell_edge[stable]),
    }
    change_test = distribution_test(latent_edge[stable], latent_edge[changed])
    change_result = {
        "reference": "multiscale dilated teacher surface layout; no free/unknown or manual labels",
        "stable_threshold": stable_threshold, "change_threshold": change_threshold, "change_edge_count": int(changed.sum()),
        "latent_distance_test": change_test,
        "teacher_geometry_distance": {"stable": describe(teacher_edge[stable]), "changed": describe(teacher_edge[changed])},
        "input_coverage_shape_distance": {"stable": describe(coverage_edge[stable]), "changed": describe(coverage_edge[changed])},
        "surface_cell_count_difference": {"stable": describe(cell_edge[stable]), "changed": describe(cell_edge[changed])},
        "all_adjacent_teacher_latent_spearman": spearman(teacher_edge, latent_edge),
    }
    left, right, world_distance = all_cross_pairs(samples, float(audit["cross_location_min_distance_m"]))
    pair_latent = cosine_distance(latent[left], latent[right]); pair_teacher = cosine_distance(teacher_descriptor[left], teacher_descriptor[right])
    pair_coverage = cosine_distance(input_descriptor[left], input_descriptor[right])
    pair_cells = np.abs(cells[left] - cells[right]) / np.maximum(np.maximum(cells[left], cells[right]), 1)
    pair_density = np.abs(density[left] - density[right])
    pair_robot = np.asarray([samples[l]["robot"] != samples[r]["robot"] for l, r in zip(left, right)], dtype=np.float64)
    pair_time = np.log1p(np.abs(np.asarray([samples[l]["stamp_ns"] - samples[r]["stamp_ns"] for l, r in zip(left, right)], dtype=np.float64)) / 1e9)
    similar_threshold = float(np.quantile(pair_teacher, float(audit["cross_similar_quantile"])))
    different_threshold = float(np.quantile(pair_teacher, float(audit["cross_different_quantile"])))
    similar = pair_teacher <= similar_threshold; different = pair_teacher >= different_threshold
    cross_test = distribution_test(pair_latent[similar], pair_latent[different])
    cross_result = {
        "definition": f"world center distance >= {audit['cross_location_min_distance_m']} m", "pair_count": len(pair_latent),
        "similar_teacher_threshold": similar_threshold, "different_teacher_threshold": different_threshold,
        "similar_structure_pairs": describe(pair_latent[similar]), "different_structure_pairs": describe(pair_latent[different]),
        "distance_test": cross_test,
        "world_distance_m": {"similar": describe(world_distance[similar]), "different": describe(world_distance[different])},
        "retrieval": retrieval_audit(latent, teacher_descriptor, samples, float(audit["cross_location_min_distance_m"]), int(audit["retrieval_k"]), similar_threshold),
    }
    nuisance = {
        "pair_count": len(pair_latent), "pair_definition": cross_result["definition"],
        "spearman": {
            "teacher_geometry_difference": spearman(pair_latent, pair_teacher),
            "input_coverage_shape_difference": spearman(pair_latent, pair_coverage),
            "surface_cell_count_difference": spearman(pair_latent, pair_cells),
            "log_density_proxy_difference": spearman(pair_latent, pair_density),
            "robot_identity_difference": spearman(pair_latent, pair_robot),
            "log_time_interval": spearman(pair_latent, pair_time),
            "world_distance": spearman(pair_latent, world_distance),
        },
        "robot_conditioned": {"same_robot_latent_distance": describe(pair_latent[pair_robot == 0]), "different_robot_latent_distance": describe(pair_latent[pair_robot == 1])},
    }
    write_json(root / "stable_segment_audit" / "summary.json", stable_result)
    write_json(root / "structural_change_audit" / "summary.json", change_result)
    write_json(root / "cross_location_similarity" / "summary.json", cross_result)
    write_json(root / "nuisance_factor_audit" / "summary.json", nuisance)
    save_trajectory_preview(root / "trajectory_previews" / "validation_trajectory.png", samples, edges, teacher_edge, latent_edge, stable, changed)
    save_map_examples(root / "trajectory_previews" / "stable_and_change_examples.png", samples, edges, teacher_edge, stable, changed, int(audit["preview_examples"]))
    save_cross_preview(root / "trajectory_previews" / "cross_location_examples.png", samples, left, right, similar, different, pair_latent, pair_teacher, int(audit["preview_examples"]))
    return {"stable": stable_result, "change": change_result, "cross": cross_result, "nuisance": nuisance, "changed_reference": latent_edge[changed]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/learning/structural_semantic_projection.yaml"); parser.add_argument("--run-id", default=None); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["experiment"]["output_root"]) / run_id
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    for name in ("baseline_metrics", "final_metrics", "stable_segment_audit", "structural_change_audit", "cross_location_similarity", "nuisance_factor_audit", "mtare_transfer_audit", "trajectory_previews", "config", "checkpoint", "training_log"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "config" / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    seed = int(cfg["experiment"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if cfg["experiment"]["deterministic"]: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda"); channels = list(cfg["model"]["input_channels"])
    base = ForcedGlobalBottleneckNet(input_channels=len(channels), latent_dim=int(cfg["model"]["latent_dim"]), base_channels=int(cfg["model"]["base_channels"])).to(device)
    checkpoint_path = Path(cfg["experiment"]["checkpoint"]); checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); base.load_state_dict(checkpoint["model"]); base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    dataset_root = Path(cfg["experiment"]["dataset_dir"]); train = load_split(dataset_root, "train"); val = load_split(dataset_root, "val")
    train_inputs = np.stack([sample["input"] for sample in train]); val_inputs = np.stack([sample["input"] for sample in val]); train_teachers = np.stack([sample["teacher"] for sample in train]); val_teachers = np.stack([sample["teacher"] for sample in val])
    train_base = encode(base, train_inputs, channels, device); val_base = encode(base, val_inputs, channels, device); train_perturbed = encode_perturbed_base(base, train_inputs, channels, device)
    train_teacher_descriptor = surface_descriptors(train_teachers, device); val_teacher_descriptor = surface_descriptors(val_teachers, device); val_input_descriptor = surface_descriptors(val_inputs, device)
    projection, training, pca_mean, pca_components = train_projection(train_base, train_perturbed, train_teacher_descriptor, cfg, device); projection.eval()
    final_latent = project(projection, val_base, device)
    cells = (val_inputs[:, 0] > 0.5).sum(axis=(1, 2)).astype(np.float64)
    density = np.asarray([float(sample["input"][1][sample["input"][0] > 0.5].mean()) if np.any(sample["input"][0] > 0.5) else 0.0 for sample in val])
    audit_result = run_validation(val, final_latent, val_input_descriptor, val_teacher_descriptor, cells, density, cfg, root)
    perturb = projected_perturbation_audit(base, projection, val_inputs, channels, device, int(cfg["audit"]["perturbation_samples"]), audit_result["changed_reference"])
    mtare = projected_mtare_audit(base, projection, dataset_root / "mtare_samples", channels, device)
    write_json(root / "final_metrics" / "perturbation.json", perturb); write_json(root / "mtare_transfer_audit" / "summary.json", mtare); write_json(root / "training_log" / "summary.json", training)
    torch.save({"projection": projection.state_dict(), "pca_mean": pca_mean, "pca_components": pca_components, "config": cfg, "base_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}, root / "checkpoint" / "projection.pt")
    baseline_path = Path(cfg["experiment"]["baseline_validation_run"]) / "baseline_metrics" / "summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")); write_json(root / "baseline_metrics" / "summary.json", baseline)
    correlations = audit_result["nuisance"]["spearman"]
    nuisance_max = max(abs(correlations[name]) for name in ("input_coverage_shape_difference", "surface_cell_count_difference", "log_density_proxy_difference", "robot_identity_difference", "log_time_interval"))
    change_test = audit_result["change"]["latent_distance_test"]; cross_test = audit_result["cross"]["distance_test"]
    change_pass = change_test["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_change_to_stable_ratio"]) and change_test["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
    cross_pass = cross_test["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_cross_location_ratio"]) and cross_test["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
    geometry_pass = correlations["teacher_geometry_difference"] > nuisance_max
    perturb_pass = all(value["same_sample"]["median"] < value["structural_change_reference"]["median"] for value in perturb.values())
    mtare_pass = mtare["forward"] and mtare["finite"] and mtare["output_not_identical"] and mtare["changed_input_geometry_latent_distance"]["median"] > mtare["stable_input_geometry_latent_distance"]["median"]
    all_pass = change_pass and cross_pass and geometry_pass and perturb_pass and mtare_pass
    core_pass = change_pass and cross_pass and perturb_pass and mtare["forward"] and mtare["finite"]
    known_limitations = [name for name, present in cfg.get("limitations", {}).items() if present]
    status = "STRUCTURAL_REPRESENTATION_READY_WITH_LIMITATIONS" if all_pass and known_limitations else "STRUCTURAL_REPRESENTATION_READY" if all_pass else "STRUCTURAL_REPRESENTATION_READY_WITH_LIMITATIONS" if core_pass else "STRUCTURAL_REPRESENTATION_NOT_READY"
    final = {
        "status": status,
        "representation": f"64-dimensional projection of frozen 128-dimensional bottleneck",
        "new_training": True, "supervised_parameters": sum(parameter.numel() for parameter in projection.parameters()),
        "base_encoder_frozen": True, "teacher_use": "training-only PCA geometry target and offline validation reference",
        "acceptance": {"stable_vs_change": change_pass, "cross_location_similarity": cross_pass, "geometry_dominates_nuisance": geometry_pass, "observation_degradation_stability": perturb_pass, "mtare_transfer": mtare_pass},
        "known_limitations": known_limitations,
        "dominant_nuisance_abs_spearman": nuisance_max, "teacher_geometry_spearman": correlations["teacher_geometry_difference"],
        "baseline_teacher_geometry_spearman": baseline["nuisance"]["spearman"]["teacher_geometry_difference"],
        "baseline_dominant_nuisance_abs_spearman": max(abs(baseline["nuisance"]["spearman"][name]) for name in ("input_coverage_shape_difference", "surface_cell_count_difference", "log_density_proxy_difference", "robot_identity_difference", "log_time_interval")),
        "environment": {"hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }
    write_json(root / "final_metrics" / "summary.json", final)
    print(json.dumps({"result_dir": str(root), "status": status, "acceptance": final["acceptance"], "correlations": correlations}, indent=2), flush=True)


if __name__ == "__main__":
    main()
