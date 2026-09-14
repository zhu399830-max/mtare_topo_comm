from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from scipy.stats import ks_2samp
from torch.nn import functional as F

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.tools.run_bottleneck_feasibility import make_perturbations
from learning.structural_learning.tools.run_structural_semantic_validation import (
    cosine_distance,
    describe,
    distribution_test,
    retrieval_audit,
    save_cross_preview,
    save_map_examples,
    spearman,
    surface_descriptors,
    write_json,
)


def save_world_trajectory_preview(
    path: Path,
    world: str,
    samples: list[dict[str, Any]],
    edges: list[tuple[int, int, float, float]],
    teacher_change: np.ndarray,
    latent_change: np.ndarray,
    stable: np.ndarray,
    changed: np.ndarray,
) -> None:
    xy = np.stack([sample["pose"][:2] for sample in samples])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=1)
    for edge_index, (left, right, _, _) in enumerate(edges):
        segment = np.stack((samples[left]["pose"][:2], samples[right]["pose"][:2]))
        if changed[edge_index]:
            axes[0].plot(segment[:, 0], segment[:, 1], color="crimson", linewidth=2.2)
        elif stable[edge_index]:
            axes[0].plot(segment[:, 0], segment[:, 1], color="seagreen", linewidth=1.5)
    axes[0].set_title(f"{world}: stable (green), change (red)")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("world x [m]")
    axes[0].set_ylabel("world y [m]")
    axes[1].scatter(teacher_change, latent_change, s=18, alpha=0.65)
    axes[1].set_xlabel("teacher multiscale geometry distance")
    axes[1].set_ylabel("representation cosine distance")
    axes[1].set_title("Consecutive trajectory edges")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_world(root: Path, world: str) -> list[dict[str, Any]]:
    samples = []
    for path in sorted((root / world / "samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            samples.append({
                "path": str(path), "input": data["input_surface"].astype(np.float32), "teacher": data["teacher_surface"].astype(np.float32),
                "pose": data["center_pose"].astype(np.float64), "stamp_ns": int(data["stamp_ns"]), "world": str(data["world"]),
                "raw_point_count": int(data["raw_point_count"]), "source_frame_index": int(data["source_frame_index"]),
                "teacher_sources": int(len(data["teacher_source_indices"])), "key": int(data["source_frame_index"]), "robot": world,
            })
    if not samples: raise RuntimeError(f"no exported samples for {world}")
    return samples


@torch.no_grad()
def represent(base: ForcedGlobalBottleneckNet, projection: StructuralProjectionHead, inputs: np.ndarray, channels: list[int], device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(inputs), 64):
        batch = torch.from_numpy(inputs[start:start + 64, channels]).to(device)
        output.append(projection(base.encode(batch)).cpu().numpy())
    return np.concatenate(output)


def sequential_edges(samples: list[dict[str, Any]], max_step: float, max_dt: float) -> list[tuple[int, int, float, float]]:
    edges = []
    for left, right in zip(range(len(samples) - 1), range(1, len(samples))):
        dt = (samples[right]["stamp_ns"] - samples[left]["stamp_ns"]) / 1e9
        step = float(np.linalg.norm(samples[right]["pose"][:3] - samples[left]["pose"][:3]))
        if 0 < dt <= max_dt and step <= max_step:
            edges.append((left, right, dt, step))
    return edges


def overlap_metrics(stable: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    result = distribution_test(stable, changed); ks = ks_2samp(stable, changed, alternative="two-sided", method="auto")
    result["ks_statistic"] = float(ks.statistic); result["ks_p_value"] = float(ks.pvalue); result["distribution_overlap_one_minus_ks"] = float(1.0 - ks.statistic)
    return result


def world_pairs(samples: list[dict[str, Any]], minimum_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(samples), k=1); xyz = np.stack([sample["pose"][:3] for sample in samples]); distance = np.linalg.norm(xyz[left] - xyz[right], axis=1); keep = distance >= minimum_distance
    return left[keep], right[keep], distance[keep]


@torch.no_grad()
def perturbation_audit(base: ForcedGlobalBottleneckNet, projection: StructuralProjectionHead, inputs: np.ndarray, channels: list[int], device: torch.device, count: int, different_reference: np.ndarray) -> dict[str, Any]:
    indices = np.linspace(0, len(inputs) - 1, min(count, len(inputs)), dtype=int); batch = torch.from_numpy(inputs[indices][:, channels]).to(device); original = projection(base.encode(batch)); result = {}
    for name, changed in make_perturbations(batch).items():
        altered = projection(base.encode(changed)); distance = torch.clamp(1.0 - (original * altered).sum(1), min=0).cpu().numpy(); reference = np.resize(different_reference, len(distance))
        result[name] = {"same_sample": describe(distance), "different_structure_reference": describe(reference), "same_less_fraction": float(np.mean(distance < reference))}
    return result


def save_world_previews(root: Path, world: str, samples: list[dict[str, Any]], edges: list[tuple[int, int, float, float]], teacher_edge: np.ndarray, latent_edge: np.ndarray, stable: np.ndarray, changed: np.ndarray, left: np.ndarray, right: np.ndarray, similar: np.ndarray, different: np.ndarray, pair_latent: np.ndarray, pair_teacher: np.ndarray, count: int) -> None:
    target = root / "trajectory_previews" / world; target.mkdir(parents=True, exist_ok=True)
    save_world_trajectory_preview(target / "trajectory.png", world, samples, edges, teacher_edge, latent_edge, stable, changed)
    save_map_examples(target / "stable_change_examples.png", samples, edges, teacher_edge, stable, changed, count)
    save_cross_preview(target / "cross_location_examples.png", samples, left, right, similar, different, pair_latent, pair_teacher, count)


def evaluate_world(world: str, samples: list[dict[str, Any]], latent: np.ndarray, input_descriptor: np.ndarray, teacher_descriptor: np.ndarray, cfg: dict[str, Any], root: Path, base: ForcedGlobalBottleneckNet, projection: StructuralProjectionHead, device: torch.device) -> dict[str, Any]:
    audit = cfg["audit"]; inputs = np.stack([sample["input"] for sample in samples]); cells = (inputs[:, 0] > .5).sum(axis=(1, 2)).astype(np.float64)
    density = np.asarray([float(x[1][x[0] > .5].mean()) if np.any(x[0] > .5) else 0.0 for x in inputs]); raw_points = np.asarray([sample["raw_point_count"] for sample in samples], dtype=np.float64)
    edges = sequential_edges(samples, float(audit["trajectory_max_step_m"]), float(audit["trajectory_max_dt_s"])); edge_left = np.asarray([edge[0] for edge in edges]); edge_right = np.asarray([edge[1] for edge in edges])
    teacher_edge = cosine_distance(teacher_descriptor[edge_left], teacher_descriptor[edge_right]); latent_edge = cosine_distance(latent[edge_left], latent[edge_right]); coverage_edge = cosine_distance(input_descriptor[edge_left], input_descriptor[edge_right])
    stable_threshold = float(np.quantile(teacher_edge, float(audit["stable_quantile"]))); change_threshold = float(np.quantile(teacher_edge, float(audit["change_quantile"]))); stable = teacher_edge <= stable_threshold; changed = teacher_edge >= change_threshold
    stable_change = {
        "world": world, "samples": len(samples), "valid_edges": len(edges), "stable_edges": int(stable.sum()), "change_edges": int(changed.sum()),
        "stable_teacher_threshold": stable_threshold, "change_teacher_threshold": change_threshold,
        "latent_distance": overlap_metrics(latent_edge[stable], latent_edge[changed]),
        "teacher_geometry": {"stable": describe(teacher_edge[stable]), "changed": describe(teacher_edge[changed])},
        "input_coverage": {"stable": describe(coverage_edge[stable]), "changed": describe(coverage_edge[changed])},
        "motion_distance_m": {"stable": describe(np.asarray([edges[i][3] for i in np.flatnonzero(stable)])), "changed": describe(np.asarray([edges[i][3] for i in np.flatnonzero(changed)]))},
        "time_interval_s": {"stable": describe(np.asarray([edges[i][2] for i in np.flatnonzero(stable)])), "changed": describe(np.asarray([edges[i][2] for i in np.flatnonzero(changed)]))},
        "teacher_latent_spearman": spearman(teacher_edge, latent_edge),
    }
    left, right, spatial_distance = world_pairs(samples, float(audit["cross_location_min_distance_m"])); pair_latent = cosine_distance(latent[left], latent[right]); pair_teacher = cosine_distance(teacher_descriptor[left], teacher_descriptor[right]); pair_coverage = cosine_distance(input_descriptor[left], input_descriptor[right])
    pair_cells = np.abs(cells[left] - cells[right]) / np.maximum(np.maximum(cells[left], cells[right]), 1); pair_density = np.abs(density[left] - density[right]); pair_points = np.abs(raw_points[left] - raw_points[right]) / np.maximum(np.maximum(raw_points[left], raw_points[right]), 1); pair_time = np.log1p(np.abs(np.asarray([samples[l]["stamp_ns"] - samples[r]["stamp_ns"] for l, r in zip(left, right)], dtype=np.float64)) / 1e9)
    similar_threshold = float(np.quantile(pair_teacher, float(audit["cross_similar_quantile"]))); different_threshold = float(np.quantile(pair_teacher, float(audit["cross_different_quantile"]))); similar = pair_teacher <= similar_threshold; different = pair_teacher >= different_threshold
    cross = {
        "world": world, "minimum_spatial_distance_m": float(audit["cross_location_min_distance_m"]), "pair_count": len(pair_latent), "similar_pairs": int(similar.sum()), "different_pairs": int(different.sum()),
        "similar_structure_distance": describe(pair_latent[similar]), "different_structure_distance": describe(pair_latent[different]), "distance_test": overlap_metrics(pair_latent[similar], pair_latent[different]),
        "spatial_distance_m": {"similar": describe(spatial_distance[similar]), "different": describe(spatial_distance[different])},
        "retrieval": retrieval_audit(latent, teacher_descriptor, samples, float(audit["cross_location_min_distance_m"]), int(audit["retrieval_k"]), similar_threshold),
    }
    nuisance = {
        "world": world, "pair_count": len(pair_latent), "spearman": {
            "teacher_geometry_difference": spearman(pair_latent, pair_teacher), "input_coverage_shape_difference": spearman(pair_latent, pair_coverage),
            "surface_cell_count_difference": spearman(pair_latent, pair_cells), "log_density_proxy_difference": spearman(pair_latent, pair_density),
            "raw_point_count_difference": spearman(pair_latent, pair_points), "robot_motion_spatial_distance": spearman(pair_latent, spatial_distance), "log_time_interval": spearman(pair_latent, pair_time),
        },
        "input_statistics": {"surface_cells": describe(cells), "density_proxy": describe(density), "raw_points": describe(raw_points)},
    }
    perturb = perturbation_audit(base, projection, inputs, list(cfg["model"]["input_channels"]), device, int(audit["perturbation_samples"]), pair_latent[different])
    write_json(root / "stable_change_metrics" / f"{world}.json", stable_change); write_json(root / "cross_location_metrics" / f"{world}.json", cross); write_json(root / "nuisance_factor_metrics" / f"{world}.json", nuisance); write_json(root / "perturbation_metrics" / f"{world}.json", perturb)
    save_world_previews(root, world, samples, edges, teacher_edge, latent_edge, stable, changed, left, right, similar, different, pair_latent, pair_teacher, int(audit["preview_examples"]))
    return {"stable_change": stable_change, "cross": cross, "nuisance": nuisance, "perturbation": perturb, "latent": latent, "teacher_descriptor": teacher_descriptor, "input_descriptor": input_descriptor, "cells": cells, "density": density, "raw_points": raw_points}


def pooled_world_audit(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    worlds = sorted(results); features = np.concatenate([results[world]["latent"] for world in worlds]); teacher = np.concatenate([results[world]["teacher_descriptor"] for world in worlds]); coverage = np.concatenate([results[world]["input_descriptor"] for world in worlds]); cells = np.concatenate([results[world]["cells"] for world in worlds]); density = np.concatenate([results[world]["density"] for world in worlds]); points = np.concatenate([results[world]["raw_points"] for world in worlds]); labels = np.concatenate([[world] * len(results[world]["latent"]) for world in worlds])
    left, right = np.triu_indices(len(features), k=1); latent_distance = cosine_distance(features[left], features[right]); teacher_distance = cosine_distance(teacher[left], teacher[right]); coverage_distance = cosine_distance(coverage[left], coverage[right]); cell_distance = np.abs(cells[left] - cells[right]) / np.maximum(np.maximum(cells[left], cells[right]), 1); density_distance = np.abs(density[left] - density[right]); point_distance = np.abs(points[left] - points[right]) / np.maximum(np.maximum(points[left], points[right]), 1); world_difference = (labels[left] != labels[right]).astype(np.float64)
    return {
        "samples": len(features), "pairs": len(left), "worlds": worlds,
        "spearman": {"teacher_geometry_difference": spearman(latent_distance, teacher_distance), "input_coverage_shape_difference": spearman(latent_distance, coverage_distance), "surface_cell_count_difference": spearman(latent_distance, cell_distance), "log_density_proxy_difference": spearman(latent_distance, density_distance), "raw_point_count_difference": spearman(latent_distance, point_distance), "world_identity_difference": spearman(latent_distance, world_difference)},
        "same_world_latent_distance": describe(latent_distance[world_difference == 0]), "different_world_latent_distance": describe(latent_distance[world_difference == 1]),
    }


def lamp_input_statistics(dataset_root: Path) -> dict[str, Any]:
    cells = []; density = []
    for path in sorted((dataset_root / "val").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            surface = data["input_surface"]; mask = surface[0] > .5; cells.append(int(mask.sum())); density.append(float(surface[1][mask].mean()) if np.any(mask) else 0.0)
    return {"samples": len(cells), "surface_cells": describe(np.asarray(cells)), "density_proxy": describe(np.asarray(density))}


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/learning/mtare_unseen_transfer.yaml"); parser.add_argument("--run-id", default=None); args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"); root = Path(cfg["experiment"]["final_output_root"]) / run_id
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    for name in ("world_audit", "input_contract_audit", "stable_change_metrics", "cross_location_metrics", "nuisance_factor_metrics", "perturbation_metrics", "cross_world_comparison", "trajectory_previews", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, root / "config" / "config.yaml")
    source_paths = [Path(__file__), Path("learning/structural_learning/bottleneck_model.py"), Path("learning/structural_learning/semantic_projection.py"), Path("learning/structural_learning/surface_evidence.py")]
    provenance = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
    }
    write_json(root / "config" / "provenance.json", provenance)
    (root / "config" / "package_list.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"])["stdout"], encoding="utf-8")
    random.seed(int(cfg["experiment"]["seed"])); np.random.seed(int(cfg["experiment"]["seed"])); torch.manual_seed(int(cfg["experiment"]["seed"])); torch.cuda.manual_seed_all(int(cfg["experiment"]["seed"])); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda"); channels = list(cfg["model"]["input_channels"]); base = ForcedGlobalBottleneckNet(input_channels=len(channels), latent_dim=int(cfg["model"]["encoder_latent_dim"]), base_channels=int(cfg["model"]["encoder_base_channels"])).to(device); base_checkpoint_path = Path(cfg["experiment"]["encoder_checkpoint"]); base_checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False); base.load_state_dict(base_checkpoint["model"]); base.eval()
    projection = StructuralProjectionHead(int(cfg["model"]["encoder_latent_dim"]), int(cfg["model"]["projection_hidden_dim"]), int(cfg["model"]["representation_dim"])).to(device); projection_checkpoint_path = Path(cfg["experiment"]["projection_checkpoint"]); projection_checkpoint = torch.load(projection_checkpoint_path, map_location=device, weights_only=False); projection.load_state_dict(projection_checkpoint["projection"]); projection.eval()
    for parameter in list(base.parameters()) + list(projection.parameters()): parameter.requires_grad_(False)
    dataset_root = Path(cfg["experiment"]["dataset_output"]); input_contract = json.loads((dataset_root / "input_contract.json").read_text()); canonical_contract = json.loads(Path("results/structural_dataset_v3/input_contract.json").read_text()); contract_audit = {"exact_contract_json_match": input_contract == canonical_contract, "shape": input_contract["shape"], "contract_version": input_contract["contract_version"], "model_input_channel_indices": channels, "model_input_channel_names": [input_contract["channels"][index]["name"] for index in channels], "same_builder": True, "builder_class": "learning.structural_learning.surface_evidence.SurfaceEvidenceBuilder", "encoder_checkpoint_sha256": hashlib.sha256(base_checkpoint_path.read_bytes()).hexdigest(), "projection_checkpoint_sha256": hashlib.sha256(projection_checkpoint_path.read_bytes()).hexdigest(), "model_parameters_changed": False}
    write_json(root / "input_contract_audit" / "summary.json", contract_audit)
    results = {}
    for world, world_cfg in cfg["worlds"].items():
        samples = load_world(dataset_root, world); inputs = np.stack([sample["input"] for sample in samples]); teachers = np.stack([sample["teacher"] for sample in samples]); latent = represent(base, projection, inputs, channels, device); input_descriptor = surface_descriptors(inputs, device); teacher_descriptor = surface_descriptors(teachers, device); results[world] = evaluate_world(world, samples, latent, input_descriptor, teacher_descriptor, cfg, root, base, projection, device)
        recording = Path(world_cfg["recording_root"]); stats = json.loads((dataset_root / world / "stats.json").read_text()); world_audit = {"world": world, "recording_root": str(recording), "bag": world_cfg["bag"], "recording_status": json.loads((recording / "status.json").read_text()), "recording_config": yaml.safe_load((recording / "config.yaml").read_text()), "recording_container_image": json.loads((recording / "runtime" / "docker_image.json").read_text()), "world_asset_sha256": (recording / "world_audit" / "assets.sha256").read_text().splitlines(), "rosbag_info": (recording / "world_audit" / "rosbag_info.txt").read_text(), "dataset_stats": stats, "training_or_supervision_use": False}
        write_json(root / "world_audit" / f"{world}.json", world_audit)
    pooled = pooled_world_audit(results); write_json(root / "nuisance_factor_metrics" / "cross_world.json", pooled)
    lamp_run = Path(cfg["experiment"]["lamp_reference_run"]); lamp_final = json.loads((lamp_run / "final_metrics" / "summary.json").read_text()); lamp_stable = json.loads((lamp_run / "stable_segment_audit" / "summary.json").read_text()); lamp_change = json.loads((lamp_run / "structural_change_audit" / "summary.json").read_text()); lamp_cross = json.loads((lamp_run / "cross_location_similarity" / "summary.json").read_text()); lamp_nuisance = json.loads((lamp_run / "nuisance_factor_audit" / "summary.json").read_text())
    lamp_perturbation = json.loads((lamp_run / "final_metrics" / "perturbation.json").read_text())
    comparison: dict[str, Any] = {"LAMP_validation": {"samples": 268, "stable_distance": lamp_change["latent_distance_test"]["stable"], "change_distance": lamp_change["latent_distance_test"]["changed"], "change_stable_ratio": lamp_change["latent_distance_test"]["median_ratio_changed_over_stable"], "similar_structure_distance": lamp_cross["similar_structure_pairs"], "different_structure_distance": lamp_cross["different_structure_pairs"], "retrieval": lamp_cross["retrieval"], "nuisance_spearman": lamp_nuisance["spearman"], "perturbation": lamp_perturbation, "input_statistics": lamp_input_statistics(Path("results/structural_dataset_v3"))}}
    for world, result in results.items():
        comparison[world] = {"samples": result["stable_change"]["samples"], "stable_distance": result["stable_change"]["latent_distance"]["stable"], "change_distance": result["stable_change"]["latent_distance"]["changed"], "change_stable_ratio": result["stable_change"]["latent_distance"]["median_ratio_changed_over_stable"], "similar_structure_distance": result["cross"]["similar_structure_distance"], "different_structure_distance": result["cross"]["different_structure_distance"], "nuisance_spearman": result["nuisance"]["spearman"], "input_statistics": result["nuisance"]["input_statistics"]}
    write_json(root / "cross_world_comparison" / "summary.json", comparison)
    acceptance = {}; all_world_pass = True
    for world, result in results.items():
        change = result["stable_change"]["latent_distance"]; cross = result["cross"]["distance_test"]; nuisance = result["nuisance"]["spearman"]; perturb = result["perturbation"]
        change_pass = change["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_change_to_stable_ratio"]) and change["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
        cross_pass = cross["median_ratio_changed_over_stable"] >= float(cfg["acceptance"]["minimum_cross_location_ratio"]) and cross["one_sided_p_value"] <= float(cfg["acceptance"]["maximum_p_value"])
        nuisance_pass = nuisance["teacher_geometry_difference"] > max(abs(nuisance["input_coverage_shape_difference"]), abs(nuisance["surface_cell_count_difference"]), abs(nuisance["log_density_proxy_difference"]), abs(nuisance["raw_point_count_difference"]))
        perturb_pass = all(value["same_sample"]["median"] < value["different_structure_reference"]["median"] for value in perturb.values()); acceptance[world] = {"stable_change": change_pass, "cross_location": cross_pass, "geometry_dominates_observation_nuisance": nuisance_pass, "perturbation": perturb_pass}; all_world_pass &= all(acceptance[world].values())
    cross_world_geometry = pooled["spearman"]["teacher_geometry_difference"]; cross_world_nuisance = max(abs(pooled["spearman"][name]) for name in ("input_coverage_shape_difference", "surface_cell_count_difference", "log_density_proxy_difference", "raw_point_count_difference", "world_identity_difference")); identity_pass = cross_world_geometry > cross_world_nuisance; acceptance["cross_world"] = {"geometry_dominates_world_and_observation_nuisance": identity_pass}
    all_pass = all_world_pass and identity_pass and contract_audit["exact_contract_json_match"] and contract_audit["model_parameters_changed"] is False
    partial = sum(all(item.values()) for world, item in acceptance.items() if world in results) >= 1
    status = "MTARE_TRANSFER_PASS" if all_pass else "MTARE_TRANSFER_MIXED" if partial else "MTARE_TRANSFER_FAIL"
    summary = {"status": status, "worlds": list(results), "acceptance": acceptance, "frozen_model": True, "supervised_mtare_training": False, "samples": {world: result["stable_change"]["samples"] for world, result in results.items()}, "cross_world_nuisance": pooled, "environment": {"hostname": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
    write_json(root / "cross_world_comparison" / "final_status.json", summary); print(json.dumps({"result_dir": str(root), "status": status, "acceptance": acceptance}, indent=2))


if __name__ == "__main__": main()
