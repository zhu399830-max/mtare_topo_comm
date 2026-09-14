"""Export a ray-free LAMP surface-geometry dataset and verify the M-TARE path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

from learning.local_structural_map.datasets.mtare_bag import MTAREBagAdapter
from learning.local_structural_map.schema import Pose2D, StandardFrame
from learning.structural_learning.surface_evidence import SurfaceEvidenceBuilder, SurfaceEvidenceConfig
from learning.structural_learning.tools.diagnose_lamp_keyed_dataset import (
    Center, audit_and_load_robot, make_standard_frames, select_centers,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"count": 0}
    return {"count": int(len(a)), "min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def pose7(pose: Pose2D) -> np.ndarray:
    return np.asarray([pose.x, pose.y, pose.z, 0.0, 0.0, math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)], dtype=np.float32)


def config_digest(cfg: Dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def assign_spatial_splits(centers: List[Center], target_train: float, buffer_m: float) -> Dict[str, object]:
    """One spatial x-axis partition with an explicit 20m discarded corridor."""
    xs = np.asarray([c.pose.x for c in centers], dtype=np.float32)
    cut = float(np.quantile(xs, target_train))
    lo, hi = cut - buffer_m / 2, cut + buffer_m / 2
    for c in centers:
        c.split = "train" if c.pose.x < lo else "val" if c.pose.x > hi else "buffer"
        c.region_id = "train_x" if c.split == "train" else "val_x" if c.split == "val" else "buffer_x"
    return {"method": "world_x_partition_with_discarded_buffer", "cut_x_m": cut, "buffer_x_interval_m": [lo, hi], "buffer_m": buffer_m, "test_policy": "not exported: single tunnel has no independent third region"}


def frame_split(frame: StandardFrame, split_info: Dict[str, object]) -> str:
    lo, hi = split_info["buffer_x_interval_m"]
    return "train" if frame.pose.x < lo else "val" if frame.pose.x > hi else "buffer"


def teacher_sources(frames_by_robot: Dict[str, List[StandardFrame]], center: Center, split_info: Dict[str, object], cfg: Dict[str, object]) -> List[Tuple[str, int, float]]:
    radius, max_frames, bin_m = float(cfg["source_radius_m"]), int(cfg["max_frames"]), float(cfg["spatial_bin_m"])
    candidates = []
    cxy = np.asarray([center.pose.x, center.pose.y], dtype=np.float32)
    for robot, frames in frames_by_robot.items():
        for idx, frame in enumerate(frames):
            if robot == center.robot and idx == center.frame_index:
                continue
            if frame_split(frame, split_info) != center.split:
                continue
            d = float(np.linalg.norm(np.asarray([frame.pose.x, frame.pose.y]) - cxy))
            if d <= radius:
                bucket = (int(math.floor((frame.pose.x - center.pose.x) / bin_m)), int(math.floor((frame.pose.y - center.pose.y) / bin_m)))
                candidates.append((bucket, d, robot, idx))
    bins: Dict[Tuple[int, int], List[Tuple[float, str, int]]] = defaultdict(list)
    for bucket, distance, robot, idx in candidates:
        bins[bucket].append((distance, robot, idx))
    chosen = []
    for values in bins.values():
        chosen.append(min(values, key=lambda item: item[0]))
    used = {(r, i) for _, r, i in chosen}
    chosen.extend(item for _, item in sorted((d, (d, r, i)) for _, d, r, i in candidates if (r, i) not in used))
    return [(robot, idx, distance) for distance, robot, idx in chosen[:max_frames]]


def surface_metrics(inp: np.ndarray, teacher: np.ndarray, inp_points: np.ndarray, teacher_points: np.ndarray, resolution: float) -> Dict[str, float]:
    imask, tmask = inp[0] > 0.5, teacher[0] > 0.5
    support = ndimage.binary_dilation(tmask, iterations=1)
    input_cells = int(imask.sum())
    teacher_cells = int(tmask.sum())
    supported = int((imask & support).sum())
    conflict = int((imask & ~support).sum())
    metric = {
        "input_surface_cells": input_cells,
        "teacher_surface_cells": teacher_cells,
        "input_supported_cells": supported,
        "input_unsupported_cells": conflict,
        "input_teacher_support_ratio": float(supported / max(input_cells, 1)),
        "input_teacher_conflict_ratio": float(conflict / max(input_cells, 1)),
        "teacher_new_surface_cells": int((tmask & ~ndimage.binary_dilation(imask, iterations=1)).sum()),
        "teacher_new_surface_ratio": float((tmask & ~ndimage.binary_dilation(imask, iterations=1)).sum() / max(teacher_cells, 1)),
    }
    if len(inp_points) and len(teacher_points):
        d = cKDTree(teacher_points).query(inp_points, k=1)[0]
        metric["input_to_teacher_nn_median_m"] = float(np.median(d))
        metric["input_to_teacher_nn_p90_m"] = float(np.percentile(d, 90))
    else:
        metric["input_to_teacher_nn_median_m"] = float("inf")
        metric["input_to_teacher_nn_p90_m"] = float("inf")
    return metric


def save_preview(path: Path, inp: np.ndarray, teacher: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, tensor, name in zip(axes[:2], (inp, teacher), ("online input", "multi-view teacher")):
        rgb = np.zeros((tensor.shape[1], tensor.shape[2], 3), dtype=np.float32)
        rgb[..., 1] = tensor[0]
        rgb[..., 2] = tensor[2] * tensor[0]
        rgb[..., 0] = tensor[3] * tensor[0]
        ax.imshow(rgb, origin="upper")
        ax.set_title(name)
        ax.set_axis_off()
    axes[2].imshow(np.maximum(teacher[0] - inp[0], 0), origin="upper", cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("teacher-only surface evidence")
    axes[2].set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_diagnostics(centers: List[Center], count: int) -> List[Center]:
    usable = [c for c in centers if c.split in {"train", "val"}]
    order = np.argsort([c.pose.x + 0.37 * c.pose.y for c in usable])
    return [usable[int(order[i])] for i in np.linspace(0, len(order) - 1, min(count, len(order)), dtype=int)]


def make_sample(center: Center, frames_by_robot: Dict[str, List[StandardFrame]], split_info: Dict[str, object], builder: SurfaceEvidenceBuilder, cfg: Dict[str, object]):
    current = frames_by_robot[center.robot][center.frame_index]
    teacher_refs = teacher_sources(frames_by_robot, center, split_info, cfg["teacher"])
    teacher_frames = [frames_by_robot[robot][idx] for robot, idx, _ in teacher_refs]
    input_tensor, input_local = builder.build([current.points_world], center.pose)
    teacher_tensor, teacher_local = builder.build([frame.points_world for frame in teacher_frames], center.pose)
    metrics = surface_metrics(input_tensor, teacher_tensor, input_local, teacher_local, builder.config.resolution_m)
    source_meta = [{"robot": robot, "key": int(frames_by_robot[robot][idx].metadata["key"]), "frame_index": idx, "distance_to_center_m": d, "split": frame_split(frames_by_robot[robot][idx], split_info)} for robot, idx, d in teacher_refs]
    return current, input_tensor, teacher_tensor, input_local, teacher_local, teacher_refs, source_meta, metrics


def validate_mtare(builder: SurfaceEvidenceBuilder, cfg: Dict[str, object], out: Path) -> Dict[str, object]:
    adapter = MTAREBagAdapter(cfg["dataset"]["mtare_bag"], max_sync_dt_sec=float(cfg["mtare"]["max_sync_dt_sec"]), max_samples=int(cfg["mtare"]["sample_count"]), stride=int(cfg["mtare"]["stride"]), sync_policy="exact_stamp")
    samples = list(adapter)
    info = {"requested": int(cfg["mtare"]["sample_count"]), "emitted": len(samples), "stride": int(cfg["mtare"]["stride"]), "same_builder": True, "bag": cfg["dataset"]["mtare_bag"], "samples": []}
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(samples):
        tensor, local = builder.build([frame.points_world], frame.pose)
        np.savez_compressed(out / f"mtare_{i:04d}.npz", input_surface=tensor, center_pose=pose7(frame.pose), stamp_ns=np.int64(frame.timestamp_ns))
        info["samples"].append({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": bool(np.isfinite(tensor).all()), "range": [float(tensor.min()), float(tensor.max())], "surface_cells": int((tensor[0] > 0.5).sum()), "raw_points": int(len(frame.points_world)), "local_points": int(len(local)), "sync_dt_ns": int(frame.metadata["sync_dt_ns"])})
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/structural_dataset_v3.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["dataset"]["output_dir"])
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {root}")
    root.mkdir(parents=True)
    for name in ("train", "val", "logs", "previews", "diagnostics", "mtare_samples"):
        (root / name).mkdir()
    shutil.copy2(args.config, root / "logs" / "config.yaml")
    evidence_cfg = SurfaceEvidenceConfig(**cfg["input"])
    builder = SurfaceEvidenceBuilder(evidence_cfg)
    capability = {
        "reliable_online_input": ["KeyedScan surface points after exact KeyedScan.key -> final PoseGraphNode.key association", "final optimized pose applied exactly once", "local surface projection: mask, capped density, mean height, height span"],
        "reliable_training_teacher": ["same-split multi-view fusion of final-pose KeyedScan surface points", "surface correspondence and teacher-only surface coverage after local crop"],
        "unreliable_or_unavailable": ["per-beam sensor origins for keyed accumulated scans", "raycast-derived free/unknown", "occupancy/free reachability labels", "complete tunnel.pcd as model input or free-space source"],
        "tunnel_pcd_policy": "optional geometric inspection only; not used to construct input, teacher, or labels",
    }
    write_json(root / "diagnostics" / "data_capability_audit.json", capability)
    input_contract = {"contract_version": evidence_cfg.contract_version, "shape": [4, evidence_cfg.grid_size, evidence_cfg.grid_size], "dtype": "float32", "coordinates": {"origin": "robot center pose", "x": "forward", "y": "left", "z": "up", "tensor_rows": "left to right y descending", "tensor_cols": "rear to forward x ascending"}, "range_m": {"x": [-10, 10], "y": [-10, 10], "z": [evidence_cfg.z_min_m, evidence_cfg.z_max_m]}, "resolution_m": evidence_cfg.resolution_m, "channels": [{"name": "surface_mask", "range": [0, 1], "missing": 0}, {"name": "log_density", "range": [0, 1], "definition": "log1p(min(points_per_cell, density_cap))/log1p(density_cap)", "missing": 0}, {"name": "mean_height", "range": [0, 1], "definition": "mean z normalized over configured z range", "missing": 0}, {"name": "height_span", "range": [0, 1], "definition": "per-cell z max-min normalized over configured z range", "missing": 0}], "preprocessing": ["world points -> center-pose local coordinates", "crop local xy/z bounds", "2D grid aggregation", "no raycasting and no free/unknown inference"]}
    supervision = {"contract_version": evidence_cfg.contract_version, "model_input": "one current online point cloud only", "training_teacher": "same-split, final-pose, spatially distributed nearby KeyedScans excluding the input scan, fused in the same local center frame", "metadata_only": ["robot", "key", "timestamp", "center pose", "split", "teacher source keys/distances"], "physical_basis": "stable surfaces seen from multiple viewpoints should recur in the local coordinate frame; no claim about unobserved space", "known_limitations": ["teacher inherits LAMP final-pose/map consistency", "single tunnel limits structural diversity", "teacher is not a free-space map", "LAMP and M-TARE point density/sensor models differ"]}
    write_json(root / "input_contract.json", input_contract); write_json(root / "supervision_contract.json", supervision)
    lamp_root = Path(cfg["dataset"]["lamp_root"])
    frames_by_robot, audits = {}, {}
    for robot in cfg["dataset"]["robots"]:
        scans, poses, audit = audit_and_load_robot(lamp_root, robot, 1000.0)
        frames, standard_audit = make_standard_frames(scans, poses, 1000.0)
        frames_by_robot[robot] = frames; audits[robot] = {"key_association": audit, "standard_frames": standard_audit}
    write_json(root / "logs" / "lamp_key_pose_audit.json", audits)
    centers = select_centers(frames_by_robot, {"sampling": {"min_center_translation_m": cfg["sampling"]["min_center_translation_m"], "min_center_yaw_deg": cfg["sampling"]["min_center_yaw_deg"], "repeat_neighbor_radius_m": 2.0}})
    split_info = assign_spatial_splits(centers, float(cfg["split"]["target_train"]), float(cfg["split"]["buffer_m"]))
    distance = None
    train_xy = np.asarray([[c.pose.x, c.pose.y] for c in centers if c.split == "train"], dtype=np.float32)
    val_xy = np.asarray([[c.pose.x, c.pose.y] for c in centers if c.split == "val"], dtype=np.float32)
    if len(train_xy) and len(val_xy): distance = float(cKDTree(train_xy).query(val_xy, k=1)[0].min())
    write_json(root / "split_regions.json", {**split_info, "counts": dict(Counter(c.split for c in centers)), "train_val_nearest_center_m": distance, "repeat_location_policy": "spatial partition assigns nearby repeated visits to the same split or buffer"})
    diagnostics, discards = [], Counter()
    for center in select_diagnostics(centers, int(cfg["sampling"]["diagnostic_count"])):
        current, inp, teacher, inp_pts, teacher_pts, refs, meta, metrics = make_sample(center, frames_by_robot, split_info, builder, cfg)
        if len(refs) < int(cfg["teacher"]["min_frames"]): discards["diagnostic_teacher_too_few"] += 1; continue
        diagnostics.append({"robot": center.robot, "key": center.key, "frame_index": center.frame_index, "split": center.split, "center_pose": pose7(center.pose).tolist(), "teacher_sources": meta, "metrics": metrics})
        save_preview(root / "previews" / f"diagnostic_{center.split}_{center.robot}_{center.frame_index:04d}.png", inp, teacher, f"{center.split} {center.robot} key={center.key}")
    write_json(root / "diagnostics" / "diagnostic_samples.json", diagnostics)
    support = [d["metrics"]["input_teacher_support_ratio"] for d in diagnostics]; nn = [d["metrics"]["input_to_teacher_nn_median_m"] for d in diagnostics]; conflict = [d["metrics"]["input_teacher_conflict_ratio"] for d in diagnostics]
    gates = {"diagnostic_count": len(diagnostics), "support_ratio": describe(support), "nn_median_m": describe(nn), "conflict_ratio": describe(conflict), "pass": bool(diagnostics and np.median(support) >= cfg["quality"]["min_input_teacher_support_ratio"] and np.median(nn) <= cfg["quality"]["max_input_teacher_nn_median_m"] and np.median(conflict) <= cfg["quality"]["max_input_teacher_conflict_ratio"] and (distance is None or distance >= 20.0))}
    write_json(root / "diagnostics" / "geometric_consistency.json", gates)
    if not gates["pass"]:
        write_json(root / "dataset_stats.json", {"status": "NOT_READY", "reason": "diagnostic geometric gates failed", "gates": gates, "discard_reasons": dict(discards)})
        return 2
    samples, export_discards = [], Counter()
    for center in centers:
        if center.split not in {"train", "val"}: continue
        current, inp, teacher, inp_pts, teacher_pts, refs, meta, metrics = make_sample(center, frames_by_robot, split_info, builder, cfg)
        if len(refs) < int(cfg["teacher"]["min_frames"]): export_discards["teacher_too_few_frames"] += 1; continue
        if metrics["input_surface_cells"] < int(cfg["quality"]["min_input_surface_cells"]): export_discards["input_too_sparse"] += 1; continue
        if metrics["teacher_surface_cells"] < int(cfg["quality"]["min_teacher_surface_cells"]): export_discards["teacher_too_sparse"] += 1; continue
        if metrics["input_teacher_support_ratio"] < float(cfg["quality"]["min_input_teacher_support_ratio"]): export_discards["input_teacher_unsupported"] += 1; continue
        if metrics["input_to_teacher_nn_median_m"] > float(cfg["quality"]["max_input_teacher_nn_median_m"]): export_discards["input_teacher_misaligned"] += 1; continue
        sample_id = f"{center.split}_{center.robot}_{center.frame_index:04d}"
        np.savez_compressed(root / center.split / f"{sample_id}.npz", input_surface=inp, teacher_surface=teacher, center_pose=pose7(center.pose), stamp_ns=np.int64(current.timestamp_ns), robot=np.asarray(center.robot), key=np.int64(center.key), region_id=np.asarray(center.region_id), split=np.asarray(center.split), input_source_key=np.int64(center.key), teacher_source_keys=np.asarray([m["key"] for m in meta], dtype=np.int64), teacher_source_robots=np.asarray([m["robot"] for m in meta]), teacher_source_distances_m=np.asarray([m["distance_to_center_m"] for m in meta], dtype=np.float32), geometry_metrics=np.asarray([metrics["input_teacher_support_ratio"], metrics["input_to_teacher_nn_median_m"], metrics["teacher_new_surface_ratio"], metrics["input_teacher_conflict_ratio"]], dtype=np.float32), contract_version=np.asarray(evidence_cfg.contract_version), config_hash=np.asarray(config_digest(cfg)))
        samples.append({"id": sample_id, "split": center.split, "robot": center.robot, "metrics": metrics, "teacher_frames": len(refs), "input_points": len(inp_pts), "teacher_points": len(teacher_pts)})
    mtare = validate_mtare(builder, cfg, root / "mtare_samples")
    if mtare["emitted"] < int(cfg["mtare"]["sample_count"]): raise RuntimeError("M-TARE contract validation did not yield required samples")
    stats = {"status": "DATASET_READY_WITH_LIMITATIONS", "raw_keyed_scans": int(sum(a["key_association"]["keyed_scan_total"] for a in audits.values())), "valid_standard_frames": int(sum(a["standard_frames"]["valid_standard_frames"] for a in audits.values())), "candidate_centers": len(centers), "samples": len(samples), "split_counts": dict(Counter(s["split"] for s in samples)), "robot_counts": dict(Counter(s["robot"] for s in samples)), "buffer_centers": int(sum(c.split == "buffer" for c in centers)), "train_val_nearest_center_m": distance, "geometric_consistency": gates, "export_metrics": {"support_ratio": describe(s["metrics"]["input_teacher_support_ratio"] for s in samples), "nn_median_m": describe(s["metrics"]["input_to_teacher_nn_median_m"] for s in samples), "teacher_new_surface_ratio": describe(s["metrics"]["teacher_new_surface_ratio"] for s in samples), "conflict_ratio": describe(s["metrics"]["input_teacher_conflict_ratio"] for s in samples)}, "teacher_cross_split_violations": 0, "discard_reasons": dict(export_discards), "mtare_contract_validation": mtare, "limitations": supervision["known_limitations"]}
    write_json(root / "dataset_stats.json", stats)
    write_json(root / "manifest.json", {"contract_version": evidence_cfg.contract_version, "config_hash": config_digest(cfg), "samples": samples, "data_capability_audit": "diagnostics/data_capability_audit.json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
