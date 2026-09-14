from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "min": float(values.min()), "mean": float(values.mean()),
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(values.max()),
    }


def pose7(pose: Pose2D) -> np.ndarray:
    return np.asarray([pose.x, pose.y, pose.z, 0.0, 0.0, math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)], dtype=np.float32)


def choose_teacher_sources(frames: list[StandardFrame], center: int, radius: float, maximum: int, bin_m: float) -> list[int]:
    position = np.asarray([frames[center].pose.x, frames[center].pose.y], dtype=np.float64)
    candidates = []
    for index, frame in enumerate(frames):
        if index == center:
            continue
        xy = np.asarray([frame.pose.x, frame.pose.y], dtype=np.float64)
        distance = float(np.linalg.norm(xy - position))
        if distance <= radius:
            bucket = tuple(np.floor((xy - position) / bin_m).astype(int))
            candidates.append((index, distance, bucket, xy))
    by_bucket: dict[tuple[int, int], tuple[int, float, np.ndarray]] = {}
    for index, distance, bucket, xy in candidates:
        previous = by_bucket.get(bucket)
        if previous is None or distance < previous[1]:
            by_bucket[bucket] = (index, distance, xy)
    selected = [item[0] for item in sorted(by_bucket.values(), key=lambda item: item[1])]
    selected = selected[:maximum]
    if len(selected) < maximum:
        used = set(selected)
        remaining = [item for item in candidates if item[0] not in used]
        remaining.sort(key=lambda item: (abs(item[0] - center), item[1]))
        selected.extend(item[0] for item in remaining[: maximum - len(selected)])
    return sorted(set(selected), key=lambda index: frames[index].timestamp_ns)


def geometry_metrics(input_tensor: np.ndarray, teacher_tensor: np.ndarray, input_points: np.ndarray, teacher_points: np.ndarray) -> dict[str, float | int]:
    input_mask = input_tensor[0] > 0.5; teacher_mask = teacher_tensor[0] > 0.5
    support = ndimage.binary_dilation(teacher_mask, iterations=1)
    supported = int((input_mask & support).sum()); input_cells = int(input_mask.sum()); teacher_cells = int(teacher_mask.sum())
    result: dict[str, float | int] = {
        "input_surface_cells": input_cells, "teacher_surface_cells": teacher_cells,
        "input_teacher_support_ratio": float(supported / max(input_cells, 1)),
        "teacher_new_surface_ratio": float((teacher_mask & ~ndimage.binary_dilation(input_mask, iterations=1)).sum() / max(teacher_cells, 1)),
    }
    if len(input_points) and len(teacher_points):
        distances = cKDTree(teacher_points).query(input_points, k=1)[0]
        result["input_to_teacher_nn_median_m"] = float(np.median(distances))
        result["input_to_teacher_nn_p90_m"] = float(np.percentile(distances, 90))
    else:
        result["input_to_teacher_nn_median_m"] = float("inf"); result["input_to_teacher_nn_p90_m"] = float("inf")
    return result


def save_preview(path: Path, input_tensor: np.ndarray, teacher_tensor: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(input_tensor[0], origin="upper", cmap="gray", vmin=0, vmax=1); axes[0].set_title("online single scan")
    axes[1].imshow(teacher_tensor[0], origin="upper", cmap="gray", vmin=0, vmax=1); axes[1].set_title("offline multi-view teacher")
    axes[2].imshow(np.maximum(teacher_tensor[0] - input_tensor[0], 0), origin="upper", cmap="magma", vmin=0, vmax=1); axes[2].set_title("teacher-only surface")
    for axis in axes: axis.set_axis_off()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def export_world(world: str, world_cfg: dict[str, Any], cfg: dict[str, Any], builder: SurfaceEvidenceBuilder, root: Path) -> dict[str, Any]:
    export_cfg = cfg["export"]; bag = Path(world_cfg["bag"])
    frames = list(MTAREBagAdapter(bag, max_sync_dt_sec=float(export_cfg["max_sync_dt_sec"]), stride=int(export_cfg["raw_frame_stride"]), sync_policy="exact_stamp"))
    if not frames: raise RuntimeError(f"no synchronized frames for {world}")
    positions = np.asarray([[frame.pose.x, frame.pose.y, frame.pose.z] for frame in frames], dtype=np.float64)
    initial = positions[0, :2]; distance_from_start = np.linalg.norm(positions[:, :2] - initial, axis=1)
    eligible = np.flatnonzero(distance_from_start >= float(export_cfg["minimum_motion_from_start_m"]))
    if not len(eligible): raise RuntimeError(f"{world} never moved from start")
    first_moving = int(eligible[0]); candidates = list(range(first_moving, len(frames)))
    world_dir = root / world; sample_dir = world_dir / "samples"; preview_dir = root / "previews" / world
    sample_dir.mkdir(parents=True); preview_dir.mkdir(parents=True)
    records = []; discards: dict[str, int] = {}; preview_indices = set(np.linspace(0, max(len(candidates) - 1, 0), int(export_cfg["preview_count_per_world"]), dtype=int).tolist())
    for candidate_number, center in enumerate(candidates):
        frame = frames[center]
        input_tensor, input_local = builder.build([frame.points_world], frame.pose)
        if int((input_tensor[0] > 0.5).sum()) < int(export_cfg["minimum_input_surface_cells"]):
            discards["input_too_sparse"] = discards.get("input_too_sparse", 0) + 1; continue
        sources = choose_teacher_sources(frames, center, float(export_cfg["teacher_radius_m"]), int(export_cfg["teacher_max_frames"]), float(export_cfg["teacher_spatial_bin_m"]))
        if len(sources) < int(export_cfg["teacher_min_frames"]):
            discards["teacher_too_few_sources"] = discards.get("teacher_too_few_sources", 0) + 1; continue
        teacher_tensor, teacher_local = builder.build([frames[index].points_world for index in sources], frame.pose)
        if int((teacher_tensor[0] > 0.5).sum()) < int(export_cfg["minimum_teacher_surface_cells"]):
            discards["teacher_too_sparse"] = discards.get("teacher_too_sparse", 0) + 1; continue
        metrics = geometry_metrics(input_tensor, teacher_tensor, input_local, teacher_local)
        sample_index = len(records); path = sample_dir / f"{world}_{sample_index:04d}.npz"
        source_distance = [float(np.linalg.norm(positions[index, :2] - positions[center, :2])) for index in sources]
        np.savez_compressed(
            path, input_surface=input_tensor, teacher_surface=teacher_tensor, center_pose=pose7(frame.pose), stamp_ns=np.int64(frame.timestamp_ns),
            world=np.asarray(world), source_frame_index=np.int64(center), teacher_source_indices=np.asarray(sources, dtype=np.int64),
            teacher_source_stamps_ns=np.asarray([frames[index].timestamp_ns for index in sources], dtype=np.int64),
            teacher_source_distances_m=np.asarray(source_distance, dtype=np.float32), raw_point_count=np.int64(len(frame.points_world)),
            input_surface_cells=np.int64(metrics["input_surface_cells"]), teacher_surface_cells=np.int64(metrics["teacher_surface_cells"]),
            contract_version=np.asarray(builder.config.contract_version), builder_class=np.asarray("learning.structural_learning.surface_evidence.SurfaceEvidenceBuilder"),
        )
        record = {"sample_index": sample_index, "source_frame_index": center, "stamp_ns": frame.timestamp_ns, "pose": pose7(frame.pose).tolist(), "teacher_sources": len(sources), "metrics": metrics}
        records.append(record)
        if candidate_number in preview_indices:
            save_preview(preview_dir / f"{world}_{sample_index:04d}.png", input_tensor, teacher_tensor, f"{world} sample {sample_index}")
    if len(records) < int(export_cfg["minimum_samples_per_world"]):
        raise RuntimeError(f"{world} yielded only {len(records)} samples")
    support = [record["metrics"]["input_teacher_support_ratio"] for record in records]; nn = [record["metrics"]["input_to_teacher_nn_median_m"] for record in records]
    stats = {
        "world": world, "bag": str(bag), "bag_bytes": bag.stat().st_size, "adapter": "MTAREBagAdapter", "builder": "SurfaceEvidenceBuilder",
        "contract_version": builder.config.contract_version, "raw_frame_stride": int(export_cfg["raw_frame_stride"]), "synchronized_subsampled_frames": len(frames),
        "first_moving_frame": first_moving, "candidate_samples": len(candidates), "exported_samples": len(records), "discard_reasons": discards,
        "frame_id_values": sorted({frame.frame_id for frame in frames}), "sync_dt_ns": describe([frame.metadata["sync_dt_ns"] for frame in frames]),
        "raw_point_count": describe([len(frame.points_world) for frame in frames]), "path_length_m": float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
        "displacement_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])), "xyz_min": positions.min(axis=0).tolist(), "xyz_max": positions.max(axis=0).tolist(),
        "input_teacher_support_ratio": describe(support), "input_to_teacher_nn_median_m": describe(nn),
        "teacher_source_count": describe([record["teacher_sources"] for record in records]),
    }
    write_json(world_dir / "manifest.json", {"world": world, "samples": records}); write_json(world_dir / "stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/learning/mtare_unseen_transfer.yaml"); args = parser.parse_args()
    config_path = Path(args.config); cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")); root = Path(cfg["experiment"]["dataset_output"])
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True); (root / "config").mkdir(); shutil.copy2(config_path, root / "config" / "config.yaml")
    evidence_cfg = SurfaceEvidenceConfig(**cfg["surface_evidence"]); builder = SurfaceEvidenceBuilder(evidence_cfg)
    input_contract = Path("results/structural_dataset_v3/input_contract.json"); shutil.copy2(input_contract, root / "input_contract.json")
    stats = {world: export_world(world, world_cfg, cfg, builder, root) for world, world_cfg in cfg["worlds"].items()}
    digest = hashlib.sha256(json.dumps(cfg["surface_evidence"], sort_keys=True).encode()).hexdigest()
    summary = {"status": "EXPORTED", "worlds": stats, "same_builder_for_all_worlds": True, "builder_config_sha256": digest, "teacher_policy": "offline same-trajectory multiview surface only; no free/unknown and no raycasting", "model_training_use": "forbidden"}
    write_json(root / "dataset_stats.json", summary); print(json.dumps({"output": str(root), "samples": {world: value["exported_samples"] for world, value in stats.items()}}, indent=2))


if __name__ == "__main__": main()
