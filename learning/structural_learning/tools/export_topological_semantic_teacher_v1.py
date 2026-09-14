from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import ndimage
from scipy.stats import rankdata
from skimage.morphology import binary_closing, disk, skeletonize
from torch.nn import functional as F

from learning.local_structural_map.schema import Pose2D
from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.semantic_projection import StructuralProjectionHead
from learning.structural_learning.surface_evidence import SurfaceEvidenceBuilder, SurfaceEvidenceConfig
from learning.structural_learning.tools.run_structural_semantic_validation import surface_descriptors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.sum(a * b, axis=-1), 0.0, 2.0)


def load_binary_ply_xyz(path: Path, max_points: int, rng: np.random.Generator) -> np.ndarray:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"bad ply header: {path}")
            header.append(line.decode("utf-8", "ignore").strip())
            if header[-1] == "end_header":
                break
        count_line = next(line for line in header if line.startswith("element vertex"))
        count = int(count_line.split()[-1])
        data = np.fromfile(f, dtype="<f4", count=count * 3).reshape(-1, 3)
    finite = np.isfinite(data).all(axis=1)
    data = data[finite]
    if len(data) > max_points:
        data = data[rng.choice(len(data), size=max_points, replace=False)]
    return data.astype(np.float32)


@dataclass
class WorldGrid:
    world: str
    points: np.ndarray
    origin_xy: np.ndarray
    resolution: float
    traversable: np.ndarray
    occupied: np.ndarray
    inflated_obstacle: np.ndarray
    ground_support: np.ndarray
    distance_to_obstacle_m: np.ndarray

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - self.origin_xy.reshape(1, 2)) / self.resolution).astype(np.int64)

    def grid_to_world(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.float64)
        return np.c_[self.origin_xy[0] + (rc[:, 1] + 0.5) * self.resolution, self.origin_xy[1] + (rc[:, 0] + 0.5) * self.resolution]


def build_world_grid(world: str, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[WorldGrid, dict[str, Any]]:
    mesh_root = Path(cfg["experiment"]["mtare_mesh_root"])
    points = load_binary_ply_xyz(mesh_root / world / "preview" / "pointcloud.ply", int(cfg["map"]["max_points_per_world"]), rng)
    res = float(cfg["map"]["resolution_m"])
    margin = float(cfg["teacher"]["local_size_m"])
    min_xy = points[:, :2].min(axis=0) - margin
    max_xy = points[:, :2].max(axis=0) + margin
    size = np.ceil((max_xy - min_xy) / res).astype(int) + 1
    rc = np.floor((points[:, :2] - min_xy) / res).astype(np.int64)
    valid = (rc[:, 0] >= 0) & (rc[:, 0] < size[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < size[1])
    points = points[valid]
    rc = rc[valid]
    flat = rc[:, 1] * size[0] + rc[:, 0]
    n = int(size[0] * size[1])
    zmin = np.full(n, np.inf, dtype=np.float32)
    zmax = np.full(n, -np.inf, dtype=np.float32)
    count = np.bincount(flat, minlength=n)
    np.minimum.at(zmin, flat, points[:, 2])
    np.maximum.at(zmax, flat, points[:, 2])
    zmin = zmin.reshape(size[1], size[0])
    zmax = zmax.reshape(size[1], size[0])
    count = count.reshape(size[1], size[0])
    ground_support = count >= int(cfg["map"]["min_ground_support_points"])
    height_span = np.where(ground_support, zmax - zmin, 0.0)
    occupied = ground_support & (height_span > float(cfg["map"]["obstacle_height_threshold_m"]))
    closed_ground = binary_closing(ground_support, disk(2))
    radius = float(cfg["map"]["base_collision_radius_m"]) + float(cfg["map"]["safety_margin_m"])
    radius_cells = max(1, int(math.ceil(radius / res)))
    inflated = ndimage.binary_dilation(occupied, structure=disk(radius_cells))
    traversable = closed_ground & ~inflated
    distance_to_obstacle = ndimage.distance_transform_edt(~inflated) * res
    grid = WorldGrid(
        world=world,
        points=points,
        origin_xy=min_xy.astype(np.float64),
        resolution=res,
        traversable=traversable.astype(bool),
        occupied=occupied.astype(bool),
        inflated_obstacle=inflated.astype(bool),
        ground_support=ground_support.astype(bool),
        distance_to_obstacle_m=distance_to_obstacle.astype(np.float32),
    )
    audit = {
        "world": world,
        "pointcloud": str(mesh_root / world / "preview" / "pointcloud.ply"),
        "sampled_points": int(len(points)),
        "grid_shape": list(traversable.shape),
        "resolution_m": res,
        "occupied_ratio": float(occupied.mean()),
        "inflated_obstacle_ratio": float(inflated.mean()),
        "traversable_ratio": float(traversable.mean()),
        "collision_radius_m": radius,
        "source": "M-TARE vehicle_simulator mesh preview pointcloud; same source used by visualization_tools /overall_map",
    }
    return grid, audit


def choose_centers(grid: WorldGrid, count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    clearance = grid.distance_to_obstacle_m
    candidates = np.argwhere(grid.traversable & (clearance > 0.7))
    if len(candidates) == 0:
        raise RuntimeError(f"{grid.world}: no traversable candidates")
    # Farthest-point style downsampling over a random pool keeps samples decorrelated.
    pool_size = min(len(candidates), max(count * 20, count))
    pool = candidates[rng.choice(len(candidates), size=pool_size, replace=False)]
    selected = [pool[int(rng.integers(len(pool)))]]
    d2 = np.sum((pool - selected[0]) ** 2, axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(d2))
        selected.append(pool[idx])
        d2 = np.minimum(d2, np.sum((pool - pool[idx]) ** 2, axis=1))
    centers = []
    selected_arr = np.asarray(selected, dtype=np.int64)
    xy = grid.grid_to_world(selected_arr)
    for i, (cell, pos) in enumerate(zip(selected_arr, xy)):
        # Estimate a local principal direction from traversable cells around the center.
        r0, c0 = int(cell[0]), int(cell[1])
        window = 25
        rs = slice(max(0, r0 - window), min(grid.traversable.shape[0], r0 + window + 1))
        cs = slice(max(0, c0 - window), min(grid.traversable.shape[1], c0 + window + 1))
        local = np.argwhere(grid.traversable[rs, cs])
        if len(local) > 4:
            local_xy = local[:, ::-1].astype(np.float64)
            local_xy -= local_xy.mean(axis=0)
            _, _, vh = np.linalg.svd(local_xy, full_matrices=False)
            yaw = float(math.atan2(vh[0, 1], vh[0, 0]))
        else:
            yaw = 0.0
        centers.append({"index": i, "xy": pos.astype(np.float64), "cell": cell.astype(np.int64), "yaw": yaw})
    return centers


def pose_from_center(center: dict[str, Any]) -> Pose2D:
    return Pose2D(float(center["xy"][0]), float(center["xy"][1]), 0.0, float(center["yaw"]))


def select_visible_points(grid: WorldGrid, center: dict[str, Any], cfg: dict[str, Any], history_offset_m: float = 0.0) -> tuple[np.ndarray, Pose2D]:
    yaw = float(center["yaw"])
    origin_xy = center["xy"] - history_offset_m * np.asarray([math.cos(yaw), math.sin(yaw)])
    pose = Pose2D(float(origin_xy[0]), float(origin_xy[1]), 0.0, yaw)
    delta = grid.points[:, :2] - origin_xy.reshape(1, 2)
    dist = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
    angle = np.angle(np.exp(1j * angle))
    fov = math.radians(float(cfg["student_input"]["fov_deg"])) / 2.0
    keep = (dist <= float(cfg["student_input"]["current_range_m"])) & (np.abs(angle) <= fov)
    return grid.points[keep], pose


def local_traversable(grid: WorldGrid, pose: Pose2D, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size_m = float(cfg["teacher"]["local_size_m"])
    res = float(cfg["teacher"]["local_resolution_m"])
    n = int(round(size_m / res))
    coords = (np.arange(n) + 0.5) * res - size_m / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    wx = pose.x + cy * xx - sy * yy
    wy = pose.y + sy * xx + cy * yy
    rc = grid.world_to_grid(np.c_[wx.ravel(), wy.ravel()])
    valid = (rc[:, 0] >= 0) & (rc[:, 0] < grid.traversable.shape[1]) & (rc[:, 1] >= 0) & (rc[:, 1] < grid.traversable.shape[0])
    trav = np.zeros(n * n, dtype=bool)
    occ = np.zeros(n * n, dtype=bool)
    rr = rc[valid, 1]
    cc = rc[valid, 0]
    trav[valid] = grid.traversable[rr, cc]
    occ[valid] = grid.inflated_obstacle[rr, cc]
    trav = trav.reshape(n, n)
    occ = occ.reshape(n, n)
    start = np.asarray([n // 2, n // 2], dtype=np.int64)
    if not trav[start[0], start[1]]:
        cells = np.argwhere(trav)
        if len(cells):
            start = cells[np.argmin(np.sum((cells - start) ** 2, axis=1))]
    connected = np.zeros_like(trav, dtype=bool)
    if trav[start[0], start[1]]:
        labels, num = ndimage.label(trav, structure=np.ones((3, 3), dtype=np.uint8))
        connected = labels == labels[start[0], start[1]]
    return trav, occ, connected


def branch_lengths_from_skeleton(skel: np.ndarray, res: float, max_branches: int = 16) -> tuple[np.ndarray, np.ndarray, int]:
    kernel = np.ones((3, 3), dtype=np.int32)
    degree = ndimage.convolve(skel.astype(np.int32), kernel, mode="constant") - skel.astype(np.int32)
    endpoints = np.argwhere(skel & (degree == 1))
    center = np.asarray(skel.shape) // 2
    angles = np.zeros(max_branches, dtype=np.float32)
    lengths = np.zeros(max_branches, dtype=np.float32)
    if len(endpoints):
        order = np.argsort(np.sum((endpoints - center) ** 2, axis=1))[::-1][:max_branches]
        for j, idx in enumerate(order):
            dr, dc = endpoints[idx] - center
            angles[j] = math.atan2(-float(dr), float(dc))
            lengths[j] = float(np.linalg.norm([dr, dc]) * res)
    node_degree = int(np.sum(skel & (degree >= 3)))
    return angles, lengths, node_degree


def teacher_for_pose(grid: WorldGrid, pose: Pose2D, cfg: dict[str, Any]) -> dict[str, Any]:
    trav, occ, conn = local_traversable(grid, pose, cfg)
    res = float(cfg["teacher"]["local_resolution_m"])
    n = trav.shape[0]
    center = np.asarray([n // 2, n // 2])
    cells = np.argwhere(conn)
    bins = int(cfg["teacher"]["direction_bins"])
    reach = np.zeros(bins, dtype=np.float32)
    dist_out = np.zeros(bins, dtype=np.float32)
    volume = np.zeros(bins, dtype=np.float32)
    independent = np.zeros(bins, dtype=np.float32)
    if len(cells):
        delta = cells - center.reshape(1, 2)
        metric = np.c_[delta[:, 1], -delta[:, 0]] * res
        dist = np.linalg.norm(metric, axis=1)
        ang = np.mod(np.arctan2(metric[:, 1], metric[:, 0]), 2 * np.pi)
        idx = np.floor(ang / (2 * np.pi) * bins).astype(int) % bins
        for b in range(bins):
            vals = dist[idx == b]
            if len(vals):
                dist_out[b] = min(float(vals.max()), float(cfg["teacher"]["max_reachable_distance_m"])) / float(cfg["teacher"]["max_reachable_distance_m"])
                volume[b] = min(float(np.sum(vals <= float(cfg["teacher"]["near_distance_m"]))), 200.0) / 200.0
                reach[b] = float(vals.max() >= float(cfg["teacher"]["exit_min_distance_m"]))
        active = reach > 0.5
        for b in np.where(active)[0]:
            left = active[(b - 1) % bins]
            if not left:
                independent[b] = 1.0
    skel = skeletonize(conn)
    for _ in range(int(cfg["teacher"]["skeleton_prune_iterations"])):
        deg = ndimage.convolve(skel.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant") - skel.astype(np.int32)
        short_end = skel & (deg == 1)
        skel[short_end] = False
    branch_angles, branch_lengths, node_degree = branch_lengths_from_skeleton(skel, res)
    free_area = float(conn.sum() * res * res)
    dist_field = ndimage.distance_transform_edt(conn) * res
    center_clearance = float(dist_field[center[0], center[1]]) if conn[center[0], center[1]] else 0.0
    exit_count = int(independent.sum())
    angle_vals = np.where(reach > 0.5)[0]
    if len(angle_vals) > 1:
        doubled = np.exp(2j * (angle_vals + 0.5) / bins * 2 * np.pi)
        main_cont = float(abs(doubled.mean()))
    else:
        main_cont = float(reach.max())
    left_right = abs(float(dist_out[bins // 4] - dist_out[(3 * bins) // 4]))
    front_back = float(dist_out[0] + dist_out[bins // 2])
    turn_strength = float(max(dist_out[bins // 4], dist_out[(3 * bins) // 4]) * max(dist_out[0], dist_out[bins // 2]))
    bottleneck = float(np.clip(1.0 - center_clearance / 1.5, 0.0, 1.0))
    openness = float(np.clip(free_area / (float(cfg["teacher"]["local_size_m"]) ** 2), 0.0, 1.0))
    transition = float(np.std(dist_out))
    topological_node_score = float(np.clip(0.25 * exit_count + 0.35 * turn_strength + 0.25 * bottleneck + 0.15 * transition, 0.0, 1.0))
    role = np.concatenate([dist_out, reach]).astype(np.float32)
    role /= max(float(np.linalg.norm(role)), 1e-8)
    scores = {
        "branch_strength": float(np.clip(exit_count / 4.0, 0.0, 1.0)),
        "main_path_continuity": main_cont,
        "turn_strength": turn_strength,
        "bottleneck_score": bottleneck,
        "openness_score": openness,
        "transition_score": transition,
        "topological_node_score": topological_node_score,
        "exit_count": float(exit_count),
        "center_clearance_m": center_clearance,
        "free_area_m2": free_area,
        "front_back_reach": front_back,
        "lateral_asymmetry": left_right,
    }
    return {
        "traversable": trav.astype(np.uint8),
        "inflated_obstacle": occ.astype(np.uint8),
        "connected_component": conn.astype(np.uint8),
        "skeleton": skel.astype(np.uint8),
        "reachability": reach,
        "distance": dist_out,
        "volume": volume,
        "independent_exit": independent,
        "branch_angles": branch_angles,
        "branch_lengths": branch_lengths,
        "node_degree": np.int32(node_degree),
        "scores": scores,
        "role": role,
    }


def save_preview(path: Path, sample: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes[0, 0].imshow(sample["student_input_current"][0], origin="upper", cmap="gray")
    axes[0, 0].set_title("online surface")
    axes[0, 1].imshow(sample["connected_component"], origin="upper", cmap="Greens")
    axes[0, 1].imshow(sample["inflated_obstacle"], origin="upper", cmap="Reds", alpha=0.35)
    axes[0, 1].set_title("traversable / obstacle")
    axes[0, 2].imshow(sample["skeleton"], origin="upper", cmap="gray")
    axes[0, 2].set_title("local skeleton")
    bins = len(sample["traversable_direction_distribution"])
    angles = np.arange(bins) / bins * 2 * np.pi
    axes[1, 0].plot(angles, sample["reachable_distance_distribution"])
    axes[1, 0].set_title("reachable distance")
    axes[1, 1].bar(np.arange(bins), sample["traversable_direction_distribution"])
    axes[1, 1].set_title("direction reach")
    axes[1, 2].bar(range(len(sample["continuous_structural_scores"])), list(sample["continuous_structural_scores"].values()))
    axes[1, 2].set_xticks([])
    axes[1, 2].set_title("scores")
    for ax in axes.ravel():
        ax.set_axis_off() if ax not in (axes[1, 0], axes[1, 1]) else None
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def sample_record(grid: WorldGrid, center: dict[str, Any], builder: SurfaceEvidenceBuilder, cfg: dict[str, Any]) -> dict[str, Any]:
    visible, pose = select_visible_points(grid, center, cfg, 0.0)
    current, _ = builder.build([visible], pose)
    history = []
    rel = []
    for step in range(int(cfg["student_input"]["history_steps"])):
        offset = step * float(cfg["student_input"]["history_step_m"])
        pts, hp = select_visible_points(grid, center, cfg, offset)
        tensor, _ = builder.build([pts], pose)
        history.append(tensor)
        rel.append([hp.x - pose.x, hp.y - pose.y, hp.z - pose.z, math.cos(hp.yaw - pose.yaw), math.sin(hp.yaw - pose.yaw)])
    teacher = teacher_for_pose(grid, pose, cfg)
    rec = {
        "student_input_current": current.astype(np.float32),
        "student_input_history": np.stack(history).astype(np.float32),
        "relative_poses": np.asarray(rel, dtype=np.float32),
        "traversable_direction_distribution": teacher["reachability"],
        "reachable_distance_distribution": teacher["distance"],
        "reachable_area_distribution": teacher["volume"],
        "independent_exit_distribution": teacher["independent_exit"],
        "connected_component": teacher["connected_component"],
        "inflated_obstacle": teacher["inflated_obstacle"],
        "skeleton": teacher["skeleton"],
        "branch_angles": teacher["branch_angles"],
        "branch_lengths": teacher["branch_lengths"],
        "node_degree": teacher["node_degree"],
        "continuous_structural_scores": teacher["scores"],
        "topological_role_teacher": teacher["role"],
        "center_pose": np.asarray([pose.x, pose.y, pose.z, 0.0, 0.0, math.sin(pose.yaw / 2.0), math.cos(pose.yaw / 2.0)], dtype=np.float32),
        "raw_visible_points": int(len(visible)),
    }
    return rec


def save_npz(path: Path, rec: dict[str, Any], world: str, split: str, index: int, cfg_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_json = json.dumps(rec["continuous_structural_scores"], sort_keys=True)
    np.savez_compressed(
        path,
        student_input_current=rec["student_input_current"],
        student_input_history=rec["student_input_history"],
        relative_poses=rec["relative_poses"],
        traversable_direction_distribution=rec["traversable_direction_distribution"].astype(np.float32),
        reachable_distance_distribution=rec["reachable_distance_distribution"].astype(np.float32),
        reachable_area_distribution=rec["reachable_area_distribution"].astype(np.float32),
        independent_exit_distribution=rec["independent_exit_distribution"].astype(np.float32),
        local_connectivity_data=rec["connected_component"].astype(np.uint8),
        inflated_obstacle=rec["inflated_obstacle"].astype(np.uint8),
        skeleton=rec["skeleton"].astype(np.uint8),
        branch_angles=rec["branch_angles"].astype(np.float32),
        branch_lengths=rec["branch_lengths"].astype(np.float32),
        node_degree=np.int32(rec["node_degree"]),
        continuous_structural_scores_json=np.asarray(scores_json),
        topological_role_teacher=rec["topological_role_teacher"].astype(np.float32),
        center_pose=rec["center_pose"],
        world=np.asarray(world),
        split=np.asarray(split),
        sample_index=np.int64(index),
        contract_version=np.asarray("local_topological_semantic_teacher_v1"),
        config_hash=np.asarray(cfg_hash),
    )


@torch.no_grad()
def geometry_latent(inputs: np.ndarray, cfg: dict[str, Any], device: torch.device) -> np.ndarray | None:
    enc_path = Path(cfg["experiment"]["old_geometry_encoder"])
    proj_path = Path(cfg["experiment"]["old_geometry_projection"])
    if not enc_path.exists() or not proj_path.exists():
        return None
    model = ForcedGlobalBottleneckNet(input_channels=3, latent_dim=128, base_channels=24).to(device)
    model.load_state_dict(torch.load(enc_path, map_location=device, weights_only=False)["model"])
    proj = StructuralProjectionHead(128, 128, 64).to(device)
    proj.load_state_dict(torch.load(proj_path, map_location=device, weights_only=False)["projection"])
    model.eval()
    proj.eval()
    outs = []
    for start in range(0, len(inputs), 128):
        batch = torch.from_numpy(inputs[start : start + 128, [0, 2, 3]]).to(device)
        outs.append(proj(model.encode(batch)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def pair_correlation(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if len(records) < 4:
        return {}
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + 99)
    inputs = np.stack([r["student_input_current"] for r in records])
    roles = np.stack([r["topological_role_teacher"] for r in records])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surface_desc = surface_descriptors(inputs, device)
    geom = geometry_latent(inputs, cfg, device)
    n_pairs = min(30000, len(records) * (len(records) - 1) // 2)
    left = rng.integers(0, len(records), size=n_pairs)
    right = rng.integers(0, len(records), size=n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]
    topo = cosine_distance(roles[left], roles[right])
    surface = cosine_distance(surface_desc[left], surface_desc[right])
    out = {
        "surface_geometry_vs_topological_teacher_spearman": spearman(surface, topo),
        "surface_difference": describe(surface),
        "topological_teacher_difference": describe(topo),
    }
    if geom is not None:
        gd = cosine_distance(geom[left], geom[right])
        out["existing_64d_vs_surface_spearman"] = spearman(gd, surface)
        out["existing_64d_vs_topological_teacher_spearman"] = spearman(gd, topo)
        out["existing_64d_distance"] = describe(gd)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/topological_semantic_teacher_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["experiment"]["output_dir"])
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    for sub in ("train", "val", "test", "diagnostics", "counterexamples", "previews", "world_definitions"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    shutil.copy2(args.config, root / "diagnostics" / "config.yaml")
    write_json(
        root / "diagnostics" / "provenance.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "source_sha256": {str(Path(__file__)): sha256(Path(__file__)), str(Path(args.config)): sha256(Path(args.config))},
        },
    )
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    builder = SurfaceEvidenceBuilder(
        SurfaceEvidenceConfig(
            size_m=float(cfg["student_input"]["surface_evidence_size_m"]),
            resolution_m=float(cfg["student_input"]["surface_evidence_resolution_m"]),
        )
    )
    split_by_world = {world: split for split, worlds in cfg["splits"].items() for world in worlds}
    world_audits = {}
    records_all: list[dict[str, Any]] = []
    stats = {}
    for world, wcfg in cfg["worlds"].items():
        split = split_by_world[world]
        grid, audit = build_world_grid(world, cfg, rng)
        world_audits[world] = audit
        np.savez_compressed(
            root / "world_definitions" / f"{world}_traversable_grid.npz",
            traversable=grid.traversable.astype(np.uint8),
            occupied=grid.occupied.astype(np.uint8),
            inflated_obstacle=grid.inflated_obstacle.astype(np.uint8),
            origin_xy=grid.origin_xy,
            resolution=np.float32(grid.resolution),
        )
        centers = choose_centers(grid, int(wcfg["target_samples"]), rng)
        world_records = []
        for i, center in enumerate(centers):
            rec = sample_record(grid, center, builder, cfg)
            rec["world"] = world
            rec["split"] = split
            rec["sample_index"] = i
            path = root / split / f"{split}_{world}_{i:05d}.npz"
            save_npz(path, rec, world, split, i, cfg_hash)
            world_records.append(rec)
            records_all.append(rec)
            if i < int(cfg["previews"]["count_per_world"]):
                save_preview(root / "previews" / f"{world}_{i:04d}.png", rec)
        scores = [r["continuous_structural_scores"] for r in world_records]
        stats[world] = {
            "split": split,
            "sample_count": len(world_records),
            "raw_visible_points": describe([r["raw_visible_points"] for r in world_records]),
            "exit_count": describe([s["exit_count"] for s in scores]),
            "topological_node_score": describe([s["topological_node_score"] for s in scores]),
            "openness_score": describe([s["openness_score"] for s in scores]),
            "bottleneck_score": describe([s["bottleneck_score"] for s in scores]),
            "turn_strength": describe([s["turn_strength"] for s in scores]),
            "center_clearance_m": describe([s["center_clearance_m"] for s in scores]),
        }
    teacher_contract = {
        "contract_version": "local_topological_semantic_teacher_v1",
        "teacher_source": "complete M-TARE vehicle_simulator preview pointcloud converted to inflated 2D traversable grid",
        "not_model_input": ["traversable grid", "skeleton", "direction reachability", "topological role", "world", "absolute pose"],
        "student_input": "surface_evidence_v1 current plus causal history only",
        "direction_bins": int(cfg["teacher"]["direction_bins"]),
        "role_dim": int(cfg["teacher"]["role_vector_dim"]),
        "scores": {
            "branch_strength": "exit_count / 4 clipped to [0,1]",
            "main_path_continuity": "axial concentration of reachable directions",
            "turn_strength": "lateral reach times longitudinal reach",
            "bottleneck_score": "1 - center clearance / 1.5m clipped",
            "openness_score": "connected traversable area / local area",
            "transition_score": "std of normalized direction reach distances",
            "topological_node_score": "weighted branch/turn/bottleneck/transition score",
        },
    }
    input_contract = {
        "student_input_current": [4, 100, 100],
        "student_input_history": [int(cfg["student_input"]["history_steps"]), 4, 100, 100],
        "relative_poses": [int(cfg["student_input"]["history_steps"]), 5],
        "online_only": True,
        "surface_contract": "surface_evidence_v1",
    }
    split_definition = {"splits": cfg["splits"], "policy": "world-level isolation; no random frame split"}
    quality = {
        "world_audit": world_audits,
        "trajectory_consistency": "synthetic centers are sampled only from inflated traversable cells with clearance > 0.7m",
        "resolution_check": "0.2m teacher grid exported; 0.4m sensitivity recorded as limitation for v1",
        "height_terrain_limit": "2D traversability from complete pointcloud height span; M-TARE height/terrain dynamics simplified",
        "existing_64d_comparison": pair_correlation(records_all, cfg),
    }
    write_json(root / "input_contract.json", input_contract)
    write_json(root / "teacher_contract.json", teacher_contract)
    write_json(root / "split_definition.json", split_definition)
    write_json(root / "teacher_quality_metrics.json", quality)
    write_json(root / "dataset_stats.json", {"status": "TOPOLOGICAL_TEACHER_READY_WITH_LIMITATIONS", "worlds": stats, "total_samples": len(records_all), "config_hash": cfg_hash})
    # Counterexample candidates: surface-similar/topology-different and surface-different/topology-similar.
    corr = pair_correlation(records_all, cfg)
    write_json(root / "counterexamples" / "selection_policy.json", {"policy": "counterexamples identified by low/high disagreement between surface descriptor distance and topological role distance", "summary": corr})
    print(json.dumps({"output": str(root), "samples": len(records_all), "status": "TOPOLOGICAL_TEACHER_READY_WITH_LIMITATIONS"}, indent=2))


if __name__ == "__main__":
    main()
