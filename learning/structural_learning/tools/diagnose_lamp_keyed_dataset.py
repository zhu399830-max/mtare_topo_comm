from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from scipy import ndimage
from scipy.spatial import cKDTree

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.datasets.common import pointcloud2_xyz, pose_from_msg_pose, stamp_to_ns
from learning.local_structural_map.geometry import grid_indices, transform_points_local_to_world
from learning.local_structural_map.schema import Pose2D, StandardFrame
from learning.structural_learning.tools.export_lamp_structural_dataset import (
    config_hash,
    dijkstra_free,
    nearest_free_start,
    quat_xyzw_from_yaw,
    save_standard_frame_index,
    yaw_delta,
)


@dataclass
class ScanRecord:
    robot: str
    key: int
    bag_timestamp_ns: int
    scan_header_stamp_ns: int
    scan_frame_id: str
    raw_points: np.ndarray


@dataclass
class PoseRecord:
    pose: Pose2D
    covariance: List[float]
    node_stamp_ns: int
    occurrence_count: int
    q_norm: float
    incremental_values: List[bool]


@dataclass
class Center:
    index: int
    robot: str
    frame_index: int
    key: int
    stamp_ns: int
    pose: Pose2D
    split: str = ""
    region_id: str = ""
    repeat_neighbor_count: int = 0


def read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def describe(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def pose_vec(node) -> np.ndarray:
    p = node.pose.position
    q = node.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)


def audit_and_load_robot(lamp_root: Path, robot: str, max_abs_m: float) -> Tuple[List[ScanRecord], Dict[int, PoseRecord], Dict[str, object]]:
    bag = lamp_root / "rosbag" / f"{robot}.bag"
    scan_topic = f"/{robot}/lamp/keyed_scans"
    pose_topic = f"/{robot}/lamp/pose_graph_incremental"
    scans: List[ScanRecord] = []
    scan_key_counts: Counter[int] = Counter()
    scan_frame_ids: Counter[str] = Counter()
    pose_occurrence: Counter[int] = Counter()
    pose_latest: Dict[int, PoseRecord] = {}
    pose_variants: Dict[int, List[np.ndarray]] = defaultdict(list)
    incremental_counts: Counter[str] = Counter()
    sample_messages = {"keyed_scans": [], "pose_graphs": []}

    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in {scan_topic, pose_topic}]
        for conn, timestamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == scan_topic:
                key = int(msg.key)
                points = pointcloud2_xyz(msg.scan)
                stamp_ns = stamp_to_ns(msg.scan.header.stamp)
                frame_id = str(msg.scan.header.frame_id)
                scans.append(ScanRecord(robot, key, int(timestamp_ns), int(stamp_ns), frame_id, points))
                scan_key_counts[key] += 1
                scan_frame_ids[frame_id] += 1
                if len(sample_messages["keyed_scans"]) < 5:
                    sample_messages["keyed_scans"].append({
                        "bag_timestamp_ns": int(timestamp_ns),
                        "key": key,
                        "scan_header_stamp_ns": int(stamp_ns),
                        "scan_frame_id": frame_id,
                        "point_count": int(len(points)),
                        "xyz_min": np.nanmin(points, axis=0).astype(float).tolist() if len(points) else None,
                        "xyz_max": np.nanmax(points, axis=0).astype(float).tolist() if len(points) else None,
                    })
            else:
                incremental = bool(msg.incremental)
                incremental_counts[str(incremental)] += 1
                if len(sample_messages["pose_graphs"]) < 5:
                    sample_messages["pose_graphs"].append({
                        "bag_timestamp_ns": int(timestamp_ns),
                        "incremental": incremental,
                        "node_count": int(len(msg.nodes)),
                        "edge_count": int(len(msg.edges)),
                        "nodes": [
                            {
                                "key": int(n.key),
                                "ID": str(n.ID),
                                "header_stamp_ns": int(stamp_to_ns(n.header.stamp)),
                                "pose": pose_vec(n).astype(float).tolist(),
                            }
                            for n in list(msg.nodes)[:3]
                        ],
                    })
                for node in msg.nodes:
                    key = int(node.key)
                    pose_occurrence[key] += 1
                    q = node.pose.orientation
                    qn = math.sqrt(float(q.x) ** 2 + float(q.y) ** 2 + float(q.z) ** 2 + float(q.w) ** 2)
                    old = pose_latest.get(key)
                    inc_vals = [] if old is None else list(old.incremental_values)
                    inc_vals.append(incremental)
                    pose_latest[key] = PoseRecord(
                        pose=pose_from_msg_pose(node.pose),
                        covariance=[float(v) for v in node.covariance],
                        node_stamp_ns=int(stamp_to_ns(node.header.stamp)),
                        occurrence_count=int(pose_occurrence[key]),
                        q_norm=float(qn),
                        incremental_values=inc_vals,
                    )
                    pose_variants[key].append(pose_vec(node))

    scan_keys = set(scan_key_counts)
    pose_keys = set(pose_latest)
    pose_changed = 0
    pose_change_examples = []
    for key, vals in pose_variants.items():
        arr = np.stack(vals, axis=0)
        if np.max(np.ptp(arr, axis=0)) > 1e-6:
            pose_changed += 1
            if len(pose_change_examples) < 10:
                pose_change_examples.append({
                    "key": int(key),
                    "occurrences": int(len(vals)),
                    "max_abs_delta": float(np.max(np.ptp(arr, axis=0))),
                })
    finite_fail = 0
    coord_fail = 0
    point_counts = []
    for rec in scans:
        point_counts.append(int(len(rec.raw_points)))
        if len(rec.raw_points) and not np.isfinite(rec.raw_points).all():
            finite_fail += 1
        if len(rec.raw_points) and np.nanmax(np.abs(rec.raw_points)) > max_abs_m:
            coord_fail += 1

    audit = {
        "robot": robot,
        "bag": str(bag),
        "rosmsg_show": {
            "pose_graph_msgs/KeyedScan": "UNAVAILABLE: rosmsg command not found in current environment",
            "pose_graph_msgs/PoseGraph": "UNAVAILABLE: rosmsg command not found in current environment",
            "pose_graph_msgs/PoseGraphNode": "UNAVAILABLE: rosmsg command not found in current environment",
        },
        "message_definitions_expected": {
            "pose_graph_msgs/KeyedScan": ["uint64 key", "sensor_msgs/PointCloud2 scan"],
            "pose_graph_msgs/PoseGraphNode": ["Header header", "uint64 key", "string ID", "geometry_msgs/Pose pose", "float64[36] covariance"],
        },
        "scan_topic": scan_topic,
        "pose_topic": pose_topic,
        "keyed_scan_total": int(len(scans)),
        "unique_scan_key_count": int(len(scan_keys)),
        "unique_pose_node_key_count": int(len(pose_keys)),
        "exact_key_match_count": int(sum(scan_key_counts[k] for k in scan_keys & pose_keys)),
        "scan_key_without_pose_count": int(len(scan_keys - pose_keys)),
        "pose_key_without_scan_count": int(len(pose_keys - scan_keys)),
        "scan_key_without_pose_examples": [int(k) for k in sorted(scan_keys - pose_keys)[:20]],
        "pose_key_without_scan_examples": [int(k) for k in sorted(pose_keys - scan_keys)[:20]],
        "duplicate_scan_key_count": int(sum(v - 1 for v in scan_key_counts.values() if v > 1)),
        "scan_key_occurrence_histogram": {str(k): int(v) for k, v in Counter(scan_key_counts.values()).items()},
        "pose_occurrence_histogram": {str(k): int(v) for k, v in Counter(pose_occurrence.values()).items()},
        "pose_keys_with_updates": int(sum(1 for v in pose_occurrence.values() if v > 1)),
        "pose_keys_whose_pose_changed": int(pose_changed),
        "pose_change_examples": pose_change_examples,
        "pose_graph_incremental_counts": dict(incremental_counts),
        "scan_frame_id_counts": dict(scan_frame_ids),
        "scan_point_count": describe(point_counts),
        "raw_scan_nan_inf_count": int(finite_fail),
        "raw_scan_coord_out_of_range_count": int(coord_fail),
        "sample_messages": sample_messages,
    }
    return scans, pose_latest, audit


def make_standard_frames(scans: List[ScanRecord], poses: Dict[int, PoseRecord], max_abs_m: float) -> Tuple[List[StandardFrame], Dict[str, object]]:
    frames: List[StandardFrame] = []
    invalid = Counter()
    raw_ranges = []
    world_ranges = []
    for rec in sorted(scans, key=lambda s: (s.bag_timestamp_ns, s.key)):
        pose_rec = poses.get(rec.key)
        if pose_rec is None:
            invalid["missing_pose_key"] += 1
            continue
        pts = rec.raw_points.astype(np.float32, copy=False)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) == 0:
            invalid["empty_or_nonfinite_scan"] += 1
            continue
        raw_ranges.append(float(np.max(np.linalg.norm(pts[:, :3], axis=1))))
        if np.nanmax(np.abs(pts)) > max_abs_m:
            invalid["raw_points_out_of_range"] += 1
            continue
        points_world = transform_points_local_to_world(pts, pose_rec.pose)
        if not np.isfinite(points_world).all():
            invalid["world_points_nan_inf"] += 1
            continue
        if np.max(np.abs(points_world)) > max_abs_m:
            invalid["world_points_out_of_range"] += 1
            continue
        world_ranges.append(float(np.max(np.linalg.norm(points_world - np.array([pose_rec.pose.x, pose_rec.pose.y, pose_rec.pose.z]), axis=1))))
        frames.append(StandardFrame(
            timestamp_ns=int(rec.bag_timestamp_ns),
            points_world=points_world,
            sensor_origin_world=np.array([pose_rec.pose.x, pose_rec.pose.y, pose_rec.pose.z], dtype=np.float32),
            pose=pose_rec.pose,
            source="LAMP_KEYED",
            frame_id="map",
            metadata={
                "robot": rec.robot,
                "trajectory_name": rec.robot,
                "key": int(rec.key),
                "scan_bag_timestamp_ns": int(rec.bag_timestamp_ns),
                "scan_header_timestamp_ns": int(rec.scan_header_stamp_ns),
                "pose_node_header_timestamp_ns": int(pose_rec.node_stamp_ns),
                "scan_frame_id": rec.scan_frame_id,
                "raw_points": int(len(rec.raw_points)),
                "points_world_count": int(len(points_world)),
                "pose_occurrence_count": int(pose_rec.occurrence_count),
                "pose_q_norm": float(pose_rec.q_norm),
            },
        ))
    jumps = []
    yaws = []
    intervals = []
    for a, b in zip(frames, frames[1:]):
        jumps.append(float(np.linalg.norm(b.sensor_origin_world[:2] - a.sensor_origin_world[:2])))
        yaws.append(float(yaw_delta(a.pose.yaw, b.pose.yaw)))
        intervals.append((b.timestamp_ns - a.timestamp_ns) / 1e9)
    audit = {
        "valid_standard_frames": int(len(frames)),
        "invalid": dict(invalid),
        "bag_time_interval_sec": describe(intervals),
        "adjacent_position_jump_m": describe(jumps),
        "adjacent_yaw_jump_rad": describe(yaws),
        "raw_scan_range_m": describe(raw_ranges),
        "world_relative_range_m": describe(world_ranges),
        "timestamp_note": "StandardFrame.timestamp_ns uses scan bag timestamp for ordering/windowing because scan header stamp is zero; pose association is strictly by KeyedScan.key.",
    }
    return frames, audit


def select_centers(frames_by_robot: Dict[str, List[StandardFrame]], cfg: Dict[str, object]) -> List[Center]:
    centers: List[Center] = []
    min_d = float(cfg["sampling"]["min_center_translation_m"])
    min_yaw = math.radians(float(cfg["sampling"]["min_center_yaw_deg"]))
    for robot, frames in frames_by_robot.items():
        last_pose: Optional[Pose2D] = None
        for idx, frame in enumerate(frames):
            pose = frame.pose
            if last_pose is None:
                choose = True
            else:
                choose = math.hypot(pose.x - last_pose.x, pose.y - last_pose.y) >= min_d or yaw_delta(pose.yaw, last_pose.yaw) >= min_yaw
            if choose:
                centers.append(Center(
                    index=len(centers),
                    robot=robot,
                    frame_index=idx,
                    key=int(frame.metadata["key"]),
                    stamp_ns=frame.timestamp_ns,
                    pose=pose,
                ))
                last_pose = pose
    if centers:
        pts = np.array([[c.pose.x, c.pose.y] for c in centers], dtype=np.float32)
        tree = cKDTree(pts)
        for c, xy in zip(centers, pts):
            c.repeat_neighbor_count = max(0, len(tree.query_ball_point(xy, r=float(cfg["sampling"]["repeat_neighbor_radius_m"]))) - 1)
    return centers


def assign_train_val_split(centers: List[Center], cfg: Dict[str, object]) -> Dict[str, object]:
    split_cfg = cfg["split"]
    bin_m = float(split_cfg["region_bin_m"])
    buffer_m = float(split_cfg["buffer_m"])
    pts = np.array([[c.pose.x, c.pose.y] for c in centers], dtype=np.float32)
    order = np.argsort(pts[:, 0])
    train_cut = int(round(len(order) * float(split_cfg["target_train"])))
    train_ids = set(order[:train_cut].tolist())
    for idx, c in enumerate(centers):
        c.split = "train" if idx in train_ids else "val"
    train_pts = np.array([[c.pose.x, c.pose.y] for c in centers if c.split == "train"], dtype=np.float32)
    val_pts = np.array([[c.pose.x, c.pose.y] for c in centers if c.split == "val"], dtype=np.float32)
    if len(train_pts) and len(val_pts):
        train_tree = cKDTree(train_pts)
        val_tree = cKDTree(val_pts)
        for c in centers:
            xy = np.array([c.pose.x, c.pose.y], dtype=np.float32)
            if c.split == "train":
                d = float(val_tree.query(xy, k=1)[0])
            else:
                d = float(train_tree.query(xy, k=1)[0])
            if d < buffer_m:
                c.split = "buffer"
    min_xy = pts.min(axis=0)
    for c in centers:
        key = (int(math.floor((c.pose.x - min_xy[0]) / bin_m)), int(math.floor((c.pose.y - min_xy[1]) / bin_m)))
        c.region_id = f"r{key[0]}_{key[1]}"
    return {
        "method": "x_axis_train_val_with_20m_buffer",
        "test_policy": "not generated from single LAMP tunnel; reserved for another environment to avoid leakage",
        "buffer_m": buffer_m,
    }


def nearest_split_distance(centers: List[Center], a: str, b: str) -> Optional[float]:
    pa = np.array([[c.pose.x, c.pose.y] for c in centers if c.split == a], dtype=np.float32)
    pb = np.array([[c.pose.x, c.pose.y] for c in centers if c.split == b], dtype=np.float32)
    if len(pa) == 0 or len(pb) == 0:
        return None
    return float(cKDTree(pa).query(pb, k=1)[0].min())


def diagnostic_subset(centers: List[Center], count: int) -> List[Center]:
    valid = [c for c in centers if c.split in {"train", "val"}]
    if len(valid) <= count:
        return valid
    pts = np.array([[c.pose.x, c.pose.y] for c in valid], dtype=np.float32)
    arclike = np.argsort(pts[:, 0] + 0.1 * pts[:, 1])
    idxs = set(np.linspace(0, len(arclike) - 1, count, dtype=int).tolist())
    selected = [valid[int(arclike[i])] for i in sorted(idxs)]
    repeat = [c for c in valid if c.repeat_neighbor_count > 0]
    for c in repeat[:: max(1, len(repeat) // 10)]:
        if len(selected) >= count:
            break
        selected.append(c)
    return selected[:count]


def build_partial(frames: List[StandardFrame], center: Center, local_cfg: LocalMapConfig, partial_cfg: Dict[str, object]):
    builder = LocalStructuralMapBuilder(local_cfg)
    t = center.stamp_ns
    min_t = t - int(float(partial_cfg["window_sec"]) * 1e9)
    idxs = [i for i in range(center.frame_index + 1) if frames[i].timestamp_ns >= min_t]
    idxs = idxs[-int(partial_cfg["max_frames"]):]
    item = None
    for idx in idxs:
        item = builder.update(frames[idx])
    if item is None:
        item = builder.build_single(frames[center.frame_index])
    enforce_channel_exclusivity(item.tensor, local_cfg)
    return item, idxs


def teacher_indices(frames_by_robot: Dict[str, List[StandardFrame]], center: Center, split_by_key: Dict[Tuple[str, int], str], cfg: Dict[str, object]) -> List[Tuple[str, int, float]]:
    teacher_cfg = cfg["teacher"]
    radius = float(teacher_cfg["radius_m"])
    max_frames = int(teacher_cfg["max_frames"])
    bin_m = float(teacher_cfg["spatial_bin_m"])
    cxy = np.array([center.pose.x, center.pose.y], dtype=np.float32)
    candidates = []
    for robot, frames in frames_by_robot.items():
        for idx, frame in enumerate(frames):
            key = int(frame.metadata["key"])
            if split_by_key.get((robot, key), "buffer") != center.split:
                continue
            xy = np.array([frame.pose.x, frame.pose.y], dtype=np.float32)
            d = float(np.linalg.norm(xy - cxy))
            if d <= radius:
                b = (int(math.floor((xy[0] - cxy[0]) / bin_m)), int(math.floor((xy[1] - cxy[1]) / bin_m)))
                candidates.append((b, d, robot, idx))
    by_bin: Dict[Tuple[int, int], List[Tuple[float, str, int]]] = defaultdict(list)
    for b, d, robot, idx in candidates:
        by_bin[b].append((d, robot, idx))
    selected: List[Tuple[str, int, float]] = []
    for vals in by_bin.values():
        vals.sort(key=lambda x: x[0])
        d, robot, idx = vals[0]
        selected.append((robot, idx, d))
    if len(selected) < max_frames:
        used = {(r, i) for r, i, _ in selected}
        rest = sorted([(d, r, i) for _, d, r, i in candidates if (r, i) not in used], key=lambda x: x[0])
        selected.extend([(r, i, d) for d, r, i in rest[: max_frames - len(selected)]])
    selected = selected[:max_frames]
    return selected


def build_teacher(frames_by_robot: Dict[str, List[StandardFrame]], center: Center, split_by_key: Dict[Tuple[str, int], str], local_cfg: LocalMapConfig, cfg: Dict[str, object]):
    builder = LocalStructuralMapBuilder(local_cfg)
    idxs = teacher_indices(frames_by_robot, center, split_by_key, cfg)
    clean_points = []
    frame_log = []
    for robot, idx, dist in idxs:
        frame = frames_by_robot[robot][idx]
        clean = builder._filtered_copy(frame)
        if len(clean.points_world):
            clean_points.append(clean.points_world)
        key = int(frame.metadata["key"])
        frame_log.append({
            "robot": robot,
            "key": key,
            "world_position": [float(frame.pose.x), float(frame.pose.y), float(frame.pose.z)],
            "distance_to_center_m": float(dist),
            "split": split_by_key.get((robot, key), "buffer"),
        })
    points = np.concatenate(clean_points, axis=0) if clean_points else np.zeros((0, 3), dtype=np.float32)
    center_frame = frames_by_robot[center.robot][center.frame_index]
    item = builder._build_from_points(center_frame, points, center_frame.sensor_origin_world)
    item.metadata["accumulated_frames"] = len(idxs)
    item.metadata["accumulated_points"] = int(len(points))
    enforce_channel_exclusivity(item.tensor, local_cfg)
    return item, idxs, frame_log


def enforce_channel_exclusivity(tensor: np.ndarray, local_cfg: LocalMapConfig) -> None:
    names = local_cfg.channels
    free_idx = names.index("free_mask")
    occ_idx = names.index("occupied_mask")
    tensor[free_idx] = np.where(tensor[occ_idx] > 0.5, 0.0, tensor[free_idx])


def label_from_mask(free: np.ndarray, occupied: np.ndarray, local_cfg: LocalMapConfig, label_cfg: Dict[str, object], prefix_arrays: bool = False):
    free = (free > 0.5) & ~(occupied > 0.5)
    start = nearest_free_start(free, local_cfg, float(label_cfg["start_search_radius_m"]))
    if start is None:
        return None
    dist, prev_r, prev_c = dijkstra_free(free, start, local_cfg.resolution_m)
    reachable_mask = np.isfinite(dist)
    clearance = ndimage.distance_transform_edt(free) * local_cfg.resolution_m
    n = local_cfg.grid_size
    rows, cols = np.indices((n, n))
    local_x = (cols + 0.5) * local_cfg.resolution_m - local_cfg.half_size_m
    local_y = local_cfg.half_size_m - (rows + 0.5) * local_cfg.resolution_m
    angles = (np.arctan2(local_y, local_x) + 2.0 * np.pi) % (2.0 * np.pi)
    radial = np.hypot(local_x, local_y)
    sectors = np.floor(angles / (2.0 * np.pi / int(label_cfg["directions"]))).astype(np.int32)
    max_effective = local_cfg.half_size_m * math.sqrt(2.0)
    reach = np.zeros(int(label_cfg["directions"]), dtype=np.float32)
    distance = np.zeros_like(reach)
    clear = np.zeros_like(reach)
    for k in range(len(reach)):
        mask = reachable_mask & (sectors == k)
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        rr, cc = coords[int(np.argmax(dist[mask]))]
        distance[k] = np.clip(float(dist[rr, cc]) / max_effective, 0.0, 1.0)
        reach[k] = float(np.max(radial[mask]) >= float(label_cfg["reachability_distance_m"]))
        path_vals = []
        pr, pc = int(rr), int(cc)
        guard = 0
        while pr >= 0 and pc >= 0 and guard < n * n:
            path_vals.append(float(clearance[pr, pc]))
            if (pr, pc) == start:
                break
            pr, pc = int(prev_r[pr, pc]), int(prev_c[pr, pc])
            guard += 1
        clear[k] = np.clip((float(np.median(path_vals)) if path_vals else 0.0) / local_cfg.half_size_m, 0.0, 1.0)
    return reach, distance, clear, start, reachable_mask, clearance


def traversability_labels(teacher_tensor: np.ndarray, local_cfg: LocalMapConfig, cfg: Dict[str, object]):
    names = local_cfg.channels
    free_raw = teacher_tensor[names.index("free_mask")] > 0.5
    occupied = teacher_tensor[names.index("occupied_mask")] > 0.5
    structure = ndimage.generate_binary_structure(2, 1)
    closed = ndimage.binary_closing(free_raw, structure=structure, iterations=int(cfg["labels"]["closing_radius_cells"]))
    closed = closed & free_raw | (closed & ~occupied & ndimage.binary_dilation(free_raw, structure=structure, iterations=1))
    radius_cells = int(math.ceil(local_cfg.robot_self_radius_m / local_cfg.resolution_m))
    obst_struct = ndimage.generate_binary_structure(2, 1)
    inflated = ndimage.binary_dilation(occupied, structure=obst_struct, iterations=radius_cells)
    traversable = closed & ~inflated & ~occupied
    labeled = label_from_mask(traversable, occupied, local_cfg, cfg["labels"])
    return labeled, {"teacher_free_raw": free_raw, "teacher_traversable": traversable, "inflated_obstacle": inflated}


def conflict_metrics(partial: np.ndarray, teacher: np.ndarray, local_cfg: LocalMapConfig) -> Dict[str, float]:
    names = local_cfg.channels
    p_free = partial[names.index("free_mask")] > 0.5
    p_occ = partial[names.index("occupied_mask")] > 0.5
    p_obs = partial[names.index("observed_mask")] > 0.5
    t_free = teacher[names.index("free_mask")] > 0.5
    t_occ = teacher[names.index("occupied_mask")] > 0.5
    t_obs = teacher[names.index("observed_mask")] > 0.5
    pf_to = int(np.sum(p_free & t_occ))
    pf_valid = int(np.sum(p_free & t_obs))
    po_tf = int(np.sum(p_occ & t_free))
    po_valid = int(np.sum(p_occ & t_obs))
    return {
        "partial_internal_free_and_occupied": int(np.sum(p_free & p_occ)),
        "teacher_internal_free_and_occupied": int(np.sum(t_free & t_occ)),
        "partial_teacher_observed_intersection": int(np.sum(p_obs & t_obs)),
        "partial_free_teacher_occ_count": pf_to,
        "partial_free_valid_denominator": pf_valid,
        "free_conflict_rate": float(pf_to / max(pf_valid, 1)),
        "partial_occ_teacher_free_count": po_tf,
        "partial_occ_valid_denominator": po_valid,
        "occupied_conflict_rate": float(po_tf / max(po_valid, 1)),
    }


def save_map_preview(path: Path, partial, teacher, reach, dist, clearance, title: str, local_cfg: LocalMapConfig, aux: Optional[Dict[str, np.ndarray]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = local_cfg.channels
    fig, axes = plt.subplots(1, 3 if aux is None else 4, figsize=(14 if aux is None else 18, 5))
    for ax, tensor, name in [(axes[0], partial.tensor, "partial"), (axes[1], teacher.tensor, "teacher")]:
        free = tensor[names.index("free_mask")]
        occ = tensor[names.index("occupied_mask")]
        img = np.zeros((*free.shape, 3), dtype=np.float32)
        img[..., 1] = free * 0.7
        img[..., 0] = occ
        ax.imshow(img, origin="upper")
        ax.set_title(name)
        ax.set_axis_off()
    ax = axes[2]
    free = teacher.tensor[names.index("free_mask")]
    occ = teacher.tensor[names.index("occupied_mask")]
    img = np.zeros((*free.shape, 3), dtype=np.float32)
    img[..., 1] = free * 0.7
    img[..., 0] = occ
    ax.imshow(img, origin="upper")
    n = local_cfg.grid_size
    cr = cc = n // 2
    for k in range(16):
        ang = k * 2.0 * math.pi / 16.0
        x = cc + math.cos(ang) * 42
        y = cr - math.sin(ang) * 42
        color = "yellow" if reach[k] > 0.5 else "white"
        ax.plot([cc, x], [cr, y], color=color, linewidth=1.0)
        ax.text(cc + math.cos(ang) * 46, cr - math.sin(ang) * 46, f"{dist[k]:.2f}/{clearance[k]:.2f}", color=color, fontsize=6)
    ax.set_title("labels")
    ax.set_axis_off()
    if aux is not None:
        ax = axes[3]
        img = np.zeros((*aux["teacher_traversable"].shape, 3), dtype=np.float32)
        img[..., 1] = aux["teacher_traversable"].astype(np.float32)
        img[..., 0] = aux["inflated_obstacle"].astype(np.float32)
        ax.imshow(img, origin="upper")
        ax.set_title("traversable/inflated")
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scan_frame_audit(out: Path, scans_by_robot: Dict[str, List[ScanRecord]], poses_by_robot: Dict[str, Dict[int, PoseRecord]]) -> Dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    audit = {}
    correct_pts = []
    wrong_pts = []
    for robot, scans in scans_by_robot.items():
        unique = []
        seen = set()
        for rec in scans:
            if rec.key in seen or rec.key not in poses_by_robot[robot]:
                continue
            seen.add(rec.key)
            unique.append(rec)
            if len(unique) >= 20:
                break
        apply_spreads = []
        raw_spreads = []
        nn_apply = []
        nn_raw = []
        last_apply = None
        last_raw = None
        for rec in unique:
            pose = poses_by_robot[robot][rec.key].pose
            pts = rec.raw_points[np.isfinite(rec.raw_points).all(axis=1)]
            if len(pts) > 2000:
                pts = pts[:: max(1, len(pts) // 2000)]
            applied = transform_points_local_to_world(pts, pose)
            raw = pts.copy()
            correct_pts.append(applied[:, :2])
            wrong_pts.append(raw[:, :2])
            apply_spreads.append((np.nanmin(applied, axis=0).tolist(), np.nanmax(applied, axis=0).tolist()))
            raw_spreads.append((np.nanmin(raw, axis=0).tolist(), np.nanmax(raw, axis=0).tolist()))
            if last_apply is not None and len(applied) and len(last_apply):
                nn_apply.append(float(np.median(cKDTree(last_apply[:, :2]).query(applied[:, :2], k=1)[0])))
                nn_raw.append(float(np.median(cKDTree(last_raw[:, :2]).query(raw[:, :2], k=1)[0])))
            last_apply = applied
            last_raw = raw
        audit[robot] = {
            "sampled_unique_keys": int(len(unique)),
            "scan_frame_ids": sorted(set(r.scan_frame_id for r in unique)),
            "raw_xyz_bounds_examples": raw_spreads[:3],
            "applied_xyz_bounds_examples": apply_spreads[:3],
            "median_nn_distance_apply_pose": describe(nn_apply),
            "median_nn_distance_no_pose": describe(nn_raw),
            "judgment": "raw scan coordinates are local/sensor-like because xyz ranges are near the sensor while pose positions are global-scale; apply final key pose once",
        }
    def plot_overlay(path: Path, arrays: List[np.ndarray], title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        for i, arr in enumerate(arrays[:40]):
            if len(arr):
                ax.scatter(arr[:, 0], arr[:, 1], s=0.2, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    plot_overlay(out / "scan_overlay_correct.png", correct_pts, "apply final pose once")
    plot_overlay(out / "scan_overlay_wrong_transform.png", wrong_pts, "no pose transform")
    return audit


def run_diagnostics(cfg: Dict[str, object], out: Path, export_full: bool = False) -> Dict[str, object]:
    local_cfg = LocalMapConfig()
    lamp_root = Path(cfg["dataset"]["lamp_root"])
    if out.exists():
        shutil.rmtree(out)
    for sub in ["logs", "previews", "train", "val", "test"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    scans_by_robot = {}
    poses_by_robot = {}
    key_audit = {}
    frames_by_robot = {}
    frame_audit = {}
    for robot in cfg["dataset"]["robots"]:
        scans, poses, audit = audit_and_load_robot(lamp_root, robot, float(cfg["quality"]["max_coord_abs_m"]))
        frames, fa = make_standard_frames(scans, poses, float(cfg["quality"]["max_coord_abs_m"]))
        scans_by_robot[robot] = scans
        poses_by_robot[robot] = poses
        key_audit[robot] = audit
        frames_by_robot[robot] = frames
        frame_audit[robot] = fa
    write_json(out / "logs" / "key_association_audit.json", key_audit)
    write_json(out / "logs" / "standard_frame_audit.json", frame_audit)
    save_standard_frame_index(out / "logs" / "standard_frame_index.npz", frames_by_robot)
    sf_audit = scan_frame_audit(out / "previews", scans_by_robot, poses_by_robot)
    write_json(out / "logs" / "scan_frame_audit.json", sf_audit)

    centers = select_centers(frames_by_robot, cfg)
    split_info = assign_train_val_split(centers, cfg)
    split_by_key = {(c.robot, c.key): c.split for c in centers}
    chosen = centers if export_full else diagnostic_subset(centers, int(cfg["sampling"]["diagnostic_count"]))
    sample_stats = []
    teacher_logs = []
    discard = Counter()
    label_sparse_probe = []
    samples_to_save = []
    for c in chosen:
        if c.split not in {"train", "val"}:
            discard["split_buffer"] += 1
            continue
        partial, p_idxs = build_partial(frames_by_robot[c.robot], c, local_cfg, cfg["partial"])
        teacher, t_idxs, t_log = build_teacher(frames_by_robot, c, split_by_key, local_cfg, cfg)
        names = local_cfg.channels
        raw_label = label_from_mask(teacher.tensor[names.index("free_mask")], teacher.tensor[names.index("occupied_mask")], local_cfg, cfg["labels"])
        if raw_label is None:
            discard["no_raw_label_start"] += 1
            continue
        reach, dist, clear, _, _, _ = raw_label
        aux = None
        label_sparse_probe.append(float(np.mean(reach)))
        metrics = conflict_metrics(partial.tensor, teacher.tensor, local_cfg)
        if metrics["partial_internal_free_and_occupied"] or metrics["teacher_internal_free_and_occupied"]:
            discard["internal_free_occupied_overlap"] += 1
            continue
        if not (np.isfinite(partial.tensor).all() and np.isfinite(teacher.tensor).all()):
            discard["nan_inf"] += 1
            continue
        sample = {
            "center": c,
            "partial": partial,
            "teacher": teacher,
            "reach": reach,
            "dist": dist,
            "clear": clear,
            "partial_indices": p_idxs,
            "teacher_log": t_log,
            "metrics": metrics,
            "aux": aux,
        }
        samples_to_save.append(sample)
        teacher_logs.append({"sample_index": c.index, "sample_split": c.split, "frames": t_log})
        sample_stats.append({
            "sample_index": int(c.index),
            "robot": c.robot,
            "key": int(c.key),
            "split": c.split,
            "repeat_neighbor_count": int(c.repeat_neighbor_count),
            "partial_frame_count": int(len(p_idxs)),
            "teacher_frame_count": int(len(t_idxs)),
            "reachable_direction_count": int(np.sum(reach > 0.5)),
            "reachable_positive_ratio": float(np.mean(reach)),
            **metrics,
        })

    no_reach_fraction = float(np.mean([s["reachable_direction_count"] == 0 for s in sample_stats])) if sample_stats else 1.0
    positive_ratio = float(np.mean([s["reachable_positive_ratio"] for s in sample_stats])) if sample_stats else 0.0
    traversability_enabled = False
    if cfg["labels"].get("enable_traversability_if_sparse", True) and (
        no_reach_fraction > float(cfg["labels"]["sparse_no_reachable_fraction"]) or positive_ratio < float(cfg["labels"]["sparse_positive_ratio"])
    ):
        traversability_enabled = True
        sample_stats = []
        for sample in samples_to_save:
            labeled, aux = traversability_labels(sample["teacher"].tensor, local_cfg, cfg)
            if labeled is None:
                reach = np.zeros(int(cfg["labels"]["directions"]), dtype=np.float32)
                dist = np.zeros_like(reach)
                clear = np.zeros_like(reach)
            else:
                reach, dist, clear, _, _, _ = labeled
            sample["reach"], sample["dist"], sample["clear"], sample["aux"] = reach, dist, clear, aux
            c = sample["center"]
            metrics = sample["metrics"]
            sample_stats.append({
                "sample_index": int(c.index),
                "robot": c.robot,
                "key": int(c.key),
                "split": c.split,
                "repeat_neighbor_count": int(c.repeat_neighbor_count),
                "partial_frame_count": int(len(sample["partial_indices"])),
                "teacher_frame_count": int(len(sample["teacher_log"])),
                "reachable_direction_count": int(np.sum(reach > 0.5)),
                "reachable_positive_ratio": float(np.mean(reach)),
                **metrics,
            })

    for sample in samples_to_save:
        c = sample["center"]
        if export_full:
            rel = Path(c.split) / f"{c.split}_{c.robot}_{c.index:06d}.npz"
            center_pose = np.concatenate([np.array([c.pose.x, c.pose.y, c.pose.z], dtype=np.float32), quat_xyzw_from_yaw(c.pose.yaw)])
            np.savez_compressed(
                out / rel,
                partial_map=sample["partial"].tensor.astype(np.float32),
                teacher_map=sample["teacher"].tensor.astype(np.float32),
                direction_reachability=sample["reach"].astype(np.float32),
                direction_distance=sample["dist"].astype(np.float32),
                direction_clearance=sample["clear"].astype(np.float32),
                center_pose=center_pose,
                stamp_ns=np.int64(c.stamp_ns),
                trajectory_name=np.array(c.robot),
                region_id=np.array(c.region_id),
                split=np.array(c.split),
                partial_frame_count=np.int32(len(sample["partial_indices"])),
                teacher_frame_count=np.int32(len(sample["teacher_log"])),
                partial_point_count=np.int32(sample["partial"].metadata.get("accumulated_points", 0)),
                teacher_point_count=np.int32(sample["teacher"].metadata.get("accumulated_points", 0)),
                config_hash=np.array(config_hash(local_cfg)),
            )
        if len(list((out / "previews").glob("diagnostic_*.png"))) < int(cfg["preview"]["diagnostic_count"]):
            save_map_preview(
                out / "previews" / f"diagnostic_{c.split}_{c.robot}_{c.index:06d}.png",
                sample["partial"],
                sample["teacher"],
                sample["reach"],
                sample["dist"],
                sample["clear"],
                f"{c.robot} key={c.key} split={c.split}",
                local_cfg,
                sample["aux"],
            )

    free_rates = [s["free_conflict_rate"] for s in sample_stats]
    occ_rates = [s["occupied_conflict_rate"] for s in sample_stats]
    reachable_counts = [s["reachable_direction_count"] for s in sample_stats]
    split_counts = Counter(c.split for c in centers)
    min_train_val = nearest_split_distance(centers, "train", "val")
    summary = {
        "mode": "export_full" if export_full else "diagnostic_50",
        "status": "NOT_READY",
        "standard_frame_total": int(sum(len(v) for v in frames_by_robot.values())),
        "candidate_center_total": int(len(centers)),
        "diagnostic_sample_count": int(len(sample_stats)),
        "split_counts_all_centers": dict(split_counts),
        "split_nearest_distance_m": {"train_val": min_train_val, "train_test": None, "val_test": None},
        "teacher_cross_split_violations": int(sum(1 for item in teacher_logs for f in item["frames"] if f["split"] != item["sample_split"])),
        "traversability_enabled": traversability_enabled,
        "conflict_rates": {
            "free_conflict": describe(free_rates),
            "occupied_conflict": describe(occ_rates),
        },
        "internal_overlap_total": {
            "partial": int(sum(s["partial_internal_free_and_occupied"] for s in sample_stats)),
            "teacher": int(sum(s["teacher_internal_free_and_occupied"] for s in sample_stats)),
        },
        "reachability": {
            "samples_with_any_reachable_fraction": float(np.mean([v > 0 for v in reachable_counts])) if reachable_counts else 0.0,
            "no_reachable_fraction": float(np.mean([v == 0 for v in reachable_counts])) if reachable_counts else 1.0,
            "positive_direction_ratio": float(np.mean([s["reachable_positive_ratio"] for s in sample_stats])) if sample_stats else 0.0,
            "reachable_direction_count": describe(reachable_counts),
        },
        "partial_frame_count": describe([s["partial_frame_count"] for s in sample_stats]),
        "teacher_frame_count": describe([s["teacher_frame_count"] for s in sample_stats]),
        "sample_stats": sample_stats,
        "discard_reasons": dict(discard),
    }
    q = cfg["quality"]
    pass_diag = (
        sample_stats
        and summary["internal_overlap_total"]["partial"] == 0
        and summary["internal_overlap_total"]["teacher"] == 0
        and summary["conflict_rates"]["free_conflict"]["median"] < float(q["median_free_conflict_max"])
        and summary["conflict_rates"]["occupied_conflict"]["median"] < float(q["median_occupied_conflict_max"])
        and summary["reachability"]["samples_with_any_reachable_fraction"] >= float(q["required_reachable_sample_fraction"])
        and min_train_val is not None
        and min_train_val >= float(cfg["split"]["preferred_min_split_distance_m"])
        and summary["teacher_cross_split_violations"] == 0
    )
    if pass_diag:
        summary["status"] = "DATASET_READY" if export_full else "DIAGNOSTIC_READY"
    elif sample_stats:
        summary["status"] = "PIPELINE_READY_LABELS_INVALID"
    write_json(out / "logs" / "teacher_frame_selection.json", teacher_logs)
    write_json(out / "logs" / "diagnostic_samples.json", sample_stats)
    write_json(out / "dataset_stats.json", summary)
    write_json(out / "split_regions.json", split_info)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/lamp_structural_dataset_v2.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export-full", action="store_true")
    args = parser.parse_args()
    cfg = read_yaml(Path(args.config))
    out = Path(args.output_dir or (cfg["dataset"]["output_dir"] if args.export_full else cfg["dataset"]["diagnostics_dir"]))
    summary = run_diagnostics(cfg, out, export_full=args.export_full)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
