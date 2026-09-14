from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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


@dataclass
class Center:
    index: int
    frame_index: int
    robot: str
    stamp_ns: int
    pose: Pose2D
    repeat_neighbor_count: int = 0
    region_id: str = ""
    split: str = ""


def read_jsonable(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def config_hash(config: LocalMapConfig) -> str:
    blob = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)], dtype=np.float32)


def yaw_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def git_snapshot(repo: Path) -> Dict[str, object]:
    def run(args: List[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, stderr=subprocess.STDOUT, text=True).strip()
        except Exception as exc:
            return f"UNAVAILABLE: {type(exc).__name__}: {exc}"

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "diff_stat": run(["git", "diff", "--stat"]),
    }


def load_paired_lamp_frames(lamp_root: Path, robot: str, limits: Dict[str, float]) -> Tuple[List[StandardFrame], Dict[str, object]]:
    bag = lamp_root / "rosbag" / f"{robot}.bag"
    scan_topic = f"/{robot}/lamp/keyed_scans"
    pose_topic = f"/{robot}/lamp/pose_graph_incremental"
    records: Dict[int, Dict[str, object]] = defaultdict(dict)
    topic_counts = Counter()
    key_counts = Counter()
    q_norm_bad = 0
    sync_missing = 0
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in {scan_topic, pose_topic}]
        for conn, timestamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            rec = records[int(timestamp_ns)]
            topic_counts[conn.topic] += 1
            if conn.topic == scan_topic:
                rec["scan_msg"] = msg
                key_counts[int(msg.key)] += 1
            else:
                if len(msg.nodes) != 1:
                    rec["pose_error"] = f"expected 1 incremental node, got {len(msg.nodes)}"
                elif msg.nodes:
                    node = msg.nodes[0]
                    q = node.pose.orientation
                    qn = math.sqrt(float(q.x) ** 2 + float(q.y) ** 2 + float(q.z) ** 2 + float(q.w) ** 2)
                    if abs(qn - 1.0) > 1e-3:
                        q_norm_bad += 1
                    rec["pose_node"] = node

    frames: List[StandardFrame] = []
    invalid = Counter()
    previous_stamp = None
    previous_xy = None
    dt_values = []
    step_values = []
    max_abs = float(limits["max_coord_abs_m"])
    for stamp in sorted(records):
        rec = records[stamp]
        if "scan_msg" not in rec or "pose_node" not in rec:
            sync_missing += 1
            invalid["missing_scan_or_pose"] += 1
            continue
        pose_node = rec["pose_node"]
        pose = pose_from_msg_pose(pose_node.pose)
        pose_stamp_ns = stamp_to_ns(pose_node.header.stamp) or int(stamp)
        points_local = pointcloud2_xyz(rec["scan_msg"].scan)
        finite_local = np.isfinite(points_local).all(axis=1)
        points_local = points_local[finite_local]
        if len(points_local) == 0:
            invalid["empty_scan"] += 1
            continue
        points_world = transform_points_local_to_world(points_local, pose)
        if not np.isfinite(points_world).all():
            invalid["nan_inf_world_points"] += 1
            continue
        if np.max(np.abs(points_world)) > max_abs:
            invalid["world_coord_out_of_range"] += 1
            continue
        if previous_stamp is not None:
            if pose_stamp_ns <= previous_stamp:
                invalid["non_increasing_pose_stamp"] += 1
                continue
            dt_values.append((pose_stamp_ns - previous_stamp) / 1e9)
            xy = np.array([pose.x, pose.y], dtype=np.float32)
            step_values.append(float(np.linalg.norm(xy - previous_xy)))
            previous_xy = xy
        else:
            previous_xy = np.array([pose.x, pose.y], dtype=np.float32)
        previous_stamp = pose_stamp_ns
        frames.append(StandardFrame(
            timestamp_ns=pose_stamp_ns,
            points_world=points_world,
            sensor_origin_world=np.array([pose.x, pose.y, pose.z], dtype=np.float32),
            pose=pose,
            source="LAMP",
            frame_id="map",
            metadata={
                "robot": robot,
                "trajectory_name": robot,
                "bag": str(bag),
                "scan_topic": scan_topic,
                "pose_topic": pose_topic,
                "bag_timestamp_ns": int(stamp),
                "pose_stamp_ns": int(pose_stamp_ns),
                "scan_pose_sync_error_ns": int(stamp - pose_stamp_ns),
                "key": int(rec["scan_msg"].key),
                "raw_points": int(len(points_local)),
            },
        ))

    audit = {
        "bag": str(bag),
        "scan_topic": scan_topic,
        "pose_topic": pose_topic,
        "topic_counts": dict(topic_counts),
        "frames": len(frames),
        "invalid": dict(invalid),
        "sync_missing": sync_missing,
        "duplicate_key_count": int(sum(v - 1 for v in key_counts.values() if v > 1)),
        "quaternion_norm_bad": q_norm_bad,
        "time_range_ns": [frames[0].timestamp_ns, frames[-1].timestamp_ns] if frames else None,
        "dt_sec": describe(dt_values),
        "trajectory_step_m": describe(step_values),
        "sync_error_ns": describe([abs(f.metadata["scan_pose_sync_error_ns"]) for f in frames]),
    }
    return frames, audit


def save_standard_frame_index(path: Path, all_frames: Dict[str, List[StandardFrame]]) -> None:
    rows = []
    for robot, frames in all_frames.items():
        for idx, frame in enumerate(frames):
            rows.append((
                robot,
                idx,
                frame.timestamp_ns,
                frame.pose.x,
                frame.pose.y,
                frame.pose.z,
                frame.pose.yaw,
                int(frame.metadata.get("raw_points", len(frame.points_world))),
                int(frame.metadata.get("scan_pose_sync_error_ns", 0)),
            ))
    dtype = [
        ("robot", "U16"),
        ("frame_index", "i4"),
        ("stamp_ns", "i8"),
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("yaw", "f4"),
        ("raw_points", "i4"),
        ("scan_pose_sync_error_ns", "i8"),
    ]
    arr = np.array(rows, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, standard_frame_index=arr)
    path.chmod(0o444)


def describe(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def select_centers(frames: List[StandardFrame], robot: str, start_index: int, cfg: Dict[str, object]) -> List[Center]:
    out: List[Center] = []
    min_d = float(cfg["sampling"]["min_center_translation_m"])
    min_yaw = math.radians(float(cfg["sampling"]["min_center_yaw_deg"]))
    last_pose: Optional[Pose2D] = None
    for i, frame in enumerate(frames):
        p = frame.pose
        if last_pose is None:
            choose = True
        else:
            dist = math.hypot(p.x - last_pose.x, p.y - last_pose.y)
            choose = dist >= min_d or yaw_delta(p.yaw, last_pose.yaw) >= min_yaw
        if choose:
            out.append(Center(start_index + len(out), i, robot, frame.timestamp_ns, p))
            last_pose = p
    return out


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
    return item, len(idxs), int(item.metadata.get("accumulated_points", 0))


def spatially_uniform_indices(frames: List[StandardFrame], center: Center, teacher_cfg: Dict[str, object]) -> List[int]:
    radius = float(teacher_cfg["radius_m"])
    max_frames = int(teacher_cfg["max_frames"])
    bin_m = float(teacher_cfg["spatial_bin_m"])
    cxy = np.array([center.pose.x, center.pose.y], dtype=np.float32)
    candidates = []
    for i, f in enumerate(frames):
        xy = np.array([f.pose.x, f.pose.y], dtype=np.float32)
        d = float(np.linalg.norm(xy - cxy))
        if d <= radius:
            b = (int(math.floor((xy[0] - cxy[0]) / bin_m)), int(math.floor((xy[1] - cxy[1]) / bin_m)))
            candidates.append((b, d, i))
    by_bin: Dict[Tuple[int, int], List[Tuple[float, int]]] = defaultdict(list)
    for b, d, i in candidates:
        by_bin[b].append((d, i))
    selected = []
    for vals in by_bin.values():
        vals.sort()
        selected.append(vals[0][1])
    if len(selected) < max_frames:
        used = set(selected)
        rest = sorted([(d, i) for _, d, i in candidates if i not in used])
        selected.extend([i for _, i in rest[: max_frames - len(selected)]])
    selected = sorted(selected[:max_frames], key=lambda i: frames[i].timestamp_ns)
    return selected


def build_teacher(frames: List[StandardFrame], center: Center, local_cfg: LocalMapConfig, teacher_cfg: Dict[str, object]):
    builder = LocalStructuralMapBuilder(local_cfg)
    idxs = spatially_uniform_indices(frames, center, teacher_cfg)
    clean_points = []
    for idx in idxs:
        clean = builder._filtered_copy(frames[idx])
        if len(clean.points_world):
            clean_points.append(clean.points_world)
    points = np.concatenate(clean_points, axis=0) if clean_points else np.zeros((0, 3), dtype=np.float32)
    item = builder._build_from_points(frames[center.frame_index], points, frames[center.frame_index].sensor_origin_world)
    item.metadata["accumulated_frames"] = len(idxs)
    item.metadata["accumulated_points"] = int(len(points))
    return item, len(idxs), int(len(points))


def dijkstra_free(free: np.ndarray, start: Tuple[int, int], resolution: float):
    import heapq

    n = free.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float32)
    prev_r = np.full((n, n), -1, dtype=np.int16)
    prev_c = np.full((n, n), -1, dtype=np.int16)
    sr, sc = start
    dist[sr, sc] = 0.0
    heap = [(0.0, sr, sc)]
    neigh = [(-1, 0, resolution), (1, 0, resolution), (0, -1, resolution), (0, 1, resolution),
             (-1, -1, resolution * math.sqrt(2.0)), (-1, 1, resolution * math.sqrt(2.0)),
             (1, -1, resolution * math.sqrt(2.0)), (1, 1, resolution * math.sqrt(2.0))]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > float(dist[r, c]) + 1e-6:
            continue
        for dr, dc, w in neigh:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= n or cc < 0 or cc >= n or not free[rr, cc]:
                continue
            nd = d + w
            if nd < float(dist[rr, cc]):
                dist[rr, cc] = nd
                prev_r[rr, cc] = r
                prev_c[rr, cc] = c
                heapq.heappush(heap, (nd, rr, cc))
    return dist, prev_r, prev_c


def nearest_free_start(free: np.ndarray, local_cfg: LocalMapConfig, radius_m: float) -> Optional[Tuple[int, int]]:
    n = local_cfg.grid_size
    center = np.array([[0.0, 0.0]], dtype=np.float32)
    rows, cols, _ = grid_indices(center, local_cfg.size_m, local_cfg.resolution_m)
    cr, cc = int(rows[0]), int(cols[0])
    yy, xx = np.indices((n, n))
    d = np.hypot((yy - cr) * local_cfg.resolution_m, (xx - cc) * local_cfg.resolution_m)
    mask = free & (d <= radius_m)
    if not mask.any():
        return None
    choices = np.argwhere(mask)
    best = choices[np.argmin(d[mask])]
    return int(best[0]), int(best[1])


def label_directions(teacher_tensor: np.ndarray, local_cfg: LocalMapConfig, label_cfg: Dict[str, object]):
    names = local_cfg.channels
    free = teacher_tensor[names.index("free_mask")] > 0.5
    occupied = teacher_tensor[names.index("occupied_mask")] > 0.5
    free = free & ~occupied
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
    target = float(label_cfg["reachability_distance_m"])
    for k in range(len(reach)):
        mask = reachable_mask & (sectors == k)
        if not mask.any():
            continue
        finite_d = dist[mask]
        far_flat = np.argmax(finite_d)
        coords = np.argwhere(mask)
        rr, cc = coords[far_flat]
        max_d = float(dist[rr, cc])
        distance[k] = np.clip(max_d / max_effective, 0.0, 1.0)
        reach[k] = float(np.max(radial[mask]) >= target)
        path_vals = []
        pr, pc = int(rr), int(cc)
        guard = 0
        while pr >= 0 and pc >= 0 and guard < n * n:
            path_vals.append(float(clearance[pr, pc]))
            if (pr, pc) == start:
                break
            nr, nc = int(prev_r[pr, pc]), int(prev_c[pr, pc])
            pr, pc = nr, nc
            guard += 1
        clear[k] = np.clip((float(np.median(path_vals)) if path_vals else 0.0) / local_cfg.half_size_m, 0.0, 1.0)
    return reach, distance.astype(np.float32), clear.astype(np.float32), start, reachable_mask, clearance


def split_centers(centers: List[Center], cfg: Dict[str, object]) -> Dict[str, object]:
    split_cfg = cfg["split"]
    bin_m = float(split_cfg["region_bin_m"])
    pts = np.array([[c.pose.x, c.pose.y] for c in centers], dtype=np.float32)
    min_xy = pts.min(axis=0)
    region_keys = []
    for c in centers:
        key = (int(math.floor((c.pose.x - min_xy[0]) / bin_m)), int(math.floor((c.pose.y - min_xy[1]) / bin_m)))
        c.region_id = f"r{key[0]}_{key[1]}"
        region_keys.append(key)
    regions = sorted(set(region_keys), key=lambda k: (k[0], k[1]))
    targets = {"train": float(split_cfg["target_train"]), "val": float(split_cfg["target_val"]), "test": float(split_cfg["target_test"])}
    n_regions = len(regions)
    train_end = max(1, int(round(n_regions * targets["train"])))
    val_count = max(1, int(round(n_regions * targets["val"]))) if n_regions >= 3 else 0
    test_count = max(1, n_regions - train_end - val_count) if n_regions >= 3 else 0
    if train_end + val_count + test_count > n_regions:
        train_end = max(1, n_regions - val_count - test_count)
    val_start = train_end
    val_end = min(n_regions, val_start + val_count)
    test_start = max(val_end, n_regions - test_count)
    split_by_region = {}
    for idx, key in enumerate(regions):
        if idx < train_end:
            sp = "train"
        elif idx < val_end:
            sp = "val"
        elif idx >= test_start:
            sp = "test"
        else:
            sp = "buffer"
        split_by_region[key] = sp
    for c, key in zip(centers, region_keys):
        c.split = split_by_region[key]
    apply_split_buffer(centers, float(split_cfg["buffer_m"]))
    return {"region_bin_m": bin_m, "buffer_m": float(split_cfg["buffer_m"]), "regions": {f"r{k[0]}_{k[1]}": split_by_region[k] for k in regions}}


def apply_split_buffer(centers: List[Center], buffer_m: float) -> None:
    split_names = ["train", "val", "test"]
    points = {
        sp: np.array([[c.pose.x, c.pose.y] for c in centers if c.split == sp], dtype=np.float32)
        for sp in split_names
    }
    trees = {sp: cKDTree(arr) for sp, arr in points.items() if len(arr)}
    for c in centers:
        if c.split not in split_names:
            continue
        xy = np.array([c.pose.x, c.pose.y], dtype=np.float32)
        too_close = False
        for other, tree in trees.items():
            if other == c.split:
                continue
            d, _ = tree.query(xy, k=1)
            if float(d) < buffer_m:
                too_close = True
                break
        if too_close:
            c.split = "buffer"


def split_distance(samples: List[Dict[str, object]], a: str, b: str) -> Optional[float]:
    pa = np.array([s["xy"] for s in samples if s["split"] == a], dtype=np.float32)
    pb = np.array([s["xy"] for s in samples if s["split"] == b], dtype=np.float32)
    if len(pa) == 0 or len(pb) == 0:
        return None
    return float(cKDTree(pa).query(pb, k=1)[0].min())


def map_ratios(tensor: np.ndarray, cfg: LocalMapConfig) -> Dict[str, float]:
    total = float(cfg.grid_size * cfg.grid_size)
    names = cfg.channels
    return {
        "observed": float(np.mean(tensor[names.index("observed_mask")] > 0.5)),
        "free": float(np.mean(tensor[names.index("free_mask")] > 0.5)),
        "occupied": float(np.mean(tensor[names.index("occupied_mask")] > 0.5)),
    }


def save_preview(path: Path, sample: Dict[str, object], local_cfg: LocalMapConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    teacher = sample["teacher_map"]
    names = local_cfg.channels
    free = teacher[names.index("free_mask")]
    occ = teacher[names.index("occupied_mask")]
    img = np.zeros((*free.shape, 3), dtype=np.float32)
    img[..., 1] = free * 0.7
    img[..., 0] = occ
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="upper")
    n = local_cfg.grid_size
    cr = cc = n // 2
    for k in range(16):
        ang = k * 2.0 * math.pi / 16.0
        x = cc + math.cos(ang) * 42
        y = cr - math.sin(ang) * 42
        color = "yellow" if sample["direction_reachability"][k] > 0.5 else "white"
        ax.plot([cc, x], [cr, y], color=color, linewidth=1.0, alpha=0.8)
        tx = cc + math.cos(ang) * 47
        ty = cr - math.sin(ang) * 47
        ax.text(tx, ty, f'{sample["direction_distance"][k]:.2f}/{sample["direction_clearance"][k]:.2f}', color=color, fontsize=6)
    ax.set_title(f'{sample["trajectory_name"]} {sample["split"]} {sample["stamp_ns"]}')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/lamp_structural_dataset_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-centers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    cfg = read_jsonable(Path(args.config))
    out = Path(args.output_dir or cfg["dataset"]["output_dir"])
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace this derived dataset directory")
        shutil.rmtree(out)
    for sub in ["train", "val", "test", "previews", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    local_cfg = LocalMapConfig()
    chash = config_hash(local_cfg)
    lamp_root = Path(cfg["dataset"]["lamp_root"])
    shutil.copy2(args.config, out / "logs" / "config.yaml")

    all_frames: Dict[str, List[StandardFrame]] = {}
    frame_audits = {}
    center_counts = {}
    centers: List[Center] = []
    for robot in cfg["dataset"]["robots"]:
        frames, audit = load_paired_lamp_frames(lamp_root, robot, cfg["quality"])
        all_frames[robot] = frames
        frame_audits[robot] = audit
        selected = select_centers(frames, robot, len(centers), cfg)
        center_counts[robot] = len(selected)
        centers.extend(selected)
    save_standard_frame_index(out / "logs" / "standard_frame_index.npz", all_frames)
    if args.limit_centers:
        centers = centers[: args.limit_centers]

    pts = np.array([[c.pose.x, c.pose.y] for c in centers], dtype=np.float32)
    tree = cKDTree(pts) if len(pts) else None
    repeat_positions = 0
    if tree is not None:
        for c, xy in zip(centers, pts):
            near = tree.query_ball_point(xy, r=float(cfg["sampling"]["repeat_neighbor_radius_m"]))
            c.repeat_neighbor_count = max(0, len(near) - 1)
            if c.repeat_neighbor_count:
                repeat_positions += 1
    split_regions = split_centers(centers, cfg)

    samples_for_stats = []
    discard = Counter()
    preview_budget = int(cfg["preview"]["count"])
    manifest_samples = []
    qcfg = cfg["quality"]
    for c in centers:
        if c.split == "buffer":
            discard["split_buffer"] += 1
            continue
        frames = all_frames[c.robot]
        try:
            partial, p_frames, p_points = build_partial(frames, c, local_cfg, cfg["partial"])
            teacher, t_frames, t_points = build_teacher(frames, c, local_cfg, cfg["teacher"])
            labels = label_directions(teacher.tensor, local_cfg, cfg["labels"])
        except Exception as exc:
            discard[f"exception_{type(exc).__name__}"] += 1
            continue
        if labels is None:
            discard["no_center_free"] += 1
            continue
        reach, dist, clearance, _, _, _ = labels
        if not (np.isfinite(partial.tensor).all() and np.isfinite(teacher.tensor).all() and np.isfinite(dist).all() and np.isfinite(clearance).all()):
            discard["nan_inf"] += 1
            continue
        pr = map_ratios(partial.tensor, local_cfg)
        tr = map_ratios(teacher.tensor, local_cfg)
        if pr["observed"] < float(qcfg["min_partial_observed_ratio"]):
            discard["partial_observed_low"] += 1
            continue
        if tr["observed"] < max(float(qcfg["min_teacher_observed_ratio"]), pr["observed"]):
            discard["teacher_observed_low"] += 1
            continue
        free = teacher.tensor[local_cfg.channels.index("free_mask")] > 0.5
        n = local_cfg.grid_size
        yy, xx = np.indices((n, n))
        center_mask = np.hypot((yy - n // 2) * local_cfg.resolution_m, (xx - n // 2) * local_cfg.resolution_m) <= 3.0
        if int(np.sum(free & center_mask)) < int(qcfg["min_center_free_cells_3m"]):
            discard["center_free_low"] += 1
            continue
        p_occ = partial.tensor[local_cfg.channels.index("occupied_mask")] > 0.5
        p_free = partial.tensor[local_cfg.channels.index("free_mask")] > 0.5
        t_occ = teacher.tensor[local_cfg.channels.index("occupied_mask")] > 0.5
        t_free = teacher.tensor[local_cfg.channels.index("free_mask")] > 0.5
        free_occ_conflict = float(np.sum(p_free & t_occ) / max(1, np.sum(p_free)))
        occ_free_conflict = float(np.sum(p_occ & t_free) / max(1, np.sum(p_occ)))
        occ_iou = float(np.sum(p_occ & t_occ) / max(1, np.sum(p_occ | t_occ)))
        if free_occ_conflict > float(qcfg["max_free_occupied_conflict_ratio"]):
            discard["partial_free_teacher_occupied_conflict"] += 1
            continue
        center_pose = np.concatenate([np.array([c.pose.x, c.pose.y, c.pose.z], dtype=np.float32), quat_xyzw_from_yaw(c.pose.yaw)])
        fname = f"{c.split}_{c.robot}_{c.index:06d}.npz"
        rel_path = Path(c.split) / fname
        np.savez_compressed(
            out / rel_path,
            partial_map=partial.tensor.astype(np.float32),
            teacher_map=teacher.tensor.astype(np.float32),
            direction_reachability=reach.astype(np.float32),
            direction_distance=dist.astype(np.float32),
            direction_clearance=clearance.astype(np.float32),
            center_pose=center_pose,
            stamp_ns=np.int64(c.stamp_ns),
            trajectory_name=np.array(c.robot),
            region_id=np.array(c.region_id),
            split=np.array(c.split),
            partial_frame_count=np.int32(p_frames),
            teacher_frame_count=np.int32(t_frames),
            partial_point_count=np.int32(p_points),
            teacher_point_count=np.int32(t_points),
            config_hash=np.array(chash),
        )
        sample_stat = {
            "file": str(rel_path),
            "split": c.split,
            "robot": c.robot,
            "trajectory_name": c.robot,
            "stamp_ns": int(c.stamp_ns),
            "xy": [float(c.pose.x), float(c.pose.y)],
            "region_id": c.region_id,
            "repeat_neighbor_count": c.repeat_neighbor_count,
            "partial_frame_count": p_frames,
            "teacher_frame_count": t_frames,
            "partial_point_count": p_points,
            "teacher_point_count": t_points,
            "partial_ratios": pr,
            "teacher_ratios": tr,
            "partial_occ_teacher_occ_iou": occ_iou,
            "partial_free_teacher_occ_conflict": free_occ_conflict,
            "partial_occ_teacher_free_conflict": occ_free_conflict,
            "reachability_positive_ratio": float(np.mean(reach)),
            "direction_distance_mean": float(np.mean(dist)),
            "direction_clearance_mean": float(np.mean(clearance)),
        }
        samples_for_stats.append({**sample_stat, "xy": np.array(sample_stat["xy"], dtype=np.float32)})
        manifest_samples.append(sample_stat)
        if preview_budget > 0:
            save_preview(out / "previews" / f"{Path(fname).stem}.png", {
                "teacher_map": teacher.tensor,
                "direction_reachability": reach,
                "direction_distance": dist,
                "direction_clearance": clearance,
                "trajectory_name": c.robot,
                "split": c.split,
                "stamp_ns": int(c.stamp_ns),
            }, local_cfg)
            preview_budget -= 1

    split_counts = Counter(s["split"] for s in samples_for_stats)
    robot_counts = Counter(s["robot"] for s in samples_for_stats)
    def avg(key: str) -> float:
        return float(np.mean([s[key] for s in samples_for_stats])) if samples_for_stats else 0.0
    def avg_ratio(kind: str, name: str) -> float:
        return float(np.mean([s[f"{kind}_ratios"][name] for s in samples_for_stats])) if samples_for_stats else 0.0

    split_bounds = {}
    for sp in ["train", "val", "test"]:
        arr = np.array([s["xy"] for s in samples_for_stats if s["split"] == sp], dtype=np.float32)
        split_bounds[sp] = None if len(arr) == 0 else {"min_xy": arr.min(axis=0).tolist(), "max_xy": arr.max(axis=0).tolist()}

    nearest_distances = {
        "train_val": split_distance(samples_for_stats, "train", "val"),
        "train_test": split_distance(samples_for_stats, "train", "test"),
        "val_test": split_distance(samples_for_stats, "val", "test"),
    }
    split_reliable = all(v is not None and v >= float(cfg["split"]["buffer_m"]) for v in nearest_distances.values())
    reachability_mean = np.mean([s["reachability_positive_ratio"] for s in samples_for_stats]).item() if samples_for_stats else 0.0
    if not samples_for_stats or reachability_mean <= 0.0:
        status = "NOT_READY"
    elif len(samples_for_stats) >= 300 and all(split_counts[s] > 0 for s in ["train", "val", "test"]) and split_reliable:
        status = "DATASET_READY"
    else:
        status = "PIPELINE_READY_DATA_INSUFFICIENT"

    stats = {
        "status": status,
        "standard_frame_total": int(sum(len(v) for v in all_frames.values())),
        "candidate_centers_total": int(sum(center_counts.values())),
        "candidate_centers_by_robot": center_counts,
        "repeat_visit_center_count": int(repeat_positions),
        "valid_sample_count": int(len(samples_for_stats)),
        "split_counts": dict(split_counts),
        "robot_counts": dict(robot_counts),
        "partial_frame_count_mean": avg("partial_frame_count"),
        "teacher_frame_count_mean": avg("teacher_frame_count"),
        "partial_point_count_mean": avg("partial_point_count"),
        "teacher_point_count_mean": avg("teacher_point_count"),
        "partial_ratios_mean": {k: avg_ratio("partial", k) for k in ["observed", "free", "occupied"]},
        "teacher_ratios_mean": {k: avg_ratio("teacher", k) for k in ["observed", "free", "occupied"]},
        "observed_lift_mean": avg_ratio("teacher", "observed") - avg_ratio("partial", "observed"),
        "reachability_positive_ratio_16": reachability_mean,
        "direction_distance": describe([s["direction_distance_mean"] for s in samples_for_stats]),
        "direction_clearance": describe([s["direction_clearance_mean"] for s in samples_for_stats]),
        "partial_teacher_conflicts": {
            "occupied_iou_mean": avg("partial_occ_teacher_occ_iou"),
            "partial_free_teacher_occupied_mean": avg("partial_free_teacher_occ_conflict"),
            "partial_occupied_teacher_free_mean": avg("partial_occ_teacher_free_conflict"),
        },
        "split_bounds": split_bounds,
        "split_nearest_distance_m": nearest_distances,
        "split_reliable_12m_buffer": split_reliable,
        "discard_reasons": dict(discard),
        "config_hash": chash,
        "builder_class": "learning.local_structural_map.builder.LocalStructuralMapBuilder",
        "channel_names": list(local_cfg.channels),
        "git": git_snapshot(repo),
        "docker_image_identifier": os.environ.get("HOSTNAME", "UNAVAILABLE"),
        "ros_version": os.environ.get("ROS_DISTRO", "UNAVAILABLE"),
        "package_list": "UNAVAILABLE",
    }
    manifest = {
        "dataset": cfg["dataset"],
        "local_map_config": local_cfg.to_dict(),
        "config_hash": chash,
        "frame_audit": frame_audits,
        "samples": manifest_samples,
        "stats_file": "dataset_stats.json",
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "split_regions.json", split_regions)
    write_json(out / "dataset_stats.json", stats)
    write_json(out / "logs" / "standard_frame_audit.json", frame_audits)
    write_json(out / "logs" / "discard_reasons.json", dict(discard))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
