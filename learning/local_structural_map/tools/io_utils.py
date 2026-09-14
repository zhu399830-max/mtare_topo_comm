import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import LocalStructuralMap


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_map_npz(path: str | Path, item: LocalStructuralMap) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tensor=item.tensor.astype(np.float32),
        channel_names=np.asarray(item.channel_names),
        timestamp_ns=np.asarray(item.timestamp_ns, dtype=np.int64),
        source=np.asarray(item.source),
        pose=np.asarray([item.pose.x, item.pose.y, item.pose.z, item.pose.yaw], dtype=np.float32),
        metadata=np.asarray(json.dumps(item.metadata, sort_keys=True)),
    )


def load_map_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "tensor": data["tensor"].astype(np.float32),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
        "timestamp_ns": int(data["timestamp_ns"]),
        "source": str(data["source"]),
        "pose": data["pose"].astype(np.float32),
        "metadata": json.loads(str(data["metadata"])),
    }


def sample_stats(paths: list[Path]) -> dict[str, Any]:
    tensors = [load_map_npz(p)["tensor"] for p in paths]
    if not tensors:
        return {"count": 0}
    arr = np.stack(tensors, axis=0)
    names = load_map_npz(paths[0])["channel_names"]
    stats = {
        "count": len(paths),
        "shape": list(arr.shape[1:]),
        "dtype": str(arr.dtype),
        "channels": names,
        "per_channel": {},
    }
    for i, name in enumerate(names):
        ch = arr[:, i]
        stats["per_channel"][name] = {
            "min": float(np.min(ch)),
            "max": float(np.max(ch)),
            "mean": float(np.mean(ch)),
            "std": float(np.std(ch)),
            "nonzero_ratio": float(np.count_nonzero(ch) / ch.size),
        }
    for name in ["observed_mask", "free_mask", "occupied_mask"]:
        if name in names:
            idx = names.index(name)
            stats[f"{name}_ratio"] = float(np.mean(arr[:, idx] > 0.5))
    if "endpoint_height_range" in names:
        stats["endpoint_height_range_nonzero_ratio"] = float(np.mean(arr[:, names.index("endpoint_height_range")] > 0))
    band_names = [n for n in names if n.startswith("occupancy_z_band_")]
    if band_names:
        stats["z_band_occupancy_ratios"] = {
            n: float(np.mean(arr[:, names.index(n)] > 0.5))
            for n in band_names
        }
    raw_points = []
    valid_points = []
    for p in paths:
        md = load_map_npz(p)["metadata"]
        raw_points.append(int(md.get("raw_points", 0)))
        valid_points.append(int(md.get("valid_points_after_filter", 0)))
    stats["raw_points_per_frame"] = {"min": min(raw_points), "max": max(raw_points), "mean": float(np.mean(raw_points))}
    stats["valid_points_per_frame"] = {"min": min(valid_points), "max": max(valid_points), "mean": float(np.mean(valid_points))}
    voxel_points = []
    for p in paths:
        md = load_map_npz(p)["metadata"]
        if "points_after_voxel_downsample" in md:
            voxel_points.append(int(md["points_after_voxel_downsample"]))
    if voxel_points:
        stats["voxel_points_per_frame"] = {"min": min(voxel_points), "max": max(voxel_points), "mean": float(np.mean(voxel_points))}
    return stats
