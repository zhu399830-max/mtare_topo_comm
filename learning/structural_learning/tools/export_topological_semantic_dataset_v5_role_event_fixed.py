"""Regenerate only v5 role/event supervision from the fixed v4 dataset.

Student observations and split membership are copied byte-for-byte at the
field level.  The teacher is re-expressed from existing traversability facts;
no M-TARE mesh, future scan, or test result is used for student input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from learning.structural_learning.topological_supervision import (
    aligned_directional_change,
    canonical_role_layout_v2,
    circular_directional_topology_distance,
)


DYNAMIC_NAMES = ["turn_strength", "transition_score", "new_branch_appearance", "topological_node_score"]


def scalar(value: np.ndarray | Any) -> Any:
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def pose_xy_yaw(data: dict[str, np.ndarray]) -> tuple[float, float, float]:
    pose = data["center_pose"].astype(np.float64)
    qz, qw = float(pose[5]), float(pose[6])
    return float(pose[0]), float(pose[1]), float(2.0 * math.atan2(qz, qw))


def facts(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (record["direction"], record["distance"], record["area"], record["exit"])


def event_labels(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((len(records), len(DYNAMIC_NAMES)), dtype=np.float32)
    masks = np.zeros_like(values)
    provenance = np.zeros((len(records), 3), dtype=np.float32)  # prev motion, next motion, persistence
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_trajectory[record["trajectory"]].append(index)
    max_gap = int(float(config["max_gap_sec"]) * 1e9)
    for indices in by_trajectory.values():
        indices.sort(key=lambda i: records[i]["timestamp"])
        for position in range(1, len(indices) - 1):
            pi, ci, ni = indices[position - 1], indices[position], indices[position + 1]
            previous, current, following = records[pi], records[ci], records[ni]
            if current["timestamp"] - previous["timestamp"] <= 0 or following["timestamp"] - current["timestamp"] <= 0:
                continue
            if current["timestamp"] - previous["timestamp"] > max_gap or following["timestamp"] - current["timestamp"] > max_gap:
                continue
            px, py, pyaw = previous["pose"]
            cx, cy, cyaw = current["pose"]
            nx, ny, nyaw = following["pose"]
            motion_previous = math.hypot(cx - px, cy - py)
            motion_next = math.hypot(nx - cx, ny - cy)
            if not (float(config["min_motion_m"]) <= motion_previous <= float(config["max_motion_m"])):
                continue
            if not (float(config["min_motion_m"]) <= motion_next <= float(config["max_motion_m"])):
                continue
            # Teacher change is measured after heading alignment.  Persistence
            # requires the changed state to continue into the next moved sample.
            prev_to_cur = aligned_directional_change(previous["distance"], current["distance"], cyaw - pyaw)
            cur_to_next = aligned_directional_change(current["distance"], following["distance"], nyaw - cyaw)
            exit_previous = aligned_directional_change(previous["exit"], current["exit"], cyaw - pyaw)
            exit_following = aligned_directional_change(current["exit"], following["exit"], nyaw - cyaw)
            persistent = float(min(prev_to_cur, cur_to_next) >= float(config["min_persistent_change"]))
            # New branches must first appear after movement and remain open one
            # step later; transient sector jitter has no valid event label.
            previous_exit = np.roll(previous["exit"], int(np.rint((cyaw - pyaw) / (2.0 * math.pi) * 32)))
            current_exit = current["exit"]
            following_exit = np.roll(following["exit"], int(np.rint((nyaw - cyaw) / (2.0 * math.pi) * 32)))
            new_branch = float(np.mean((previous_exit < 0.5) & (current_exit >= 0.5) & (following_exit >= 0.5)))
            heading_turn = abs(math.atan2(math.sin(cyaw - pyaw), math.cos(cyaw - pyaw))) / math.pi
            transition = float(np.clip(max(prev_to_cur, exit_previous) * persistent, 0.0, 1.0))
            values[ci] = np.asarray([heading_turn * persistent, transition, new_branch, max(transition, new_branch)], dtype=np.float32)
            masks[ci] = 1.0
            provenance[ci] = np.asarray([motion_previous, motion_next, persistent], dtype=np.float32)
    return values, masks, provenance


def coverage_mask(record: dict[str, Any]) -> np.ndarray:
    return record["data"]["student_input_current"][0] >= 0.5


def surface_distance(first: np.ndarray, second: np.ndarray) -> float:
    inter = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return 1.0 - inter / max(union, 1)


def pair_candidates(records: list[dict[str, Any]], config: dict[str, Any], rng: np.random.Generator) -> list[tuple[int, int, float, float, int]]:
    candidates: list[tuple[int, int, float, float, int]] = []
    trials = min(len(records) * int(config["pair_trials_per_sample"]), int(config["pair_max_trials"]))
    for _ in range(trials):
        first, second = rng.integers(0, len(records), 2)
        if first == second:
            continue
        a, b = records[int(first)], records[int(second)]
        sd = surface_distance(coverage_mask(a), coverage_mask(b))
        td = circular_directional_topology_distance(facts(a), facts(b))
        count_delta = abs(int(a["exit_count"]) - int(b["exit_count"]))
        candidates.append((int(first), int(second), sd, td, count_delta))
    return candidates


def calibrate_pair_thresholds(train_records: list[dict[str, Any]], config: dict[str, Any], seed: int) -> dict[str, float | int]:
    """Freeze pair thresholds from training teacher facts only."""
    candidates = pair_candidates(train_records, config, np.random.default_rng(seed))
    if not candidates:
        raise ValueError("cannot calibrate pair thresholds from empty train candidates")
    surface = np.asarray([row[2] for row in candidates])
    topology = np.asarray([row[3] for row in candidates])
    positive_topology = float(np.quantile(topology, float(config["positive_topology_quantile"])))
    negative_topology = max(
        float(np.quantile(topology, float(config["negative_topology_quantile"]))),
        positive_topology + float(config["minimum_topology_gap"]),
    )
    return {
        "positive_min_surface_distance": float(np.quantile(surface, float(config["positive_surface_quantile"]))),
        "positive_max_topology_distance": positive_topology,
        "positive_max_exit_count_delta": int(config["positive_max_exit_count_delta"]),
        "negative_max_surface_distance": float(np.quantile(surface, float(config["negative_surface_quantile"]))),
        "negative_min_topology_distance": negative_topology,
        "calibration_split": "train",
        "calibration_candidate_count": len(candidates),
    }


def make_pairs(records_by_split: dict[str, list[dict[str, Any]]], output: Path, config: dict[str, Any], seed: int) -> dict[str, Any]:
    thresholds = calibrate_pair_thresholds(records_by_split["train"], config, seed)
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        candidates = pair_candidates(records, config, rng)
        for first, second, sd, td, count_delta in candidates:
            if sd >= float(thresholds["positive_min_surface_distance"]) and td <= float(thresholds["positive_max_topology_distance"]) and count_delta <= int(thresholds["positive_max_exit_count_delta"]):
                kind = "positive_invariance"
            elif sd <= float(thresholds["negative_max_surface_distance"]) and td >= float(thresholds["negative_min_topology_distance"]):
                kind = "hard_negative"
            else:
                continue
            a, b = records[first], records[second]
            rows.append({
                "pair_id": f"{split}_{len(rows):07d}", "split": split, "pair_type": kind,
                "sample_a": a["output_name"], "sample_b": b["output_name"],
                "surface_distance": sd, "directional_topology_distance": td,
                "exit_count_delta": count_delta,
                "teacher_confidence": 1.0,
                "construction": "teacher_fact_thresholds_not_descriptor_percentiles",
            })
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return {"count": len(rows), "by_split": {split: dict(Counter(row["pair_type"] for row in rows if row["split"] == split)) for split in records_by_split}, "selection": thresholds, "selection_provenance": "thresholds calibrated from train teacher facts and online coverage only"}


def json_dump(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, dict): return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, list): return [convert(v) for v in item]
        if isinstance(item, np.generic): return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root, output_root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    for name in ["train", "val", "test", "contracts", "coverage_pairs", "diagnostics", "previews"]:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        rows: list[dict[str, Any]] = []
        for index, path in enumerate(sorted((source_root / split).glob("*.npz"))):
            data = load(path)
            direction = data["direction_traversable"].astype(np.float32)
            distance = data["direction_reachable_distance"].astype(np.float32)
            area = data["direction_reachable_area"].astype(np.float32)
            exit_soft = data["exit_sector_soft"].astype(np.float32)
            role, canonical_meta = canonical_role_layout_v2(direction, distance, area, exit_soft)
            world, trajectory = str(scalar(data["world"])), str(scalar(data["trajectory"]))
            name = f"{index:06d}_{world}_{path.stem}.npz"
            rows.append({"data": data, "source_path": path, "output_name": name, "world": world, "trajectory": trajectory,
                         "timestamp": int(scalar(data["timestamp"])), "direction": direction, "distance": distance, "area": area,
                         "exit": exit_soft, "exit_count": int(scalar(data["exit_count"])), "role": role,
                         "canonical_meta": canonical_meta, "pose": pose_xy_yaw(data)})
        by_split[split] = rows
        values, masks, provenance = event_labels(rows, cfg["dynamic_events"])
        for index, row in enumerate(rows):
            fields = dict(row["data"])
            fields["canonical_topological_role_v2"] = row["role"].astype(np.float32)
            fields["canonical_topological_role_v2_valid"] = np.asarray(1, dtype=np.float32)
            fields["canonical_topological_role_v1_deprecated_for_training"] = np.asarray(1, dtype=np.uint8)
            fields["canonical_role_v2_metadata"] = np.asarray(json.dumps(row["canonical_meta"], sort_keys=True))
            fields["continuous_dynamic_scores_v2"] = values[index]
            fields["continuous_dynamic_score_mask_v2"] = masks[index]
            fields["dynamic_event_provenance_v2"] = provenance[index]
            fields["continuous_dynamic_score_names_v2"] = np.asarray(DYNAMIC_NAMES)
            fields["teacher_contract_version"] = np.asarray("local_topological_semantic_teacher_v5_role_event")
            fields["contract_version"] = np.asarray("topological_semantic_dataset_v5_role_event_fixed")
            fields["config_hash_v5"] = np.asarray(config_hash)
            np.savez_compressed(output_root / split / row["output_name"], **fields)
    pair_summary = make_pairs(by_split, output_root / "coverage_pairs", cfg["pairs"], int(cfg["seed"]))
    shutil.copy2(Path(cfg["contracts_root"]) / "canonical_role_teacher_v3.json", output_root / "contracts" / "canonical_role_teacher_v3.json")
    shutil.copy2(Path(cfg["contracts_root"]) / "topological_task_decomposition_v3.json", output_root / "contracts" / "topological_task_decomposition_v3.json")
    json_dump(output_root / "dataset_stats.json", {"counts": {key: len(value) for key, value in by_split.items()}, "worlds": {key: dict(Counter(row["world"] for row in value)) for key, value in by_split.items()}, "pair_summary": pair_summary, "config_hash": config_hash})
    json_dump(output_root / "input_contract.json", {"copied_from": str(source_root), "student_observation_source": "registered_scan", "no_student_input_reconstruction": True, "history_order": "old_to_new"})
    json_dump(output_root / "teacher_contract.json", {"role_field": "canonical_topological_role_v2", "role_dimension": 128, "dynamic_fields": DYNAMIC_NAMES, "dynamic_validity": cfg["dynamic_events"], "role_source": "complete traversability teacher directional facts", "test_not_used_for_parameters": True})
    json_dump(output_root / "split_definition.json", {"train": ["tunnel", "garage"], "val": ["forest"], "test": ["campus", "indoor"], "inherited": True})
    print(json.dumps({"output": str(output_root), "counts": {key: len(value) for key, value in by_split.items()}, "pairs": pair_summary}, indent=2))


if __name__ == "__main__":
    main()
