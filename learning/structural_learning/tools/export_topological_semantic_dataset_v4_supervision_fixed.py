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

from learning.structural_learning.topological_metrics import direction_baselines, direction_metric_report
from learning.structural_learning.topological_supervision import canonical_role_descriptor, extract_exit_sectors


SCORE_NAMES = [
    "branch_strength", "main_path_continuity", "turn_strength", "bottleneck_score",
    "openness_score", "transition_score", "topological_node_score",
]
STATIC_NAMES = ["openness_score", "bottleneck_score", "main_path_continuity"]
DYNAMIC_NAMES = ["turn_strength", "transition_score", "new_branch_appearance", "topological_node_score"]


def write_json(path: Path, value: Any) -> None:
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_source(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    return result


def scalar(data: dict[str, Any], name: str, default: Any = "") -> Any:
    value = data.get(name, default)
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def source_cfg_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def dynamic_labels(records: list[dict[str, Any]], max_gap_ns: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(records), len(DYNAMIC_NAMES)), dtype=np.float32)
    masks = np.zeros_like(values)
    by_traj: dict[str, list[int]] = defaultdict(list)
    for index, rec in enumerate(records):
        by_traj[str(rec["trajectory"])].append(index)
    for indices in by_traj.values():
        indices.sort(key=lambda i: int(records[i]["timestamp"]))
        for pos in range(1, len(indices)):
            cur_i, prev_i = indices[pos], indices[pos - 1]
            dt = int(records[cur_i]["timestamp"]) - int(records[prev_i]["timestamp"])
            if dt <= 0 or dt > max_gap_ns:
                continue
            cur = records[cur_i]
            prev = records[prev_i]
            cur_exit = cur["exit_sector_soft"]
            prev_exit = prev["exit_sector_soft"]
            distance_delta = float(np.mean(np.abs(cur["distance"] - prev["distance"])))
            exit_delta = float(np.mean(np.abs(cur_exit - prev_exit)))
            new_branch = float(np.clip(np.sum((cur_exit > 0.5) & (prev_exit <= 0.5)) / 4.0, 0.0, 1.0))
            values[cur_i] = np.asarray([distance_delta, exit_delta, new_branch, max(distance_delta, exit_delta, new_branch)], dtype=np.float32)
            masks[cur_i] = 1.0
    return values, masks


def build_record(path: Path, split: str, cfg: dict[str, Any], history_dt_sec: float) -> dict[str, Any]:
    src = load_source(path)
    direction = src["traversable_direction_distribution"].astype(np.float32)
    distance = src["reachable_distance_distribution"].astype(np.float32)
    area = src["reachable_area_distribution"].astype(np.float32)
    sectors = extract_exit_sectors(direction, distance, direction_count=32, max_sectors=int(cfg["max_exit_sectors"]))
    role = canonical_role_descriptor(sectors["soft"], distance, sectors["widths"], sectors["lengths"], sectors["valid_mask"], int(cfg["canonical_role_dim"]))
    valid_history = src["history_valid_mask"].astype(np.float32)
    t = len(valid_history)
    history_time_deltas = -np.arange(t, 0, -1, dtype=np.float32) * float(history_dt_sec)
    record = {
        "source": src, "path": path, "split": split,
        "world": str(scalar(src, "world")), "trajectory": str(scalar(src, "trajectory")),
        "timestamp": int(scalar(src, "timestamp", 0)), "direction": direction,
        "distance": distance, "area": area, "exit_sector_soft": sectors["soft"],
        "exit_sector_binary": sectors["binary"], "exit_centers": sectors["centers"],
        "exit_widths": sectors["widths"], "exit_lengths": sectors["lengths"],
        "exit_valid_mask": sectors["valid_mask"], "exit_count": np.int64(sectors["count"]),
        "canonical_role": role, "canonical_role_valid": np.float32(1.0),
        "history_time_deltas": history_time_deltas,
        "history_time_delta_approximate": np.asarray(1, dtype=np.uint8),
        "history_dt_sec": np.float32(history_dt_sec),
    }
    return record


def save_record(record: dict[str, Any], path: Path, cfg_hash: str) -> None:
    src = record["source"]
    old_scores = src["continuous_structural_scores"].astype(np.float32)
    old_names = [str(x) for x in src["continuous_structural_score_names"].tolist()]
    score_map = {name: float(old_scores[i]) for i, name in enumerate(old_names)}
    static_scores = np.asarray([score_map.get(name, 0.0) for name in STATIC_NAMES], dtype=np.float32)
    static_mask = np.ones(len(STATIC_NAMES), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        student_input_current=src["student_input_current"].astype(np.float32),
        student_input_history=src["student_input_history"].astype(np.float32),
        history_valid_mask=src["history_valid_mask"].astype(np.float32),
        history_time_deltas=record["history_time_deltas"],
        relative_poses=src["relative_poses"].astype(np.float32),
        direction_traversable=record["direction"],
        direction_reachable_distance=record["distance"],
        direction_reachable_area=record["area"],
        exit_sector_soft=record["exit_sector_soft"],
        exit_sector_binary=record["exit_sector_binary"],
        exit_centers=record["exit_centers"],
        exit_widths=record["exit_widths"],
        exit_lengths=record["exit_lengths"],
        exit_valid_mask=record["exit_valid_mask"],
        exit_count=record["exit_count"],
        canonical_topological_role=record["canonical_role"],
        canonical_role_valid=np.asarray(record["canonical_role_valid"], dtype=np.float32),
        canonicalization_metadata=np.asarray("circular_autocorrelation_plus_sorted_sector_geometry_v1"),
        continuous_static_scores=static_scores,
        continuous_static_score_names=np.asarray(STATIC_NAMES),
        continuous_static_score_mask=static_mask,
        continuous_dynamic_scores=np.zeros(len(DYNAMIC_NAMES), dtype=np.float32),
        continuous_dynamic_score_names=np.asarray(DYNAMIC_NAMES),
        continuous_dynamic_score_mask=np.zeros(len(DYNAMIC_NAMES), dtype=np.float32),
        # Preserve old facts for audit and backward compatibility, but never
        # expose them as the v4 training target without the deprecated marker.
        traversable_direction_distribution=record["direction"],
        reachable_distance_distribution=record["distance"],
        reachable_area_distribution=record["area"],
        independent_exit_distribution=src["independent_exit_distribution"].astype(np.float32),
        topological_role_teacher=src["topological_role_teacher"].astype(np.float32),
        topological_role_teacher_deprecated_for_training=np.asarray(1, dtype=np.uint8),
        rotation_canonical_role_teacher=src["rotation_canonical_role_teacher"].astype(np.float32),
        local_connectivity_data=src["local_connectivity_data"].astype(np.uint8),
        center_pose=src["center_pose"].astype(np.float32),
        timestamp=np.int64(record["timestamp"]),
        stamp_ns=np.int64(scalar(src, "stamp_ns", record["timestamp"])),
        world=np.asarray(record["world"]), trajectory=np.asarray(record["trajectory"]), split=np.asarray(record["split"]),
        source_scan_identifier=np.asarray(scalar(src, "source_scan_identifier", "")),
        student_observation_source=np.asarray("registered_scan"),
        input_contract_version=np.asarray("surface_evidence_v1"),
        teacher_contract_version=np.asarray("local_topological_semantic_teacher_v2"),
        contract_version=np.asarray("topological_semantic_dataset_v4_supervision_fixed"),
        config_hash=np.asarray(cfg_hash), history_order=np.asarray("old_to_new"),
        history_time_delta_approximate=record["history_time_delta_approximate"],
    )


def update_dynamic_fields(root: Path, records: list[dict[str, Any]], cfg_hash: str, max_gap_ns: int) -> None:
    values, masks = dynamic_labels(records, max_gap_ns)
    for index, record in enumerate(records):
        path = root / record["split"] / f"{index:06d}_{record['world']}_{Path(record['path']).stem}.npz"
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        fields["continuous_dynamic_scores"] = values[index]
        fields["continuous_dynamic_score_mask"] = masks[index]
        np.savez_compressed(path, **fields)


def make_pairs(records_by_split: dict[str, list[dict[str, Any]]], out: Path, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_rows = []
    for split, records in records_by_split.items():
        n = len(records)
        if n < 2:
            continue
        coverage = np.asarray([r["source"]["student_input_current"][0] >= 0.5 for r in records]).reshape(n, -1).astype(np.float32)
        roles = np.asarray([r["canonical_role"] for r in records])
        directional = np.asarray([np.r_[r["direction"], r["distance"], r["area"], r["exit_sector_soft"]] for r in records])
        role_n = roles / np.maximum(np.linalg.norm(roles, axis=1, keepdims=True), 1e-8)
        dir_n = directional / np.maximum(np.linalg.norm(directional, axis=1, keepdims=True), 1e-8)
        candidates = []
        for _ in range(min(n * 10, 10000)):
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            inter = float(coverage[i] @ coverage[j])
            surface_distance = float(1.0 - inter / max(float(coverage[i].sum() + coverage[j].sum() - inter), 1.0))
            role_distance = float(1.0 - role_n[i] @ role_n[j])
            directional_distance = float(1.0 - dir_n[i] @ dir_n[j])
            candidates.append((i, j, surface_distance, directional_distance, role_distance))
        if not candidates:
            continue
        surface = np.asarray([x[2] for x in candidates])
        role_d = np.asarray([x[4] for x in candidates])
        for k, (i, j, sd, dd, rd) in enumerate(candidates):
            if sd >= np.percentile(surface, 75) and rd <= np.percentile(role_d, 25):
                kind = "positive_invariance"
            elif sd <= np.percentile(surface, 25) and rd >= np.percentile(role_d, 75):
                kind = "hard_negative"
            else:
                continue
            all_rows.append({"pair_id": f"{split}_{len(all_rows):06d}", "split": split, "sample_a": str(records[i]["path"]), "sample_b": str(records[j]["path"]), "pair_type": kind, "surface_distance": sd, "directional_topology_distance": dd, "canonical_role_distance": rd, "teacher_confidence": float(records[i]["canonical_role_valid"] * records[j]["canonical_role_valid"])})
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in all_rows) + "\n", encoding="utf-8")
    return {"count": len(all_rows), "by_type": dict(Counter(row["pair_type"] for row in all_rows)), "path": str(out / "pairs.jsonl"), "train_only_for_future_training": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = Path(cfg["source_root"])
    output_root = Path(cfg["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    for name in ["train", "val", "test", "contracts", "coverage_pairs", "diagnostics", "previews"]:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    cfg_hash = source_cfg_hash(cfg)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    world_counts = {}
    for split in ["train", "val", "test"]:
        paths = sorted((source_root / split).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"empty source split: {split}")
        timestamps_by_traj: dict[str, list[int]] = defaultdict(list)
        raw = []
        for path in paths:
            src = load_source(path)
            timestamps_by_traj[str(scalar(src, "trajectory"))].append(int(scalar(src, "timestamp")))
            raw.append((path, src))
        diffs = [d for values in timestamps_by_traj.values() for d in np.diff(sorted(values)).tolist() if d > 0]
        dt_ns = int(np.median(diffs)) if diffs else 400_000_000
        records = [build_record(path, split, cfg, dt_ns / 1e9) for path, _ in raw]
        records_by_split[split] = records
        world_counts[split] = dict(Counter(r["world"] for r in records))
        for index, record in enumerate(records):
            save_record(record, output_root / split / f"{index:06d}_{record['world']}_{Path(record['path']).stem}.npz", cfg_hash)
        update_dynamic_fields(output_root, records, cfg_hash, max(2 * dt_ns, dt_ns + 1))
    for source in ["exit_supervision_v2.json", "directional_teacher_v2.json", "canonical_role_teacher_v2.json", "topological_task_decomposition_v2.json"]:
        shutil.copy2(Path(cfg["contracts_root"]) / source, output_root / "contracts" / source)
    arrays_train = [r["direction"] for r in records_by_split["train"]]
    train_direction = np.stack(arrays_train)
    metrics = {}
    for split, records in records_by_split.items():
        labels = np.stack([r["direction"] for r in records])
        metrics[split] = direction_baselines(labels, train_direction)
    write_json(output_root / "diagnostics" / "direction_metric_audit.json", metrics)
    pair_summary = make_pairs(records_by_split, output_root / "coverage_pairs", int(cfg["seed"]))
    exit_stats = {split: {"samples": len(rows), "exit_count": {"mean": float(np.mean([r["exit_count"] for r in rows])), "max": int(max(r["exit_count"] for r in rows))}, "sector_width_bins": [float(np.sum(r["exit_sector_binary"])) for r in rows]} for split, rows in records_by_split.items()}
    write_json(output_root / "dataset_stats.json", {"source_root": str(source_root), "world_counts": world_counts, "counts": {k: len(v) for k, v in records_by_split.items()}, "exit": exit_stats, "pair_summary": pair_summary, "config_hash": cfg_hash})
    write_json(output_root / "input_contract.json", {"version": "surface_evidence_v1", "student_fields": ["student_input_current", "student_input_history", "history_valid_mask", "history_time_deltas", "relative_poses"], "history_order": "old_to_new", "student_observation_source": "registered_scan"})
    write_json(output_root / "teacher_contract.json", {"version": "local_topological_semantic_teacher_v2", "directional_fields": ["direction_traversable", "direction_reachable_distance", "direction_reachable_area", "exit_sector_soft"], "canonical_role": {"field": "canonical_topological_role", "dimension": int(cfg["canonical_role_dim"])}, "deprecated_fields": ["independent_exit_distribution", "topological_role_teacher"], "dynamic_masks": "continuous_dynamic_score_mask"})
    write_json(output_root / "split_definition.json", {"train": ["tunnel", "garage"], "val": ["forest"], "test": ["campus", "indoor"], "source_dataset": str(source_root), "no_resplit": True})
    write_json(output_root / "supervision_quality.json", {"direction_metric_implementation": "learning.structural_learning.topological_metrics.direction_metric_report", "unit_test": "tests/test_direction_metrics.py", "exit_sector_source": "contiguous traversability and reachable-distance bins with circular wrap", "history_time_delta_note": "v3 did not persist per-history timestamps; deltas are inferred from median trajectory sample spacing and marked approximate", "formal_student_source": "registered_scan inherited from v3 online dataset", "limitations": ["dynamic labels are valid only where adjacent same-trajectory teacher samples exist", "campus/indoor retain legacy traversability provenance", "history time deltas are inferred, not raw per-frame timestamps"]})
    print(json.dumps({"output_root": str(output_root), "counts": {k: len(v) for k, v in records_by_split.items()}, "pair_summary": pair_summary}, indent=2))


if __name__ == "__main__":
    main()
