"""Data-only quality gate for v5 canonical-role and dynamic-event teachers."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from learning.structural_learning.topological_supervision import circular_directional_topology_distance


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values))
        return result
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank(first), rank(second))[0, 1])


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def report(root: Path, split: str, seed: int) -> dict:
    files = sorted((root / split).glob("*.npz"))
    samples = [load(path) for path in files]
    rng = np.random.default_rng(seed)
    triples = []
    for _ in range(min(len(samples) * 30, 60000)):
        first, second = rng.integers(0, len(samples), 2)
        if first == second:
            continue
        a, b = samples[int(first)], samples[int(second)]
        topology = circular_directional_topology_distance(
            (a["direction_traversable"], a["direction_reachable_distance"], a["direction_reachable_area"], a["exit_sector_soft"]),
            (b["direction_traversable"], b["direction_reachable_distance"], b["direction_reachable_area"], b["exit_sector_soft"]),
        )
        role_a, role_b = a["canonical_topological_role_v2"], b["canonical_topological_role_v2"]
        role_distance = float(1.0 - np.dot(role_a, role_b) / max(np.linalg.norm(role_a) * np.linalg.norm(role_b), 1e-8))
        cover_a, cover_b = a["student_input_current"][0] >= 0.5, b["student_input_current"][0] >= 0.5
        inter = np.logical_and(cover_a, cover_b).sum()
        coverage_distance = float(1.0 - inter / max(np.logical_or(cover_a, cover_b).sum(), 1))
        triples.append((topology, role_distance, coverage_distance))
    array = np.asarray(triples, dtype=np.float64)
    dynamic_mask = np.stack([sample["continuous_dynamic_score_mask_v2"] for sample in samples])
    dynamic_values = np.stack([sample["continuous_dynamic_scores_v2"] for sample in samples])
    provenance = np.stack([sample["dynamic_event_provenance_v2"] for sample in samples])
    return {
        "samples": len(samples),
        "pair_count": len(array),
        "role_vs_teacher_topology_spearman": spearman(array[:, 1], array[:, 0]),
        "role_vs_surface_coverage_spearman": spearman(array[:, 1], array[:, 2]),
        "teacher_topology_vs_surface_coverage_spearman": spearman(array[:, 0], array[:, 2]),
        "dynamic_valid_rate": float(dynamic_mask.mean()),
        "dynamic_mean_valid": {name: float(dynamic_values[:, index][dynamic_mask[:, index] > 0].mean()) if np.any(dynamic_mask[:, index] > 0) else None for index, name in enumerate(["turn_strength", "transition_score", "new_branch_appearance", "topological_node_score"])},
        "dynamic_motion_median_valid": [float(np.median(provenance[:, index][dynamic_mask[:, 0] > 0])) if np.any(dynamic_mask[:, 0] > 0) else None for index in range(2)],
        "world_counts": dict(Counter(str(scalar(sample["world"])) for sample in samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = {split: report(args.dataset_root, split, args.seed + index) for index, split in enumerate(["train", "val", "test"])}
    args.output.joinpath("summary.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
