#!/usr/bin/env python3
"""Select the consensus/metric association contract using C01-C08 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mtare_topo.evaluation.gse_factorized_qualification import (
    runtime_candidate_pairs,
    select_consensus_metric_configuration,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_factorized_association import (
    FactorizedAssociationVerifier,
    factorized_pair_features,
    learned_geometry_profiles,
)
from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references,
)


TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [row for row in rows if str(row["parent_id"]).endswith(tuple(f"_C{i:02d}" for i in range(1, 9)))]
    if len(rows) != 188_126:
        raise RuntimeError(f"C01-C08 Teacher population drift: {len(rows)}")
    return rows


def _world_rows(path: Path, rows: list[dict]) -> np.ndarray:
    wanted = {int(row["global_sequence_index"]) for row in rows}
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["global_sequence_index"])
            if index in wanted:
                mapping[index] = int(row["world_sequence_row"])
    if set(mapping) != wanted:
        raise RuntimeError("C01-C08 sequence metadata identity drift")
    return np.asarray([mapping[int(row["global_sequence_index"])] for row in rows], dtype=np.int64)


def _score(
    capacity: Path,
    seed: int,
    features: np.ndarray,
) -> tuple[np.ndarray, float]:
    base = capacity / f"artifacts/models/seed{seed}/full_route_conditioned"
    with np.load(base / "normalization.npz", allow_pickle=False) as archive:
        mean = archive["mean"].astype(np.float32)
        std = archive["std"].astype(np.float32)
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "gse_factorized_association_checkpoint_v1"
        or checkpoint.get("seed") != seed
        or checkpoint.get("include_route_geometry") is not True
    ):
        raise RuntimeError(f"frozen association checkpoint drift: seed {seed}")
    model = FactorizedAssociationVerifier(include_route_geometry=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalized = ((features - mean) / std).astype(np.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(normalized))).numpy().astype(np.float64)
    summary = load_json(capacity / f"artifacts/models/seed{seed}/summary.json")
    selected = summary["full_route_conditioned"]["selection"]["threshold_selection"]
    if selected is None:
        raise RuntimeError(f"frozen association threshold missing: seed {seed}")
    return scores, float(selected["threshold"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--capacity-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
        parser.add_argument(f"--tokens{seed}", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = _read_rows(args.teacher.resolve())
    global_index = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("pair-cache/Teacher identity drift")
        association_valid = archive["association_valid"].astype(bool)
        sensor_xyz_m = archive["sensor_xyz_m"].astype(np.float64)
    all_world_rows = _world_rows(args.sequence_manifest.resolve(), rows)
    selection_indices = np.asarray([
        index for index, row in enumerate(rows)
        if str(row["parent_id"]).endswith(("_C07", "_C08"))
    ], dtype=np.int64)
    if len(selection_indices) != 45_942:
        raise RuntimeError("C07-C08 sequence population drift")
    runtime = runtime_candidate_pairs(
        [rows[index] for index in selection_indices],
        association_valid[selection_indices], sensor_xyz_m[selection_indices],
        all_world_rows[selection_indices], maximum_distance_m=16.0,
    )
    if (
        int(runtime["decision_queries"]) != 8_839
        or int(runtime["queries_with_candidate"]) != 8_604
        or int(runtime["queries_with_positive"]) != 8_565
        or len(runtime["label"]) != 9_380
        or int(np.sum(runtime["label"])) != 8_565
    ):
        raise RuntimeError("C07-C08 runtime selection population drift")
    runtime["left"] = selection_indices[runtime["left"]]
    runtime["right"] = selection_indices[runtime["right"]]

    balanced_votes = []
    runtime_votes = []
    thresholds = []
    balanced_label = balanced_family = physical = None
    runtime_scores = []
    for seed in range(3):
        observation = np.load(getattr(args, f"observation{seed}").resolve(), allow_pickle=False)
        if observation.shape != (188_126, 146):
            raise RuntimeError(f"unified observation shape drift: seed {seed}")
        with np.load(getattr(args, f"tokens{seed}").resolve(), allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_index):
                raise RuntimeError(f"token identity drift: seed {seed}")
            token = {name: archive[name] for name in TOKEN_KEYS}
        references, history_mask = causal_history_row_references(
            np.asarray([row["traversal_id"] for row in rows]),
            np.asarray([row["sequence_index"] for row in rows], dtype=np.int64),
        )
        profile = learned_geometry_profiles(observation, references, history_mask)
        features = factorized_pair_features(
            observation, token, profile, runtime["left"], runtime["right"],
            include_route_geometry=True,
        )
        score, threshold = _score(args.capacity_run.resolve(), seed, features)
        thresholds.append(threshold)
        runtime_scores.append(score)
        runtime_votes.append(score >= threshold)
        balanced_path = args.capacity_run.resolve() / f"artifacts/models/seed{seed}/full_route_conditioned/selection_outputs.npz"
        with np.load(balanced_path, allow_pickle=False) as archive:
            if balanced_label is None:
                balanced_label = archive["label"].astype(np.uint8)
                balanced_family = archive["family"].astype(str)
                physical = archive["physical_positive"].astype(bool)
            elif not (
                np.array_equal(balanced_label, archive["label"])
                and np.array_equal(balanced_family, archive["family"])
                and np.array_equal(physical, archive["physical_positive"])
            ):
                raise RuntimeError("balanced selection population differs across seeds")
            balanced_votes.append(archive["score"].astype(np.float64) >= threshold)
    assert balanced_label is not None and balanced_family is not None and physical is not None
    selection = select_consensus_metric_configuration(
        np.stack(balanced_votes), balanced_label, balanced_family, physical,
        np.stack(runtime_votes), runtime["label"], runtime["family"], runtime["distance_m"],
    )
    chosen = selection["selected"]
    if int(chosen["votes_required"]) != 2 or float(chosen["distance_cap_m"]) != 4.0:
        raise RuntimeError(f"pre-registered C07-C08 selection drift: {chosen['votes_required']}/{chosen['distance_cap_m']}")
    calibration = {
        "schema_version": "gse_factorized_consensus_metric_calibration_v1",
        "selection_worlds": 20,
        "selection_sequences": 45_942,
        "runtime_decision_queries": int(runtime["decision_queries"]),
        "runtime_pairs": len(runtime["label"]),
        "runtime_positive_pairs": int(np.sum(runtime["label"])),
        "runtime_negative_pairs": int(np.sum(1 - runtime["label"])),
        "votes_required": int(chosen["votes_required"]),
        "distance_cap_m": float(chosen["distance_cap_m"]),
        "frozen_seed_thresholds": thresholds,
        "selection_rule": selection["selection_rule"],
        "selection_margin": selection["selection_margin"],
        "qualifying_configurations": int(selection["qualifying_configurations"]),
        "selected_metrics": {
            "balanced": chosen["balanced"],
            "balanced_physical_only": chosen["balanced_physical_only"],
            "runtime": chosen["runtime"],
        },
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
        "checkpoint_selection_steps": 0,
    }
    write_json(output / "calibration.json", calibration)
    write_json(output / "selection_grid.json", selection)
    np.savez_compressed(
        output / "c07_c08_runtime_selection.npz",
        label=runtime["label"], family=runtime["family"],
        distance_m=runtime["distance_m"], scores=np.stack(runtime_scores),
        accepted=np.stack(runtime_votes),
    )
    print(json.dumps(calibration, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
