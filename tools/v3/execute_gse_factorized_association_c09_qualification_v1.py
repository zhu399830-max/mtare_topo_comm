#!/usr/bin/env python3
"""Qualify frozen unified-slope Factorized association on unseen C09."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from mtare_topo.evaluation.gse_factorized_qualification import (
    fixed_threshold_metrics,
    runtime_candidate_pairs,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_factorized_association import (
    FactorizedAssociationVerifier,
    factorized_pair_features,
    learned_geometry_profiles,
)
from mtare_topo.representation.gse_open_set_association import observation_features
from mtare_topo.representation.gse_unified_observation import replace_normalized_slope
from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references_unordered,
    choose_hard_negative_identity,
    choose_positive_rows,
    objective_geometry_profile,
)


PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
EXPECTED_PARENTS = frozenset(f"S{family:02d}_" for family in range(1, 11))
TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)


def _read_c09_teacher(path: Path, global_index: np.ndarray) -> list[dict]:
    by_index = {}
    total = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if str(row["parent_id"]).endswith("_C09"):
                by_index[int(row["global_sequence_index"])] = row
    if total != 212_588 or len(by_index) != 24_462 or set(by_index) != set(global_index.tolist()):
        raise RuntimeError("C09 Teacher population or identity drift")
    return [by_index[int(value)] for value in global_index]


def _metadata(
    dataset_run: Path, rows: list[dict], global_index: np.ndarray,
    parent: np.ndarray, association_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    manifest = {}
    with (dataset_run / "artifacts/sequence_manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["parent_id"]).endswith("_C09"):
                manifest[int(row["global_sequence_index"])] = row
    if len(manifest) != len(rows) or set(manifest) != set(global_index.tolist()):
        raise RuntimeError("C09 sequence manifest identity drift")
    xyz = np.empty((len(rows), 3), dtype=np.float64)
    world_sequence_row = np.empty(len(rows), dtype=np.int64)
    for current_parent in sorted(set(parent.tolist())):
        indices = np.flatnonzero(parent == current_parent)
        shard = zarr.open_group(
            str(dataset_run / "artifacts/dataset/validation" / f"{current_parent}.zarr"), mode="r"
        )
        sequence_rows = np.asarray(
            [int(manifest[int(global_index[index])]["world_sequence_row"]) for index in indices],
            dtype=np.int64,
        )
        anchor_rows = np.asarray(
            [int(manifest[int(global_index[index])]["world_anchor_frame_row"]) for index in indices],
            dtype=np.int64,
        )
        xyz[indices] = np.asarray(shard["sensor_xyz_m"].oindex[anchor_rows], dtype=np.float64)
        world_sequence_row[indices] = sequence_rows
        shard_valid = np.asarray(shard["association_valid_mask"].oindex[sequence_rows], dtype=np.uint8)
        if not np.array_equal(shard_valid.astype(bool), association_valid[indices]):
            raise RuntimeError(f"C09 association-valid mask drift: {current_parent}")
    return xyz, world_sequence_row


def _balanced_alias_pairs(
    rows: list[dict], association_valid: np.ndarray,
    references: np.ndarray, history_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    event = np.asarray([str(row["event"]) for row in rows])
    identity = np.asarray([str(row["identity"]) for row in rows])
    decision = np.flatnonzero(np.isin(event, ("junction", "terminal")))
    grouped: dict[str, list[int]] = {}
    for index in decision:
        grouped.setdefault(str(identity[index]), []).append(int(index))
    if len(grouped) != 130:
        raise RuntimeError("C09 decision identity count drift")
    metadata: dict[str, dict] = {}
    for name in sorted(grouped):
        indices = grouped[name]
        events = np.unique(event[indices])
        parents = np.unique([str(rows[index]["parent_id"]) for index in indices])
        if len(events) != 1 or len(parents) != 1:
            raise RuntimeError("C09 decision identity crosses event or parent")
        query, positive, kind = choose_positive_rows(rows, indices, history_mask, association_valid)
        profile, profile_mask = objective_geometry_profile(rows, references, history_mask, query)
        edges = sorted({str(rows[index]["edge_id"]) for index in indices})
        metadata[name] = {
            "partition": "c09", "event": str(events[0]), "degree": len(edges),
            "parent_id": str(parents[0]), "family": str(parents[0]).split("_", 1)[0],
            "query": query, "positive": positive, "kind": kind,
            "profile": profile, "profile_mask": profile_mask,
            "profile_observations": int(np.sum(history_mask[query])),
        }
    manifest = []
    left: list[int] = []
    right: list[int] = []
    label: list[int] = []
    family: list[str] = []
    physical: list[bool] = []
    for name in sorted(metadata):
        item = metadata[name]
        negative_name, distance = choose_hard_negative_identity(name, metadata)
        negative = metadata[negative_name]
        left.extend((item["query"], item["query"]))
        right.extend((item["positive"], negative["query"]))
        label.extend((1, 0)); family.extend((item["family"], item["family"]))
        physical.extend((item["kind"] != "singleton_circular_shift_augmentation", True))
        manifest.append({
            "query_identity": name, "query_observation_row": item["query"],
            "positive_observation_row": item["positive"], "positive_view_kind": item["kind"],
            "hard_negative_identity": negative_name,
            "hard_negative_observation_row": negative["query"],
            "hard_negative_objective_profile_distance": distance,
            "event": item["event"], "incident_degree": item["degree"],
            "family": item["family"], "parent_id": item["parent_id"],
            "query_profile_observations": item["profile_observations"],
        })
    kinds = {kind: sum(row["positive_view_kind"] == kind for row in manifest) for kind in (
        "different_physical_edge", "same_edge_reverse_view",
        "same_identity_distinct_observation", "singleton_circular_shift_augmentation",
    )}
    if kinds != {
        "different_physical_edge": 71, "same_edge_reverse_view": 55,
        "same_identity_distinct_observation": 3,
        "singleton_circular_shift_augmentation": 1,
    }:
        raise RuntimeError(f"C09 balanced positive construction drift: {kinds}")
    return {
        "left": np.asarray(left, dtype=np.int64), "right": np.asarray(right, dtype=np.int64),
        "label": np.asarray(label, dtype=np.uint8), "family": np.asarray(family),
        "physical_positive": np.asarray(physical, dtype=np.bool_),
    }, manifest


def _score(
    capacity_run: Path, seed: int, raw_features: np.ndarray,
    labels: np.ndarray, families: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    base = capacity_run / f"artifacts/models/seed{seed}/full_route_conditioned"
    with np.load(base / "normalization.npz", allow_pickle=False) as archive:
        mean = archive["mean"].astype(np.float32)
        std = archive["std"].astype(np.float32)
    normalized = ((raw_features - mean) / std).astype(np.float32)
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "gse_factorized_association_checkpoint_v1"
        or checkpoint.get("seed") != seed or checkpoint.get("include_route_geometry") is not True
    ):
        raise RuntimeError(f"C09 frozen association checkpoint drift: seed {seed}")
    model = FactorizedAssociationVerifier(include_route_geometry=True)
    model.load_state_dict(checkpoint["model"]); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(normalized))).numpy().astype(np.float64)
    summary = load_json(capacity_run / f"artifacts/models/seed{seed}/summary.json")
    threshold = summary["full_route_conditioned"]["selection"]["threshold_selection"]
    if threshold is None:
        raise RuntimeError(f"C09 frozen threshold missing: seed {seed}")
    metrics = fixed_threshold_metrics(scores, labels, families, float(threshold["threshold"]))
    return scores, metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _plot(run_dir: Path, seed_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    x = np.arange(3); width = .32
    balanced_recall = [row["balanced"]["recall"] for row in seed_rows]
    runtime_recall = [row["runtime"]["recall"] for row in seed_rows]
    axes[0].bar(x - width / 2, balanced_recall, width, label="balanced aliases", color="#4C78A8")
    axes[0].bar(x + width / 2, runtime_recall, width, label="runtime candidates", color="#54A24B")
    axes[0].axhline(.25, color="#D62728", linestyle="--", linewidth=1.2)
    axes[0].set_xticks(x, [f"seed {seed}" for seed in range(3)])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Recall at frozen threshold")
    axes[0].set_title("Unseen C09 association recall"); axes[0].legend(frameon=False)
    balanced_precision = [row["balanced"]["precision"] for row in seed_rows]
    runtime_precision = [row["runtime"]["precision"] for row in seed_rows]
    axes[1].bar(x - width / 2, balanced_precision, width, label="balanced aliases", color="#4C78A8")
    axes[1].bar(x + width / 2, runtime_precision, width, label="runtime candidates", color="#54A24B")
    axes[1].axhline(.98, color="#D62728", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x, [f"seed {seed}" for seed in range(3)])
    axes[1].set_ylim(.85, 1.005); axes[1].set_ylabel("Precision at frozen threshold")
    axes[1].set_title("Unseen C09 false-merge safety"); axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "AB"):
        axis.text(-.11, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7); axis.set_axisbelow(True)
    fig.suptitle("Factorized GSE-Graph frozen C09 qualification")
    prefix = run_dir / "previews/gse_factorized_association_c09_qualification"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(prefix.parent / f"{prefix.name}_source.json", {
        "schema_version": "gse_factorized_association_c09_figure_source_v1",
        "seeds": seed_rows, "precision_gate": .98, "false_accept_gate": .01,
        "recall_gate": .25, "selection_effect": "NONE_FROZEN_C01_C08_THRESHOLDS",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--risk-run", required=True, type=Path)
    parser.add_argument("--capacity-run", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); dataset = args.dataset_run.resolve()
    training = args.training_run.resolve(); risk = args.risk_run.resolve(); capacity = args.capacity_run.resolve()
    seed_rows = []
    reference_global = reference_parent = reference_valid = None
    rows = None; references = history_mask = xyz = world_row = None
    balanced = runtime = None
    for seed in range(3):
        with np.load(training / f"artifacts/models/seed{seed}/validation_outputs.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        global_index = arrays["global_sequence_index"].astype(np.int64)
        parent = arrays["parent_id"].astype(str)
        association_valid = arrays["target_association_valid_mask"].astype(bool)
        if (
            global_index.shape != (24_462,) or len(np.unique(global_index)) != 24_462
            or len(set(parent.tolist())) != 10
        ):
            raise RuntimeError(f"C09 frozen validation population drift: seed {seed}")
        if seed == 0:
            reference_global, reference_parent, reference_valid = global_index, parent, association_valid
            rows = _read_c09_teacher(args.teacher.resolve(), global_index)
            references, history_mask = causal_history_row_references_unordered(
                [str(row["traversal_id"]) for row in rows],
                np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64),
            )
            xyz, world_row = _metadata(dataset, rows, global_index, parent, association_valid)
            balanced, balanced_manifest = _balanced_alias_pairs(rows, association_valid, references, history_mask)
            runtime = runtime_candidate_pairs(rows, association_valid, xyz, world_row, maximum_distance_m=16.0)
            if (
                len(balanced["label"]) != 260 or int(np.sum(balanced["label"])) != 130
                or int(runtime["decision_queries"]) != 4085
                or int(runtime["queries_with_candidate"]) != 3968
                or int(runtime["queries_with_positive"]) != 3955
                or len(runtime["label"]) != 4201 or int(np.sum(runtime["label"])) != 3955
            ):
                raise RuntimeError("C09 balanced or runtime pair count drift")
            _write_jsonl(run_dir / "artifacts/c09_balanced_alias_manifest.jsonl", balanced_manifest)
            np.savez_compressed(run_dir / "artifacts/c09_balanced_alias_pairs.npz", **balanced)
            np.savez_compressed(run_dir / "artifacts/c09_runtime_candidate_pairs.npz", **runtime)
        elif (
            not np.array_equal(global_index, reference_global)
            or not np.array_equal(parent, reference_parent)
            or not np.array_equal(association_valid, reference_valid)
        ):
            raise RuntimeError(f"C09 frozen seed alignment drift: seed {seed}")
        output_fields = {
            key: arrays[key].astype(np.float32) for key in (
                "event_logits", "local_axis", "width_m", "height_m", "slope_deg",
                "curvature_per_m", "place_descriptor", "uncertainty", *TOKEN_KEYS,
            )
        }
        observation = observation_features(output_fields)
        with np.load(risk / f"artifacts/risk_calibrated_slope_c09/seed{seed}_outputs.npz", allow_pickle=False) as slope:
            observation, audit = replace_normalized_slope(
                observation, global_index, slope["global_sequence_index"], slope["corrected_slope_deg"]
            )
        if audit.changed_columns != (10,) or not audit.unchanged_columns_byte_exact:
            raise RuntimeError(f"C09 unified observation integration drift: seed {seed}")
        np.save(run_dir / f"artifacts/seed{seed}_c09_unified_observation_features.npy", observation, allow_pickle=False)
        token = {key: output_fields[key] for key in TOKEN_KEYS}
        profiles = learned_geometry_profiles(observation, references, history_mask)
        balanced_raw = factorized_pair_features(
            observation, token, profiles, balanced["left"], balanced["right"], include_route_geometry=True
        )
        runtime_raw = factorized_pair_features(
            observation, token, profiles, runtime["left"], runtime["right"], include_route_geometry=True
        )
        balanced_scores, balanced_metrics = _score(
            capacity, seed, balanced_raw, balanced["label"], balanced["family"]
        )
        runtime_scores, runtime_metrics = _score(
            capacity, seed, runtime_raw, runtime["label"], runtime["family"]
        )
        keep = balanced["physical_positive"]
        physical = fixed_threshold_metrics(
            balanced_scores[keep], balanced["label"][keep], balanced["family"][keep],
            float(balanced_metrics["threshold"]),
        )
        np.savez_compressed(
            run_dir / f"artifacts/seed{seed}_c09_qualification_scores.npz",
            balanced_score=balanced_scores, balanced_label=balanced["label"],
            runtime_score=runtime_scores, runtime_label=runtime["label"],
        )
        balanced_safe = bool(
            balanced_metrics["accepted"] > 0 and balanced_metrics["precision"] >= .98
            and balanced_metrics["false_accept_rate"] <= .01 and balanced_metrics["recall"] >= .25
            and all(value["accepted"] > 0 and value["precision"] >= .95 and value["recall"] >= .10 for value in balanced_metrics["per_family"].values())
        )
        physical_safe = bool(
            physical["accepted"] > 0 and physical["precision"] >= .98
            and physical["false_accept_rate"] <= .01 and physical["recall"] >= .25
            and all(value["accepted"] > 0 and value["precision"] >= .95 and value["recall"] >= .10 for value in physical["per_family"].values())
        )
        runtime_safe = bool(
            runtime_metrics["accepted"] > 0 and runtime_metrics["precision"] >= .98
            and runtime_metrics["false_accept_rate"] <= .01 and runtime_metrics["recall"] >= .25
        )
        seed_rows.append({
            "seed": seed, "threshold": balanced_metrics["threshold"],
            "balanced": balanced_metrics, "balanced_physical_only": physical,
            "runtime": runtime_metrics, "interface_audit": audit.to_dict(),
            "checks": {"balanced_safe": balanced_safe, "physical_only_safe": physical_safe, "runtime_aggregate_safe": runtime_safe},
        })
    checks = {
        "exact_c09_population": len(rows) == 24_462 and len(set(reference_parent.tolist())) == 10,
        "exact_balanced_population": len(balanced["label"]) == 260 and int(np.sum(balanced["label"])) == 130,
        "exact_runtime_population": len(runtime["label"]) == 4201 and int(np.sum(runtime["label"])) == 3955,
        "all_three_balanced_safe": all(row["checks"]["balanced_safe"] for row in seed_rows),
        "all_three_physical_only_safe": all(row["checks"]["physical_only_safe"] for row in seed_rows),
        "all_three_runtime_aggregate_safe": all(row["checks"]["runtime_aggregate_safe"] for row in seed_rows),
        "runtime_zero_negative_families_explicit": all(
            row["runtime"]["per_family"][family]["negative_support"] == 0
            and row["runtime"]["per_family"][family]["precision_identifiable"] is False
            for row in seed_rows for family in ("S02", "S03")
        ),
        "zero_selection_or_forbidden_operations": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_factorized_association_c09_qualification_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Do frozen unified-slope route-conditioned association models retain safety on unseen C09 balanced aliases and strictly-past runtime candidates?",
        "validation_worlds": 10, "validation_sequences": 24_462,
        "balanced_identity_units": 130, "balanced_pairs": 260,
        "runtime_decision_queries": 4085, "runtime_queries_with_candidate": 3968,
        "runtime_pairs": 4201, "runtime_positive_pairs": 3955, "runtime_negative_pairs": 246,
        "seeds": seed_rows, "checks": checks,
        "checkpoint_selection_steps": 0, "threshold_selection_steps": 0,
        "optimizer_steps": 0, "model_updates": 0, "c10_worlds_read": 0,
        "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        "recommended_next": "FREEZE_C09_OFFLINE_FACTORIZED_GRAPH_PROOF" if passed else "STOP_FACTORIZED_ASSOCIATION_CLAIM",
    }
    write_json(run_dir / "metrics/factorized_association_c09_qualification.json", result)
    _plot(run_dir, seed_rows)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
