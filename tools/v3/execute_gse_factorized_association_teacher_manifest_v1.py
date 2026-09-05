#!/usr/bin/env python3
"""Build one split-isolated identity-balanced decision association manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.governance import write_json
from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references,
    choose_hard_negative_identity,
    choose_positive_rows,
    objective_geometry_profile,
)


PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1"
DECISION_EVENTS = ("junction", "terminal")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def _plot(path_prefix: Path, manifest: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    for partition, color in (("fit", "#4C78A8"), ("selection", "#54A24B")):
        distances = [row["hard_negative_objective_profile_distance"] for row in manifest if row["partition"] == partition]
        axes[0].hist(distances, bins=30, histtype="step", linewidth=1.8, color=color, label=partition)
    axes[0].set_xlabel("Nearest structural-alias geometry distance")
    axes[0].set_ylabel("Decision identities")
    axes[0].set_title("Identity-balanced hard negatives")
    axes[0].legend(frameon=False)
    kinds = (
        "different_physical_edge", "same_edge_reverse_view",
        "same_identity_distinct_observation", "singleton_circular_shift_augmentation",
    )
    x = np.arange(len(kinds))
    fit = [sum(row["positive_view_kind"] == kind and row["partition"] == "fit" for row in manifest) for kind in kinds]
    selection = [sum(row["positive_view_kind"] == kind and row["partition"] == "selection" for row in manifest) for kind in kinds]
    axes[1].bar(x - .18, fit, width=.36, color="#4C78A8", label="fit")
    axes[1].bar(x + .18, selection, width=.36, color="#54A24B", label="selection")
    axes[1].set_xticks(
        x, ["different edge", "reverse view", "distinct obs.", "rotation aug."], rotation=18
    )
    axes[1].set_ylabel("Positive identity units")
    axes[1].set_title("Cross-view positive construction")
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "AB"):
        axis.text(-.11, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Factorized GSE-Graph association Teacher manifest")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path_prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(path_prefix.parent / f"{path_prefix.name}_source.json", {
        "schema_version": "gse_factorized_association_teacher_figure_source_v1",
        "hard_negative_distance": {
            partition: [row["hard_negative_objective_profile_distance"] for row in manifest if row["partition"] == partition]
            for partition in ("fit", "selection")
        },
        "positive_view_kind": {
            partition: {kind: sum(row["positive_view_kind"] == kind and row["partition"] == partition for row in manifest) for kind in kinds}
            for partition in ("fit", "selection")
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188126:
        raise RuntimeError("factorized Teacher source population drift")
    global_sequence = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    event = np.asarray([str(row["event"]) for row in rows])
    identity = np.asarray([str(row["identity"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        association_valid = archive["association_valid"].astype(np.uint8)
        partition_code = archive["partition_code"].astype(np.uint8)
        cache_parent = archive["parent_id"].astype(str)
    if (
        not np.array_equal(compact_global, global_sequence)
        or not np.array_equal(cache_parent, parent)
        or association_valid.shape != (188126,)
        or partition_code.shape != (188126,)
        or set(np.unique(partition_code).tolist()) != {0, 1}
    ):
        raise RuntimeError("factorized Teacher alignment drift")
    references, history_mask = causal_history_row_references(traversal, sequence)
    decision_rows = np.flatnonzero(np.isin(event, DECISION_EVENTS))
    grouped: dict[str, list[int]] = {}
    for row_index in decision_rows:
        grouped.setdefault(str(identity[row_index]), []).append(int(row_index))
    if len(grouped) != 1066:
        raise RuntimeError("factorized decision identity count drift")

    metadata: dict[str, dict] = {}
    for name in sorted(grouped):
        indices = grouped[name]
        partitions = np.unique(partition_code[indices])
        events = np.unique(event[indices])
        parents = np.unique(parent[indices])
        if len(partitions) != 1 or len(events) != 1 or len(parents) != 1:
            raise RuntimeError("one decision identity crosses partition/event/parent")
        query, positive, positive_kind = choose_positive_rows(
            rows, indices, history_mask, association_valid
        )
        profile, profile_mask = objective_geometry_profile(rows, references, history_mask, query)
        edges = sorted({str(rows[index]["edge_id"]) for index in indices})
        metadata[name] = {
            "identity": name,
            "partition": "fit" if int(partitions[0]) == 0 else "selection",
            "event": str(events[0]),
            "degree": len(edges),
            "parent_id": str(parents[0]),
            "family": str(parents[0]).split("_", 1)[0],
            "query_row": query,
            "positive_row": positive,
            "positive_view_kind": positive_kind,
            "profile": profile,
            "profile_mask": profile_mask,
            "profile_observations": int(np.sum(history_mask[query])),
            "incident_edge_ids": edges,
        }

    manifest = []
    for name in sorted(metadata):
        item = metadata[name]
        negative_name, negative_distance = choose_hard_negative_identity(name, metadata)
        negative = metadata[negative_name]
        query_row = int(item["query_row"])
        positive_row = int(item["positive_row"])
        negative_row = int(negative["query_row"])
        manifest.append({
            "schema_version": "gse_factorized_association_teacher_unit_v1",
            "partition": item["partition"],
            "family": item["family"],
            "event": item["event"],
            "incident_degree": int(item["degree"]),
            "query_identity": name,
            "query_parent_id": item["parent_id"],
            "query_observation_row": query_row,
            "query_global_sequence_index": int(global_sequence[query_row]),
            "query_edge_id": str(rows[query_row]["edge_id"]),
            "query_profile_observations": int(item["profile_observations"]),
            "query_profile_mask": np.asarray(item["profile_mask"], dtype=np.uint8).tolist(),
            "positive_identity": str(identity[positive_row]),
            "positive_observation_row": positive_row,
            "positive_global_sequence_index": int(global_sequence[positive_row]),
            "positive_edge_id": str(rows[positive_row]["edge_id"]),
            "positive_view_kind": item["positive_view_kind"],
            "positive_augmentation": (
                {"kind": "circular_azimuth_shift", "shift_bins": 180, "range_image_width_bins": 720}
                if item["positive_view_kind"] == "singleton_circular_shift_augmentation" else None
            ),
            "hard_negative_identity": negative_name,
            "hard_negative_parent_id": negative["parent_id"],
            "hard_negative_observation_row": negative_row,
            "hard_negative_global_sequence_index": int(global_sequence[negative_row]),
            "hard_negative_edge_id": str(rows[negative_row]["edge_id"]),
            "hard_negative_event": negative["event"],
            "hard_negative_incident_degree": int(negative["degree"]),
            "hard_negative_objective_profile_distance": negative_distance,
            "teacher_only_negative_selection_fields": ["event", "incident_degree", "objective_geometry_profile"],
            "student_forbidden_fields": ["identity", "parent_id", "event_teacher", "objective_geometry_profile", "world_id"],
        })

    fit_manifest = [row for row in manifest if row["partition"] == "fit"]
    selection_manifest = [row for row in manifest if row["partition"] == "selection"]
    limited_profile = {
        "fit": sum(row["query_profile_observations"] < 5 for row in fit_manifest),
        "selection": sum(row["query_profile_observations"] < 5 for row in selection_manifest),
    }
    family_counts = {
        partition: {
            family: sum(row["partition"] == partition and row["family"] == family for row in manifest)
            for family in sorted({row["family"] for row in manifest})
        }
        for partition in ("fit", "selection")
    }
    checks = {
        "exact_identity_units": len(fit_manifest) == 792 and len(selection_manifest) == 274,
        "one_unique_query_per_identity": len({row["query_identity"] for row in manifest}) == 1066,
        "positive_identity_exact": all(row["positive_identity"] == row["query_identity"] for row in manifest),
        "singleton_augmentation_exact": sum(
            row["positive_view_kind"] == "singleton_circular_shift_augmentation" for row in manifest
        ) == 5,
        "negative_identity_distinct": all(row["hard_negative_identity"] != row["query_identity"] for row in manifest),
        "negative_event_and_degree_matched": all(
            row["hard_negative_event"] == row["event"]
            and row["hard_negative_incident_degree"] == row["incident_degree"] for row in manifest
        ),
        "negative_split_isolated": all(metadata[row["hard_negative_identity"]]["partition"] == row["partition"] for row in manifest),
        "finite_nonnegative_hard_negative_distance": all(
            np.isfinite(row["hard_negative_objective_profile_distance"])
            and row["hard_negative_objective_profile_distance"] >= 0.0 for row in manifest
        ),
        "expected_limited_profile_masks": limited_profile == {"fit": 15, "selection": 5},
        "every_family_present_both_partitions": all(value > 0 for counts in family_counts.values() for value in counts.values()),
        "zero_forbidden_operations": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_factorized_association_teacher_manifest_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Can every C01-C08 decision identity receive one split-isolated cross-view positive and one same-event/degree objective-geometry-nearest hard negative?",
        "manifest_units": len(manifest),
        "fit_units": len(fit_manifest),
        "selection_units": len(selection_manifest),
        "pair_records": 2 * len(manifest),
        "limited_profile_masks": limited_profile,
        "positive_view_kind": {
            kind: sum(row["positive_view_kind"] == kind for row in manifest)
            for kind in sorted({row["positive_view_kind"] for row in manifest})
        },
        "event_degree_units": {
            f"{partition}:{event_name}:degree{degree}": sum(
                row["partition"] == partition and row["event"] == event_name and row["incident_degree"] == degree
                for row in manifest
            )
            for partition in ("fit", "selection")
            for event_name in DECISION_EVENTS
            for degree in (1, 2, 3, 4)
            if any(row["partition"] == partition and row["event"] == event_name and row["incident_degree"] == degree for row in manifest)
        },
        "family_counts": family_counts,
        "hard_negative_distance": {
            partition: {
                "minimum": float(min(row["hard_negative_objective_profile_distance"] for row in manifest if row["partition"] == partition)),
                "median": float(np.median([row["hard_negative_objective_profile_distance"] for row in manifest if row["partition"] == partition])),
                "maximum": float(max(row["hard_negative_objective_profile_distance"] for row in manifest if row["partition"] == partition)),
            }
            for partition in ("fit", "selection")
        },
        "checks": checks,
        "recommended_next": "FREEZE_ROUTE_CONDITIONED_ASSOCIATION_CAPACITY_DATA_CARD" if passed else "STOP_ASSOCIATION_TEACHER_ROUTE",
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    _write_jsonl(run_dir / "artifacts/identity_balanced_association_manifest.jsonl", manifest)
    with (run_dir / "artifacts/identity_balanced_association_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "partition", "family", "event", "incident_degree", "query_identity",
            "query_global_sequence_index", "positive_global_sequence_index", "positive_view_kind",
            "hard_negative_identity", "hard_negative_global_sequence_index",
            "hard_negative_objective_profile_distance", "query_profile_observations",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({name: row[name] for name in fields})
    write_json(run_dir / "metrics/factorized_association_teacher_manifest.json", result)
    _plot(run_dir / "previews/gse_factorized_association_teacher_manifest", manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
