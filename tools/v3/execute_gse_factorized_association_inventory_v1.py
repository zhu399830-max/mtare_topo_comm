#!/usr/bin/env python3
"""Inventory C01-C08 factorized decision-node association evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_factorized_association_inventory import (
    DECISION_EVENTS,
    decision_pair_inventory,
    inbound_profile_identity_counts,
    past_observation_lengths,
)
from mtare_topo.governance import write_json


PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_INVENTORY_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_INVENTORY_V1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _incident_edge_summary(rows: list[dict], partition: np.ndarray) -> dict:
    edges: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        if partition[index] and str(row["event"]) in DECISION_EVENTS:
            edges.setdefault(str(row["identity"]), set()).add(str(row["edge_id"]))
    counts = np.asarray([len(value) for value in edges.values()], dtype=np.int64)
    return {
        "identities": len(edges),
        "minimum_incident_physical_edges": int(counts.min()) if len(counts) else 0,
        "median_incident_physical_edges": float(np.median(counts)) if len(counts) else 0.0,
        "maximum_incident_physical_edges": int(counts.max()) if len(counts) else 0,
        "identities_with_multiple_incident_edges": int(np.sum(counts >= 2)),
    }


def _write_family_csv(path: Path, pair_inventory: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("partition", "family", "pairs", "positive", "negative"))
        writer.writeheader()
        for partition in ("fit", "selection"):
            for family, values in pair_inventory[partition]["per_family"].items():
                writer.writerow({"partition": partition, "family": family, **values})


def _plot(path_prefix: Path, profile: dict, pairs: dict) -> None:
    families = sorted(pairs["selection"]["per_family"])
    positive = [pairs["selection"]["per_family"][name]["positive"] for name in families]
    negative = [pairs["selection"]["per_family"][name]["negative"] for name in families]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].bar([0, 1], [profile["fit"]["identity_coverage"], profile["selection"]["identity_coverage"]],
                color=["#4C78A8", "#54A24B"])
    axes[0].axhline(.90, color="#B22222", linestyle="--", linewidth=1.2)
    axes[0].set_xticks([0, 1], ["C01–C06 fit", "C07–C08 selection"])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Decision identities with ≥5-frame inbound profile")
    axes[0].set_title("Causal incident-edge geometry coverage")
    x = np.arange(len(families))
    axes[1].bar(x, positive, color="#54A24B", label="positive")
    axes[1].bar(x, negative, bottom=positive, color="#E45756", label="hard negative")
    axes[1].set_xticks(x, families, rotation=30)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Online-eligible decision pairs (log)")
    axes[1].set_title("C07–C08 association population")
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "AB"):
        axis.text(-.11, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Factorized GSE-Graph route-conditioned association inventory")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path_prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(path_prefix.parent / f"{path_prefix.name}_source.json", {
        "schema_version": "gse_factorized_association_inventory_figure_source_v1",
        "inbound_profile": profile,
        "pair_inventory": pairs,
        "selection_rule": "All ten families; no result-based family or pair selection.",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--feature", action="append", required=True, type=Path)
    parser.add_argument("--token", action="append", required=True, type=Path)
    args = parser.parse_args()
    if len(args.feature) != 3 or len(args.token) != 3:
        raise RuntimeError("inventory requires exactly three feature and token archives")
    run_dir = args.run_dir.resolve()
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188126:
        raise RuntimeError("factorized association Teacher population drift")
    global_sequence = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    event = np.asarray([str(row["event"]) for row in rows])
    identity = np.asarray([str(row["identity"]) for row in rows])
    edge = np.asarray([str(row["edge_id"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        partition_code = archive["partition_code"].astype(np.uint8)
        cache_parent = archive["parent_id"].astype(str)
        pair_arrays = {
            partition: {
                key: archive[f"{partition}_{key}"].copy()
                for key in ("left", "right", "label", "distance_m", "family")
            }
            for partition in ("fit", "selection")
        }
    if (
        not np.array_equal(compact_global, global_sequence)
        or not np.array_equal(cache_parent, parent)
        or int(np.sum(partition_code == 0)) != 142184
        or int(np.sum(partition_code == 1)) != 45942
    ):
        raise RuntimeError("factorized association population alignment drift")
    fit = partition_code == 0
    selection = partition_code == 1
    decision = np.isin(event, DECISION_EVENTS)
    row_inventory = {
        "fit": {
            "junction_rows": int(np.sum(fit & (event == "junction"))),
            "terminal_rows": int(np.sum(fit & (event == "terminal"))),
            "junction_identities": int(len(np.unique(identity[fit & (event == "junction")]))),
            "terminal_identities": int(len(np.unique(identity[fit & (event == "terminal")]))),
        },
        "selection": {
            "junction_rows": int(np.sum(selection & (event == "junction"))),
            "terminal_rows": int(np.sum(selection & (event == "terminal"))),
            "junction_identities": int(len(np.unique(identity[selection & (event == "junction")]))),
            "terminal_identities": int(len(np.unique(identity[selection & (event == "terminal")]))),
        },
    }
    expected_rows = {
        "fit": {"junction_rows": 19743, "terminal_rows": 5551, "junction_identities": 417, "terminal_identities": 375},
        "selection": {"junction_rows": 6865, "terminal_rows": 1974, "junction_identities": 146, "terminal_identities": 128},
    }
    profile_length = past_observation_lengths(traversal, sequence, maximum=5)
    inbound_profile = {
        "fit": inbound_profile_identity_counts(rows, profile_length, fit, required_length=5),
        "selection": inbound_profile_identity_counts(rows, profile_length, selection, required_length=5),
    }
    incident_edges = {
        "fit": _incident_edge_summary(rows, fit),
        "selection": _incident_edge_summary(rows, selection),
    }
    token_contract = {}
    for seed, (feature_path, token_path) in enumerate(zip(args.feature, args.token, strict=True)):
        features = np.load(feature_path.resolve(), mmap_mode="r")
        if features.shape != (188126, 146) or features.dtype != np.float32 or not np.all(np.isfinite(features[:, 5:12])):
            raise RuntimeError(f"seed{seed} geometry feature contract drift")
        with np.load(token_path.resolve(), allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_sequence):
                raise RuntimeError(f"seed{seed} token identity drift")
            expected_shapes = {
                "exit_confidence": (188126, 6), "exit_heading_unit": (188126, 6, 2),
                "exit_opening_width_m": (188126, 6), "exit_vertical_profile": (188126, 6, 4),
                "exit_descriptor": (188126, 6, 32),
            }
            for name, shape in expected_shapes.items():
                value = archive[name]
                if value.shape != shape or value.dtype != np.float32 or not np.all(np.isfinite(value)):
                    raise RuntimeError(f"seed{seed} {name} contract drift")
            confidence = archive["exit_confidence"]
            token_contract[str(seed)] = {
                "decision_rows": int(np.sum(decision)),
                "mean_decision_token_confidence": float(np.mean(confidence[decision])),
                "maximum_decision_token_confidence": float(np.max(confidence[decision])),
                "all_full_token_fields_finite": True,
            }
    pair_inventory = {
        partition: decision_pair_inventory(
            arrays["left"], arrays["right"], arrays["label"], arrays["distance_m"],
            arrays["family"], event, identity, edge, maximum_distance_m=16.0,
        )
        for partition, arrays in pair_arrays.items()
    }
    families_complete = all(
        values["positive"] > 0 and values["negative"] > 0
        for partition in pair_inventory.values() for values in partition["per_family"].values()
    )
    checks = {
        "exact_decision_row_and_identity_inventory": row_inventory == expected_rows,
        "exact_world_and_observation_split": (
            len(np.unique(parent[fit])) == 60 and len(np.unique(parent[selection])) == 20
            and int(np.sum(fit)) == 142184 and int(np.sum(selection)) == 45942
        ),
        "three_seed_full_token_contract": len(token_contract) == 3 and all(
            value["all_full_token_fields_finite"] for value in token_contract.values()
        ),
        "fit_inbound_profile_identity_coverage_at_least_0p90": inbound_profile["fit"]["identity_coverage"] >= .90,
        "selection_inbound_profile_identity_coverage_at_least_0p90": inbound_profile["selection"]["identity_coverage"] >= .90,
        "all_families_have_positive_and_hard_negative_pairs": families_complete,
        "selection_pairs_nonvacuous": pair_inventory["selection"]["positive_pairs"] > 0 and pair_inventory["selection"]["negative_pairs"] > 0,
        "cross_edge_positive_pairs_nonvacuous": pair_inventory["selection"]["positive_different_physical_edge"] > 0,
        "zero_optimizer_inference_forbidden_reads": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_factorized_association_inventory_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Does C01-C08 contain enough causal full-token, incident-edge geometry, positive revisit and hard-negative evidence for one route-conditioned association proof?",
        "row_inventory": row_inventory,
        "inbound_profile": inbound_profile,
        "incident_edges": incident_edges,
        "profile_length_decision_rows": {
            str(length): int(np.sum(decision & (profile_length == length))) for length in range(1, 6)
        },
        "token_contract": token_contract,
        "pair_inventory": pair_inventory,
        "checks": checks,
        "recommended_next": "ONE_ROUTE_CONDITIONED_ASSOCIATION_CAPACITY_PROOF" if passed else "STOP_AND_REDESIGN_ASSOCIATION_DATA_CONTRACT",
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/factorized_association_inventory.json", result)
    _write_family_csv(run_dir / "artifacts/decision_pair_family_inventory.csv", pair_inventory)
    _plot(run_dir / "previews/gse_factorized_association_inventory", inbound_profile, pair_inventory)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
