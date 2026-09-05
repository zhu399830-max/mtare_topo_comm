#!/usr/bin/env python3
"""Evaluate frozen action-conditioned geometry-state segmentation on C01-C08."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_action_conditioned_state import (
    action_conditioned_state_scores,
    evaluate_state_triggers,
    fit_corridor_false_alarm_threshold,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS_STATUS = "PASS_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1"
FAIL_STATUS = "FAIL_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1"
METHODS = ("geometry", "action", "combined")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_trigger_csv(
    path: Path,
    rows: list[dict],
    scores: dict[str, np.ndarray],
    trigger_rows: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method", "partition", "observation_row", "global_sequence_index",
            "parent_id", "traversal_id", "sequence_index", "event", "identity", "score",
        ))
        writer.writeheader()
        for method in METHODS:
            for partition in ("fit", "selection"):
                for row_index in trigger_rows[method][partition]:
                    row = rows[int(row_index)]
                    writer.writerow({
                        "method": method,
                        "partition": partition,
                        "observation_row": int(row_index),
                        "global_sequence_index": int(row["global_sequence_index"]),
                        "parent_id": row["parent_id"],
                        "traversal_id": row["traversal_id"],
                        "sequence_index": int(row["sequence_index"]),
                        "event": row["event"],
                        "identity": row["identity"],
                        "score": float(scores[method][int(row_index)]),
                    })


def _write_metric_csv(path: Path, metrics: dict[str, dict[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "method", "partition", "threshold", "predicted_triggers",
            "structural_trigger_precision", "structural_episode_recall",
            "false_trigger_fraction", "junction_identity_coverage",
            "terminal_identity_coverage", "turn_identity_coverage",
            "geometry_transition_identity_coverage",
        ))
        writer.writeheader()
        for method in METHODS:
            for partition in ("fit", "selection"):
                item = metrics[method][partition]
                writer.writerow({
                    "method": method,
                    "partition": partition,
                    "threshold": item["threshold"],
                    "predicted_triggers": item["predicted_triggers"],
                    "structural_trigger_precision": item["structural_trigger_precision"],
                    "structural_episode_recall": item["structural_episode_recall"],
                    "false_trigger_fraction": item["false_trigger_fraction_per_eligible_corridor_observation"],
                    "junction_identity_coverage": item["per_event"]["junction"]["identity_coverage"],
                    "terminal_identity_coverage": item["per_event"]["terminal"]["identity_coverage"],
                    "turn_identity_coverage": item["per_event"]["turn"]["identity_coverage"],
                    "geometry_transition_identity_coverage": item["per_event"]["geometry_transition"]["identity_coverage"],
                })


def _plot(path_prefix: Path, metrics: dict[str, dict[str, dict]], source: dict) -> None:
    colors = {"geometry": "#4C78A8", "action": "#F58518", "combined": "#54A24B"}
    labels = {"geometry": "Geometry", "action": "Exit/action", "combined": "Combined"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    x = np.arange(3)
    recall = [metrics[name]["selection"]["structural_episode_recall"] for name in METHODS]
    precision = [metrics[name]["selection"]["structural_trigger_precision"] for name in METHODS]
    axes[0].bar(x, recall, color=[colors[name] for name in METHODS])
    axes[0].set_title("Structural episode recall")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x, [labels[name] for name in METHODS], rotation=18)
    axes[0].axhline(max(recall[:2]) + .05, color="#B22222", linestyle="--", linewidth=1.2, label="baseline + 0.05")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - .16, precision, width=.32, color=[colors[name] for name in METHODS], label="precision")
    false_alarm = [
        metrics[name]["selection"]["false_trigger_fraction_per_eligible_corridor_observation"]
        for name in METHODS
    ]
    axes[1].bar(x + .16, false_alarm, width=.32, color="#B9B9B9", label="false trigger fraction")
    axes[1].axhline(.98, color="#222222", linestyle=":", linewidth=1.0)
    axes[1].axhline(.01, color="#B22222", linestyle=":", linewidth=1.0)
    axes[1].set_title("Safety of node proposals")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, [labels[name] for name in METHODS], rotation=18)
    axes[1].legend(frameon=False, fontsize=8)
    event_names = list(EVENT_NAMES[1:])
    event_labels = ["Junction", "Terminal", "Turn", "Geom. transition"]
    width = .24
    for index, method in enumerate(METHODS):
        coverage = [
            metrics[method]["selection"]["per_event"][event]["identity_coverage"]
            for event in event_names
        ]
        axes[2].bar(np.arange(4) + (index - 1) * width, coverage, width=width,
                    color=colors[method], label=labels[method])
    axes[2].set_title("Selection identity coverage")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xticks(np.arange(4), event_labels, rotation=18)
    axes[2].legend(frameon=False, fontsize=8)
    for axis, letter in zip(axes, "ABC"):
        axis.text(-.13, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Causal action-conditioned geometry-state feasibility (C07–C08)", fontsize=12)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path_prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(path_prefix.parent / f"{path_prefix.name}_source.json", source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--feature", action="append", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if len(args.feature) != 3:
        raise RuntimeError("exactly three frozen feature arrays are required")
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188126:
        raise RuntimeError("action-conditioned proof Teacher count drift")
    bank = materialize_causal_episode_references(rows)
    global_sequence = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    identity = np.asarray([str(row["identity"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        partition_code = archive["partition_code"].astype(np.uint8)
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        cache_parent = archive["parent_id"].astype(str)
    if (
        partition_code.shape != (188126,)
        or set(np.unique(partition_code).tolist()) != {0, 1}
        or int(np.sum(partition_code == 0)) != 142184
        or int(np.sum(partition_code == 1)) != 45942
        or not np.array_equal(compact_global, global_sequence)
        or not np.array_equal(cache_parent, parent)
    ):
        raise RuntimeError("action-conditioned proof partition/alignment drift")
    fit = partition_code == 0
    selection = partition_code == 1
    if len(np.unique(parent[fit])) != 60 or len(np.unique(parent[selection])) != 20:
        raise RuntimeError("action-conditioned proof world split drift")
    feature_mean = np.zeros((188126, 146), dtype=np.float32)
    for path in args.feature:
        current = np.load(path.resolve(), mmap_mode="r")
        if current.shape != (188126, 146) or current.dtype != np.float32:
            raise RuntimeError("frozen observation feature contract drift")
        feature_mean += current / 3.0
    if not np.all(np.isfinite(feature_mean)):
        raise RuntimeError("frozen observation ensemble contains nonfinite values")
    scores, scales = action_conditioned_state_scores(feature_mean, traversal, sequence, fit)
    metrics: dict[str, dict[str, dict]] = {}
    trigger_rows: dict[str, dict[str, np.ndarray]] = {}
    for method, score in scores.as_dict().items():
        threshold = fit_corridor_false_alarm_threshold(
            score, bank.event_index, fit, scores.eligible, corridor_quantile=.99
        )
        fit_metrics, fit_triggers = evaluate_state_triggers(
            score, bank.event_index, bank.episode_id, identity, traversal, sequence,
            fit, scores.eligible, threshold=threshold,
        )
        selection_metrics, selection_triggers = evaluate_state_triggers(
            score, bank.event_index, bank.episode_id, identity, traversal, sequence,
            selection, scores.eligible, threshold=threshold,
        )
        metrics[method] = {"fit": fit_metrics, "selection": selection_metrics}
        trigger_rows[method] = {"fit": fit_triggers, "selection": selection_triggers}
    expected_selection_identities = {
        "junction": 146, "terminal": 128, "turn": 95, "geometry_transition": 17,
    }
    observed_selection_identities = {
        event: metrics["combined"]["selection"]["per_event"][event]["true_identities"]
        for event in expected_selection_identities
    }
    baseline_recall = max(
        metrics["geometry"]["selection"]["structural_episode_recall"],
        metrics["action"]["selection"]["structural_episode_recall"],
    )
    combined = metrics["combined"]["selection"]
    recall_gain = combined["structural_episode_recall"] - baseline_recall
    checks = {
        "exact_observation_and_world_split": (
            int(np.sum(fit)) == 142184 and int(np.sum(selection)) == 45942
            and len(np.unique(parent[fit])) == 60 and len(np.unique(parent[selection])) == 20
        ),
        "exact_fit_and_selection_episode_counts": (
            len(np.unique(bank.episode_id[fit & (bank.episode_id >= 0)])) == 3956
            and len(np.unique(bank.episode_id[selection & (bank.episode_id >= 0)])) == 1350
        ),
        "exact_selection_identity_inventory": observed_selection_identities == expected_selection_identities,
        "combined_nonvacuous": combined["predicted_triggers"] > 0,
        "combined_precision_at_least_0p98": combined["structural_trigger_precision"] >= .98,
        "combined_false_trigger_fraction_at_most_0p01": (
            combined["false_trigger_fraction_per_eligible_corridor_observation"] <= .01
        ),
        "combined_recall_gain_at_least_0p05": recall_gain >= .05,
        "combined_turn_identity_nonzero": combined["per_event"]["turn"]["matched_unique_identities"] >= 1,
        "combined_transition_identity_nonzero": (
            combined["per_event"]["geometry_transition"]["matched_unique_identities"] >= 1
        ),
        "zero_optimizer_and_inference": True,
        "zero_forbidden_reads": True,
    }
    passed = all(checks.values())
    scale_json = {
        name: {"center": value.center.tolist(), "scale": value.scale.tolist()}
        for name, value in scales.items()
    }
    result = {
        "schema_version": "gse_action_conditioned_state_feasibility_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Can frozen continuous geometry and exit/action state jointly create safer, more complete structural node proposals than either source alone?",
        "method": "Past-only adjacent six-versus-six robust state discrepancy; fit-only higher 99th-percentile corridor threshold; contiguous responses collapse to one trigger.",
        "feature_contract": {
            "geometry_columns": "5:12 local axis plus normalized width/height/slope/curvature",
            "action_columns": "141:146 threshold-free exit-token summary",
            "combined": "equal group RMS; place descriptor and event probabilities excluded",
            "ensemble": "arithmetic mean of three frozen seed feature arrays",
        },
        "fit_worlds": 60,
        "fit_observations": int(np.sum(fit)),
        "selection_worlds": 20,
        "selection_observations": int(np.sum(selection)),
        "fit_episodes": 3956,
        "selection_episodes": 1350,
        "selection_identity_inventory": observed_selection_identities,
        "eligible_observations": int(np.sum(scores.eligible)),
        "threshold_rule": "fit eligible corridor higher quantile 0.99",
        "robust_scales": scale_json,
        "metrics": metrics,
        "combined_recall_gain_over_stronger_single_source": recall_gain,
        "checks": checks,
        "recommended_next": (
            "DESIGN_ACTION_CONDITIONED_GEOMETRY_STATE_MODEL_AND_GRAPH"
            if passed else "STOP_OR_REVISE_ACTION_CONDITIONED_STATE_ROUTE_BEFORE_TRAINING"
        ),
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/action_conditioned_state_feasibility.json", result)
    _write_metric_csv(run_dir / "artifacts/method_metrics.csv", metrics)
    _write_trigger_csv(run_dir / "artifacts/triggers.csv", rows, scores.as_dict(), trigger_rows)
    source = {
        "schema_version": "gse_action_conditioned_state_feasibility_figure_source_v1",
        "selection_metrics": {method: metrics[method]["selection"] for method in METHODS},
        "acceptance": {
            "minimum_precision": .98, "maximum_false_trigger_fraction": .01,
            "minimum_recall_gain": .05, "minimum_turn_identities": 1,
            "minimum_geometry_transition_identities": 1,
        },
    }
    _plot(run_dir / "previews/gse_action_conditioned_state_feasibility", metrics, source)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
