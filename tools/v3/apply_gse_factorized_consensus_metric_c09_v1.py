#!/usr/bin/env python3
"""Apply a sealed C01-C08 consensus/metric calibration to archived C09 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_factorized_qualification import fixed_threshold_metrics
from mtare_topo.governance import load_json, write_json


PASS_STATUS = "PASS_GSE_FACTORIZED_CONSENSUS_METRIC_C09_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_CONSENSUS_METRIC_C09_V1"


def _balanced_gate(metrics: dict) -> bool:
    return bool(
        metrics["accepted"] > 0
        and metrics["precision"] >= .98
        and metrics["false_accept_rate"] <= .01
        and metrics["recall"] >= .25
        and all(
            value["accepted"] > 0
            and value["precision"] >= .95
            and value["recall"] >= .10
            for value in metrics["per_family"].values()
        )
    )


def _runtime_gate(metrics: dict) -> bool:
    return bool(
        metrics["accepted"] > 0
        and metrics["precision"] >= .98
        and metrics["false_accept_rate"] <= .01
        and metrics["recall"] >= .25
    )


def _plot(output: Path, result: dict) -> None:
    selection = result["selection_metrics"]
    validation = result["c09_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    domains = ("Balanced", "Runtime")
    x = np.arange(2)
    width = .34
    for offset, (name, data, color) in enumerate((
        ("C07–C08 selection", selection, "#72B7B2"),
        ("C09 validation", validation, "#4C78A8"),
    )):
        precision = [data["balanced"]["precision"], data["runtime"]["precision"]]
        axes[0].bar(x + (offset - .5) * width, precision, width, label=name, color=color)
        recall = [data["balanced"]["recall"], data["runtime"]["recall"]]
        axes[1].bar(x + (offset - .5) * width, recall, width, label=name, color=color)
    axes[0].axhline(.98, color="#D62728", linestyle="--", linewidth=1.2, label="precision gate")
    axes[1].axhline(.25, color="#D62728", linestyle="--", linewidth=1.2, label="recall gate")
    for axis, title, ylabel, letter in zip(
        axes, ("False-merge safety", "Retained true associations"),
        ("Precision", "Recall"), "AB",
    ):
        axis.set_xticks(x, domains)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#E6E6E6", linewidth=.7)
        axis.set_axisbelow(True)
        axis.text(-.11, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        f"Consensus association: {result['votes_required']}-of-3, "
        f"runtime distance ≤ {result['distance_cap_m']:.1f} m"
    )
    prefix = output / "gse_factorized_consensus_metric_c09"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(prefix.parent / f"{prefix.name}_source.json", {
        "schema_version": "gse_factorized_consensus_metric_c09_figure_source_v1",
        "votes_required": result["votes_required"],
        "distance_cap_m": result["distance_cap_m"],
        "selection_metrics": selection,
        "c09_metrics": validation,
        "gates": {"precision": .98, "false_accept_rate": .01, "recall": .25},
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--c09-failed-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    calibration = load_json(args.calibration.resolve())
    c09 = args.c09_failed_run.resolve()
    source = load_json(c09 / "metrics/factorized_association_c09_qualification.json")
    state = load_json(c09 / "RUN_STATE.json")
    if (
        calibration.get("schema_version") != "gse_factorized_consensus_metric_calibration_v1"
        or calibration.get("c09_worlds_read") != 0
        or state.get("state") != "FAILED"
        or source.get("overall_status") != "FAIL_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1"
        or source.get("balanced_pairs") != 260
        or source.get("runtime_pairs") != 4_201
    ):
        raise RuntimeError("calibration or archived C09 source contract drift")
    votes_required = int(calibration["votes_required"])
    distance_cap_m = float(calibration["distance_cap_m"])
    thresholds = np.asarray(calibration["frozen_seed_thresholds"], dtype=np.float64)
    if votes_required != 2 or distance_cap_m != 4.0 or thresholds.shape != (3,):
        raise RuntimeError("sealed consensus/metric calibration drift")
    with np.load(c09 / "artifacts/c09_balanced_alias_pairs.npz", allow_pickle=False) as archive:
        balanced_label = archive["label"].astype(np.uint8)
        balanced_family = archive["family"].astype(str)
        physical = archive["physical_positive"].astype(bool)
    with np.load(c09 / "artifacts/c09_runtime_candidate_pairs.npz", allow_pickle=False) as archive:
        runtime_label = archive["label"].astype(np.uint8)
        runtime_family = archive["family"].astype(str)
        runtime_distance = archive["distance_m"].astype(np.float64)
        decision_queries = int(archive["decision_queries"])
        queries_with_candidate = int(archive["queries_with_candidate"])
        queries_with_positive = int(archive["queries_with_positive"])
    balanced_votes = []
    runtime_votes = []
    archived_scores = []
    for seed in range(3):
        with np.load(c09 / f"artifacts/seed{seed}_c09_qualification_scores.npz", allow_pickle=False) as archive:
            if not (
                np.array_equal(archive["balanced_label"], balanced_label)
                and np.array_equal(archive["runtime_label"], runtime_label)
            ):
                raise RuntimeError(f"archived C09 score identity drift: seed {seed}")
            balanced_score = archive["balanced_score"].astype(np.float64)
            runtime_score = archive["runtime_score"].astype(np.float64)
        balanced_votes.append(balanced_score >= thresholds[seed])
        runtime_votes.append(runtime_score >= thresholds[seed])
        archived_scores.append((balanced_score, runtime_score))
    balanced_accept = np.sum(np.stack(balanced_votes), axis=0) >= votes_required
    runtime_accept = (
        (np.sum(np.stack(runtime_votes), axis=0) >= votes_required)
        & (runtime_distance <= distance_cap_m + 1e-12)
    )
    balanced = fixed_threshold_metrics(
        balanced_accept.astype(np.float64), balanced_label, balanced_family, .5
    )
    physical_balanced = fixed_threshold_metrics(
        balanced_accept[physical].astype(np.float64), balanced_label[physical],
        balanced_family[physical], .5,
    )
    runtime = fixed_threshold_metrics(
        runtime_accept.astype(np.float64), runtime_label, runtime_family, .5
    )
    zero_negative = sorted(
        family for family, metrics in runtime["per_family"].items()
        if metrics["negative_support"] == 0 and metrics["precision_identifiable"] is False
    )
    selection_metrics = calibration["selected_metrics"]
    checks = {
        "calibration_was_c01_c08_only": calibration["c09_worlds_read"] == 0,
        "balanced_gate": _balanced_gate(balanced),
        "physical_only_balanced_gate": _balanced_gate(physical_balanced),
        "runtime_aggregate_gate": _runtime_gate(runtime),
        "runtime_zero_negative_families_explicit": zero_negative == ["S02", "S03"],
        "exact_c09_population": (
            decision_queries == 4_085 and queries_with_candidate == 3_968
            and queries_with_positive == 3_955 and len(runtime_label) == 4_201
            and int(np.sum(runtime_label)) == 3_955
            and len(balanced_label) == 260
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_factorized_consensus_metric_c09_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Can a C07-C08-selected multi-seed consensus and metric-locality contract recover safe Factorized association on C09?",
        "votes_required": votes_required,
        "distance_cap_m": distance_cap_m,
        "selection_worlds": calibration["selection_worlds"],
        "selection_sequences": calibration["selection_sequences"],
        "selection_metrics": selection_metrics,
        "c09_worlds": 10,
        "c09_sequences": 24_462,
        "c09_balanced_pairs": len(balanced_label),
        "c09_runtime_decision_queries": decision_queries,
        "c09_runtime_pairs": len(runtime_label),
        "c09_metrics": {
            "balanced": balanced,
            "balanced_physical_only": physical_balanced,
            "runtime": runtime,
        },
        "runtime_zero_negative_families": zero_negative,
        "checks": checks,
        "optimizer_steps": 0,
        "model_updates": 0,
        "checkpoint_selection_steps": 0,
        "threshold_selection_steps": 0,
        "c10_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "recommended_next": "FREEZE_OFFLINE_GRAPH_QUALIFICATION" if passed else "STOP_FACTORIZED_ASSOCIATION_CLAIM",
    }
    np.savez_compressed(
        output / "c09_consensus_metric_decisions.npz",
        balanced_label=balanced_label, balanced_family=balanced_family,
        balanced_accept=balanced_accept, physical_mask=physical,
        runtime_label=runtime_label, runtime_family=runtime_family,
        runtime_distance_m=runtime_distance, runtime_accept=runtime_accept,
        balanced_scores=np.stack([row[0] for row in archived_scores]),
        runtime_scores=np.stack([row[1] for row in archived_scores]),
        seed_thresholds=thresholds,
    )
    write_json(output / "metrics.json", result)
    _plot(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
