#!/usr/bin/env python3
"""Execute three-seed Factorized GSE route-conditioned association capacity proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.governance import write_json
from train_gse_factorized_association_capacity_v1 import train_seed


PASS_STATUS = "PASS_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"
FAIL_STATUS = "FAIL_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1"


def _safe_recall(threshold: dict | None) -> float:
    return float(threshold["recall"]) if threshold is not None else 0.0


def _read_curve(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _plot(run_dir: Path, seeds: list[dict], source: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    names = ("descriptor", "no route geometry", "full route-conditioned")
    x = np.arange(3)
    width = .23
    colors = ("#9C755F", "#F28E2B", "#4C78A8")
    for offset, seed in enumerate(seeds):
        values = (
            _safe_recall(seed["descriptor_only_threshold"]),
            _safe_recall(seed["no_route_geometry"]["selection"]["threshold_selection"]),
            _safe_recall(seed["full_route_conditioned"]["selection"]["threshold_selection"]),
        )
        axes[0].bar(x + (offset - 1) * width, values, width=width, color=colors[offset], label=f"seed {offset}")
    axes[0].axhline(.25, color="#D62728", linestyle="--", linewidth=1.2, label="minimum recall")
    axes[0].set_xticks(x, names, rotation=14)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Safe accepted-pair recall")
    axes[0].set_title("Association capacity at safety gate")
    axes[0].legend(frameon=False, fontsize=8)
    for seed in range(3):
        curve = _read_curve(run_dir / f"artifacts/models/seed{seed}/full_route_conditioned/selection_curve.jsonl")
        axes[1].plot(
            [row["recall"] for row in curve], [row["precision"] for row in curve],
            linewidth=1.5, label=f"seed {seed}", color=colors[seed],
        )
    axes[1].axhline(.98, color="#D62728", linestyle="--", linewidth=1.2, label="precision gate")
    axes[1].axvline(.25, color="#777777", linestyle=":", linewidth=1.2, label="recall gate")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(.45, 1.005)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Full route-conditioned precision–recall")
    axes[1].legend(frameon=False, fontsize=8)
    for axis, letter in zip(axes, "AB"):
        axis.text(-.12, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.grid(color="#E6E6E6", linewidth=.7); axis.set_axisbelow(True)
    fig.suptitle("Factorized GSE-Graph decision-node association")
    prefix = run_dir / "previews/gse_factorized_association_capacity"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    write_json(prefix.parent / f"{prefix.name}_source.json", source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
        parser.add_argument(f"--tokens{seed}", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.monotonic()
    seeds = []
    for seed in range(3):
        output = run_dir / f"artifacts/models/seed{seed}"
        seeds.append(train_seed(
            args.teacher.resolve(), args.manifest.resolve(),
            getattr(args, f"observation{seed}").resolve(),
            getattr(args, f"tokens{seed}").resolve(), output, seed,
        ))
    full_recall = np.asarray([
        _safe_recall(row["full_route_conditioned"]["selection"]["threshold_selection"])
        for row in seeds
    ])
    no_route_recall = np.asarray([
        _safe_recall(row["no_route_geometry"]["selection"]["threshold_selection"])
        for row in seeds
    ])
    descriptor_recall = np.asarray([_safe_recall(row["descriptor_only_threshold"]) for row in seeds])
    physical_checks = []
    for row in seeds:
        current = row["full_route_conditioned"]["physical_only_at_selected_threshold"]
        physical_checks.append(bool(
            current is not None and current["accepted"] > 0
            and current["precision"] >= .98 and current["false_accept_rate"] <= .01
            and current["recall"] >= .25
            and all(value["accepted"] > 0 and value["precision"] >= .95 and value["recall"] >= .10
                    for value in current["per_family"].values())
        ))
    checks = {
        "all_three_full_thresholds_safe": bool(np.all(full_recall >= .25)),
        "all_three_physical_only_safe": all(physical_checks),
        "mean_recall_gain_over_descriptor_ge_0p05": float(np.mean(full_recall) - np.mean(descriptor_recall)) >= .05,
        "mean_recall_gain_over_no_route_ge_0p02": float(np.mean(full_recall) - np.mean(no_route_recall)) >= .02,
        "no_seed_route_regression_gt_0p02": bool(np.all(full_recall >= no_route_recall - .02)),
        "exact_pair_population": all(row["fit_pairs"] == 1584 and row["selection_pairs"] == 548 for row in seeds),
        "zero_forbidden_operations": all(
            row[name] == 0 for row in seeds for name in (
                "backbone_optimizer_steps", "model_inference_frames", "c09_worlds_read",
                "strict_test_worlds_read", "mtare_worlds_read",
            )
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "gse_factorized_association_capacity_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "question": "Can deployment-only learned action tokens and executed-edge geometry safely improve decision-node association over descriptor and no-route ablations?",
        "fit_identity_units": 792, "selection_identity_units": 274,
        "fit_pairs": 1584, "selection_pairs": 548,
        "safe_recall": {
            "descriptor_only": descriptor_recall.tolist(),
            "no_route_geometry": no_route_recall.tolist(),
            "full_route_conditioned": full_recall.tolist(),
        },
        "mean_safe_recall": {
            "descriptor_only": float(np.mean(descriptor_recall)),
            "no_route_geometry": float(np.mean(no_route_recall)),
            "full_route_conditioned": float(np.mean(full_recall)),
        },
        "mean_gain": {
            "over_descriptor": float(np.mean(full_recall) - np.mean(descriptor_recall)),
            "over_no_route_geometry": float(np.mean(full_recall) - np.mean(no_route_recall)),
        },
        "checks": checks, "seeds": seeds,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": int(sum(
            row[variant]["optimizer_steps"] for row in seeds
            for variant in ("full_route_conditioned", "no_route_geometry")
        )),
        "backbone_optimizer_steps": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        "recommended_next": "FREEZE_C09_FACTORIZED_ASSOCIATION_QUALIFICATION" if passed else "STOP_ROUTE_CONDITIONED_ASSOCIATION_CLAIM",
    }
    write_json(run_dir / "metrics/factorized_association_capacity.json", result)
    source = {
        "schema_version": "gse_factorized_association_capacity_figure_source_v1",
        "safe_recall": result["safe_recall"], "mean_safe_recall": result["mean_safe_recall"],
        "mean_gain": result["mean_gain"], "selection_pairs_per_seed": 548,
        "safety_gate": {"precision": .98, "false_accept_rate": .01, "recall": .25},
    }
    _plot(run_dir, seeds, source)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
