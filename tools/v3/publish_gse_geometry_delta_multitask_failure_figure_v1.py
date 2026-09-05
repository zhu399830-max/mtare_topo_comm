#!/usr/bin/env python3
"""Publish the sealed geometry-delta component-success/event-failure figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json


FIGURE_ID = "gse_geometry_delta_multitask_failure"
MULTITASK = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260827_gse_causal_geometry_delta_multitask_training_v1_seed0"
)
OBSERVABILITY = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260827_gse_causal_geometry_delta_observability_v1_seed0"
)
OLD_DIRECTIONAL = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260826_gse_directional_structural_event_training_v1_seed0"
)
EVENT_ORDER = ("junction", "terminal", "turn", "geometry_transition")
EVENT_LABELS = ("Junction", "Terminal", "Turn", "Change point")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    destination.relative_to(PROJECT_ROOT)
    seals = {
        "multitask": verify_failed_component_run_seal(
            PROJECT_ROOT,
            MULTITASK,
            "FAIL_GSE_CAUSAL_GEOMETRY_DELTA_MULTITASK_TRAINING_V1",
        ),
        "observability": verify_complete_run_seal(
            PROJECT_ROOT,
            OBSERVABILITY,
            "PASS_GSE_CAUSAL_GEOMETRY_DELTA_OBSERVABILITY_V1",
        ),
        "old_directional": verify_failed_component_run_seal(
            PROJECT_ROOT,
            OLD_DIRECTIONAL,
            "FAIL_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1",
        ),
    }
    if any(
        record.get("strict_test_worlds_read") != 0
        or record.get("mtare_worlds_read") != 0
        for record in seals.values()
    ):
        raise RuntimeError("geometry-delta paper source isolation drift")
    summary = load_json(MULTITASK / "artifacts/training/summary.json")
    if summary.get("scientific_pass") is not False or summary.get("optimizer_steps") != 52236:
        raise RuntimeError("sealed multitask failure summary drift")

    labels = ("Frozen\nbaseline", "Seed 0", "Seed 1", "Seed 2", "Ensemble")
    baseline_delta = [
        float(summary["baseline_geometry_delta_metrics"][str(seed)]["normalized_mae"])
        for seed in (0, 1, 2)
    ]
    learned_delta = [
        float(summary["seeds"][str(seed)]["metrics"]["geometry_delta"]["normalized_mae"])
        for seed in (0, 1, 2)
    ]
    delta_mae = [
        float(summary["baseline_geometry_delta_metrics"]["ensemble"]["normalized_mae"]),
        *learned_delta,
        float(summary["ensemble_geometry_delta_metrics"]["normalized_mae"]),
    ]
    delta_improvement = [
        0.0,
        *[100.0 * float(summary["seed_delta_normalized_mae_improvement"][str(seed)]) for seed in (0, 1, 2)],
        100.0 * float(summary["ensemble_delta_normalized_mae_improvement"]),
    ]
    old_event = float(
        summary["baseline_event_evidence"]["old_directional_head"]["ensemble"]["event"]["macro_f1"]
    )
    event_macro_f1 = [
        old_event,
        *[
            float(summary["seeds"][str(seed)]["metrics"]["event"]["event"]["macro_f1"])
            for seed in (0, 1, 2)
        ],
        float(summary["ensemble_event_metrics"]["event"]["macro_f1"]),
    ]
    coverage = summary["ensemble_event_metrics"]["identity_coverage"]
    identity_coverage = [
        100.0 * float(coverage[name]["correct_class_identity_coverage"])
        for name in EVENT_ORDER
    ]
    identity_counts = [
        f'{int(coverage[name]["covered_identities"])}/{int(coverage[name]["teacher_identities"])}'
        for name in EVENT_ORDER
    ]

    suffixes = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
    targets = [destination / f"{FIGURE_ID}{suffix}" for suffix in suffixes]
    if any(path.exists() for path in targets):
        raise RuntimeError("geometry-delta paper figure exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("panel", "series", "category", "value", "numerator", "denominator"))
        for label, value in zip(labels, delta_mae, strict=True):
            writer.writerow(("delta_mae", "normalized_mae", label.replace("\n", " "), value, "", ""))
        for label, value in zip(labels, delta_improvement, strict=True):
            writer.writerow(("delta_improvement", "percent", label.replace("\n", " "), value, "", ""))
        for label, value in zip(labels, event_macro_f1, strict=True):
            writer.writerow(("event_macro_f1", "macro_f1", label.replace("\n", " "), value, "", ""))
        for name, value in zip(EVENT_ORDER, identity_coverage, strict=True):
            item = coverage[name]
            writer.writerow(
                (
                    "identity_coverage",
                    "ensemble",
                    name,
                    value,
                    item["covered_identities"],
                    item["teacher_identities"],
                )
            )

    colors = ("#9AA5B1", "#5B8FF9", "#61DDAA", "#65789B", "#E8684A")
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    axes[0, 0].bar(x, delta_mae, color=colors)
    axes[0, 0].axhline(delta_mae[0] * 0.9, color="#C8553D", linestyle="--", linewidth=1.2, label="10% improvement gate")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Normalized delta MAE (lower is better)")
    axes[0, 0].set_title("(a) Continuous geometry change is learned")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.2)

    axes[0, 1].bar(x, delta_improvement, color=colors)
    axes[0, 1].axhline(10.0, color="#C8553D", linestyle="--", linewidth=1.2, label="Pre-registered gate")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Improvement over frozen prediction (%)")
    axes[0, 1].set_title("(b) All seeds exceed the geometry gate")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].bar(x, event_macro_f1, color=colors)
    axes[1, 0].axhline(0.7379041032, color="#C8553D", linestyle="--", linewidth=1.2, label="Event gate")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0.60, 0.76)
    axes[1, 0].set_ylabel("Five-class event macro-F1")
    axes[1, 0].set_title("(c) Discrete event quality does not improve")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    event_x = np.arange(len(EVENT_ORDER))
    axes[1, 1].bar(event_x, identity_coverage, color=("#376996", "#2A9D8F", "#E49B32", "#C8553D"))
    axes[1, 1].axhline(40.0, color="#C8553D", linestyle="--", linewidth=1.2, label="Rare-event floor")
    axes[1, 1].set_xticks(event_x, EVENT_LABELS, rotation=12)
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_ylabel("Correct-class identity coverage (%)")
    axes[1, 1].set_title("(d) Change-point identities remain uncovered")
    for index, text_value in enumerate(identity_counts):
        axes[1, 1].text(index, identity_coverage[index] + 2.0, text_value, ha="center", fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            destination / f"{FIGURE_ID}.{suffix}",
            dpi=280,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    source = {
        "schema_version": "gse_geometry_delta_multitask_failure_figure_source_v1",
        "labels": [value.replace("\n", " ") for value in labels],
        "baseline_seed_delta_normalized_mae": baseline_delta,
        "delta_normalized_mae": delta_mae,
        "delta_improvement_percent": delta_improvement,
        "event_macro_f1": event_macro_f1,
        "identity_coverage_percent": dict(zip(EVENT_ORDER, identity_coverage, strict=True)),
        "identity_counts": dict(zip(EVENT_ORDER, identity_counts, strict=True)),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "scientific_pass": bool(summary["scientific_pass"]),
    }
    write_json(destination / f"{FIGURE_ID}_source.json", source)
    source_files = [
        MULTITASK / "RUN_STATE.json",
        MULTITASK / "metrics/summary.json",
        MULTITASK / "artifacts/training/summary.json",
        MULTITASK / "artifacts/evidence_sha256.txt",
        OBSERVABILITY / "metrics/summary.json",
        OBSERVABILITY / "artifacts/evidence_sha256.txt",
        OLD_DIRECTIONAL / "metrics/summary.json",
        OLD_DIRECTIONAL / "artifacts/evidence_sha256.txt",
    ]
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "development_scope": "C01-C08 only; plotted evaluation is C07-C08 selection",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "source_seals": {name: value["seal_sha256"] for name, value in seals.items()},
            "source_files": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_files
            },
            "generator": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n"
            for path in sorted(targets)
            if path != manifest
        ),
        encoding="utf-8",
    )
    return {
        "figure_id": FIGURE_ID,
        "published_files": len(targets),
        "manifest_sha256": _sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/gse_graph",
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
