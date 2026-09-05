#!/usr/bin/env python3
"""Publish the transition-Teacher and rare-event failure-analysis figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_rare_event_corrective import evaluate_rare_event_corrective


FIGURE_ID = "gse_transition_teacher_audit"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
VERIFIER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
RARE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_rare_event_corrective_training_v1_seed0"
DIRECTIONAL = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
EVENT_NAMES = ("corridor", "junction", "terminal", "turn", "geometry_transition")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _baseline_metrics() -> dict:
    dataset = GSESequenceDataset(DATASET, "train", augment_azimuth=False)
    with np.load(VERIFIER / "artifacts/pair_cache/pairs.npz", allow_pickle=False) as archive:
        global_index = archive["compact_to_global_sequence_index"].astype(np.int64)
        partition = archive["partition_code"].astype(np.uint8)
        parent = archive["parent_id"].astype(str)
    probabilities = []
    for seed in (0, 1, 2):
        features = np.load(
            VERIFIER / f"artifacts/models/seed{seed}/frozen_observation_features.npy",
            mmap_mode="r",
        )
        value = np.asarray(features[:, :5], dtype=np.float64)
        value /= value.sum(axis=1, keepdims=True)
        probabilities.append(value)
    probability = np.mean(np.stack(probabilities), axis=0)
    row_by_global = {
        int(record["global_sequence_index"]): row for row, record in enumerate(dataset.records)
    }
    dataset_rows = np.asarray([row_by_global[int(value)] for value in global_index], dtype=np.int64)
    event = dataset.event_labels()[dataset_rows]
    identity = dataset.association_labels()[dataset_rows]
    selection = partition == 1
    metrics = evaluate_rare_event_corrective(
        probability[selection],
        event[selection],
        identity[selection],
        np.asarray([value[:3] for value in parent[selection]]),
    )
    return metrics


def _teacher_statistics() -> dict:
    traversals = _read_jsonl(TEACHER / "artifacts/traversal_manifest.jsonl")
    lengths = {
        (str(row["parent_id"]), str(row["edge_id"])): float(row["length_m"])
        for row in traversals
    }
    identities = [
        row
        for row in _read_jsonl(TEACHER / "artifacts/edge_event_identities.jsonl")
        if str(row["parent_id"]).endswith(("_C07", "_C08"))
    ]
    transition = [row for row in identities if row["event"] == "geometry_transition"]
    if len(transition) != 744:
        raise RuntimeError("transition identity count drift")
    endpoint_distance = []
    span = []
    one_direction = 0
    physical_edges = Counter()
    family = Counter()
    for row in transition:
        edge_id = str(row["identity"]).split(":")[1]
        length = lengths[(str(row["parent_id"]), edge_id)]
        center = float(row["canonical_center_arc_m"])
        endpoint_distance.append(min(center, length - center))
        span.append(float(row["canonical_end_arc_m"]) - float(row["canonical_start_arc_m"]))
        one_direction += int(row["traversal_count"] == 1)
        physical_edges[(str(row["parent_id"]), edge_id)] += 1
        family[str(row["parent_id"])[:3]] += 1

    dataset = GSESequenceDataset(DATASET, "train", augment_azimuth=False)
    selected_rows = np.asarray(
        [
            index
            for index, record in enumerate(dataset.records)
            if str(record["parent_id"]).endswith(("_C07", "_C08"))
        ],
        dtype=np.int64,
    )
    event = dataset.event_labels()[selected_rows]
    identity = dataset.association_labels()[selected_rows]
    identity_counts = {
        EVENT_NAMES[class_index]: int(len(np.unique(identity[event == class_index])))
        for class_index in range(1, 5)
    }
    if identity_counts != {
        "junction": 146,
        "terminal": 128,
        "turn": 95,
        "geometry_transition": 744,
    }:
        raise RuntimeError("selection identity composition drift")

    endpoint_bins = {
        "0–5 m": int(np.sum(np.asarray(endpoint_distance) <= 5.0)),
        "5–10 m": int(np.sum((np.asarray(endpoint_distance) > 5.0) & (np.asarray(endpoint_distance) <= 10.0))),
        "10–12 m": int(np.sum((np.asarray(endpoint_distance) > 10.0) & (np.asarray(endpoint_distance) <= 12.0))),
        "12–15 m": int(np.sum(np.asarray(endpoint_distance) > 12.0)),
    }
    span_bins = {
        "≤1 m": int(np.sum(np.asarray(span) <= 1.0 + 1e-9)),
        "1–2 m": int(np.sum((np.asarray(span) > 1.0 + 1e-9) & (np.asarray(span) <= 2.0 + 1e-9))),
        "2–3 m": int(np.sum((np.asarray(span) > 2.0 + 1e-9) & (np.asarray(span) <= 3.0 + 1e-9))),
        "3–5 m": int(np.sum((np.asarray(span) > 3.0 + 1e-9) & (np.asarray(span) <= 5.0 + 1e-9))),
        ">5 m": int(np.sum(np.asarray(span) > 5.0 + 1e-9)),
    }
    return {
        "identity_counts": identity_counts,
        "transition_identities": len(transition),
        "transition_fraction_of_structural_identities": len(transition) / sum(identity_counts.values()),
        "endpoint_distance_bins": endpoint_bins,
        "within_10m_fraction": sum(list(endpoint_bins.values())[:2]) / len(transition),
        "within_12m_fraction": sum(list(endpoint_bins.values())[:3]) / len(transition),
        "span_bins": span_bins,
        "span_le_2m_fraction": sum(list(span_bins.values())[:2]) / len(transition),
        "span_le_5m_fraction": sum(list(span_bins.values())[:4]) / len(transition),
        "one_direction_count": one_direction,
        "one_direction_fraction": one_direction / len(transition),
        "physical_edges_with_transition": len(physical_edges),
        "physical_edges_with_multiple_transition_identities": sum(value > 1 for value in physical_edges.values()),
        "maximum_transition_identities_per_edge": max(physical_edges.values()),
        "per_family_transition_identities": dict(sorted(family.items())),
    }


def _coverage(metrics: dict) -> dict[str, float]:
    return {
        event: float(metrics["identity_coverage"][event]["correct_class_identity_coverage"])
        for event in EVENT_NAMES[1:]
    }


def publish(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    destination.relative_to(PROJECT_ROOT)
    seals = {
        "teacher": verify_complete_run_seal(PROJECT_ROOT, TEACHER, "PASS_GSE_TEACHER_MANIFEST_V1"),
        "dataset": verify_complete_run_seal(PROJECT_ROOT, DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"),
        "verifier": verify_failed_component_run_seal(PROJECT_ROOT, VERIFIER, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"),
        "rare": verify_failed_component_run_seal(PROJECT_ROOT, RARE, "FAIL_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"),
        "directional": verify_failed_component_run_seal(PROJECT_ROOT, DIRECTIONAL, "FAIL_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1"),
    }
    if any(record.get("strict_test_worlds_read") != 0 or record.get("mtare_worlds_read") != 0 for record in seals.values()):
        raise RuntimeError("transition audit source isolation drift")
    baseline = _baseline_metrics()
    rare = load_json(RARE / "artifacts/training/summary.json")["ensemble_metrics"]
    directional = load_json(DIRECTIONAL / "artifacts/training/summary.json")["ensemble_metrics"]
    methods = {
        "Frozen event head": _coverage(baseline),
        "292D residual": _coverage(rare),
        "Directional head": _coverage(directional),
    }
    teacher = _teacher_statistics()
    suffixes = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
    targets = [destination / f"{FIGURE_ID}{suffix}" for suffix in suffixes]
    if any(path.exists() for path in targets):
        raise RuntimeError("transition audit figure target exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("section", "series", "category", "count", "total", "fraction"))
        total_identity = sum(teacher["identity_counts"].values())
        for key, value in teacher["identity_counts"].items():
            writer.writerow(("identity_composition", "Teacher", key, value, total_identity, value / total_identity))
        for key, value in teacher["endpoint_distance_bins"].items():
            writer.writerow(("endpoint_distance", "geometry_transition", key, value, 744, value / 744))
        for key, value in teacher["span_bins"].items():
            writer.writerow(("event_span", "geometry_transition", key, value, 744, value / 744))
        for method, coverage in methods.items():
            for key, value in coverage.items():
                writer.writerow(("identity_coverage", method, key, "", "", value))

    colors = ("#376996", "#E49B32", "#2A9D8F", "#C8553D")
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    identity_names = ("junction", "terminal", "turn", "geometry_transition")
    identity_values = [teacher["identity_counts"][name] for name in identity_names]
    axes[0, 0].bar(np.arange(4), identity_values, color=colors)
    axes[0, 0].set_xticks(np.arange(4), ("Junction", "Terminal", "Turn", "Transition"), rotation=18)
    axes[0, 0].set_ylabel("Unique Teacher identities")
    axes[0, 0].set_title("(a) Transition dominates node supervision")
    axes[0, 0].grid(axis="y", alpha=0.2)

    endpoint_names = tuple(teacher["endpoint_distance_bins"])
    endpoint_values = tuple(teacher["endpoint_distance_bins"].values())
    axes[0, 1].bar(np.arange(len(endpoint_names)), endpoint_values, color="#C8553D")
    axes[0, 1].set_xticks(np.arange(len(endpoint_names)), endpoint_names)
    axes[0, 1].set_ylabel("Transition identities")
    axes[0, 1].set_title("(b) Distance to physical-edge endpoint")
    axes[0, 1].grid(axis="y", alpha=0.2)

    span_names = tuple(teacher["span_bins"])
    span_values = tuple(teacher["span_bins"].values())
    axes[1, 0].bar(np.arange(len(span_names)), span_values, color="#E49B32")
    axes[1, 0].set_xticks(np.arange(len(span_names)), span_names)
    axes[1, 0].set_ylabel("Transition identities")
    axes[1, 0].set_title("(c) Canonical labeled-event span")
    axes[1, 0].grid(axis="y", alpha=0.2)

    x = np.arange(4)
    width = 0.24
    method_colors = ("#9AA5B1", "#7B61A8", "#2A9D8F")
    for index, (method, coverage) in enumerate(methods.items()):
        axes[1, 1].bar(
            x + (index - 1) * width,
            [coverage[name] * 100.0 for name in identity_names],
            width,
            color=method_colors[index],
            label=method,
        )
    axes[1, 1].axhline(40.0, color="#C8553D", linestyle="--", linewidth=1.2, label="Rare-event gate")
    axes[1, 1].set_xticks(x, ("Junction", "Terminal", "Turn", "Transition"), rotation=18)
    axes[1, 1].set_ylabel("Correct-class identity coverage (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("(d) More event capacity does not recover transition")
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    source = {
        "schema_version": "gse_transition_teacher_audit_figure_source_v1",
        "teacher": teacher,
        "method_identity_coverage": methods,
        "formal_directional_metrics": {
            "event_macro_f1": directional["event"]["macro_f1"],
            "structural_selection": directional["structural_selection"],
        },
    }
    write_json(destination / f"{FIGURE_ID}_source.json", source)
    source_files = [
        TEACHER / "artifacts/edge_event_identities.jsonl",
        TEACHER / "artifacts/traversal_manifest.jsonl",
        DATASET / "artifacts/sequence_manifest.jsonl",
        VERIFIER / "artifacts/pair_cache/pairs.npz",
        RARE / "artifacts/training/summary.json",
        DIRECTIONAL / "artifacts/training/summary.json",
        *[VERIFIER / f"artifacts/models/seed{seed}/frozen_observation_features.npy" for seed in (0, 1, 2)],
    ]
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "development_scope": "C07-C08 only for plotted Teacher/method statistics",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "source_seals": {name: value["seal_sha256"] for name, value in seals.items()},
            "source_files": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_files},
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
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

