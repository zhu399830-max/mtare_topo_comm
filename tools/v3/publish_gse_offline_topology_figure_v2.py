#!/usr/bin/env python3
"""Publish the paper-ready offline GSE topology comparison from sealed PASS evidence."""

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
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate4_20260825_gse_offline_topology_validation_v2_seed0"
EXPECTED_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V2"
FIGURE_ID = "gse_offline_topology"
METHOD_ORDER = (
    "exit_only_rule_graph",
    "nonlearning_geometry_rule_graph",
    "gse_rule_association_ablation",
    "gse_learned_association",
)
METHOD_LABELS = ("Exit-only + rule", "Geometry + rule", "GSE + rule", "GSE-Graph")
METHOD_COLORS = ("#9AA5B1", "#E49B32", "#7B6D8D", "#2A9D8F")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    runner_summary = load_json(run_dir / "metrics/summary.json")
    source_summary_path = run_dir / "artifacts/offline_topology/summary.json"
    source = load_json(source_summary_path)
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or runner_summary.get("overall_status") != EXPECTED_STATUS
        or source.get("overall_status") != EXPECTED_STATUS
        or source.get("scientific_gate", {}).get("passed") is not True
        or source.get("strict_test_worlds_read") != 0
        or source.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("offline topology paper figure requires the sealed validation PASS")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    evidence_sources = [run_dir / "RUN_STATE.json", run_dir / "metrics/summary.json", source_summary_path]
    evidence_sources.extend(
        run_dir / f"artifacts/offline_topology/{method}/summary.json" for method in METHOD_ORDER
    )
    for path in evidence_sources:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"paper source is absent from the run seal: {relative}")

    names = tuple(f"{FIGURE_ID}.{suffix}" for suffix in ("png", "pdf", "svg", "csv")) + (
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("offline topology figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    aggregates = [source["methods"][method]["selected_aggregate"] for method in METHOD_ORDER]
    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "metric", "value", "selected_grid_index", "world_seed_replays"))
        for method, summary, aggregate in zip(METHOD_ORDER, (source["methods"][m] for m in METHOD_ORDER), aggregates, strict=True):
            values = {
                "node_f1": aggregate["node"]["f1"],
                "edge_f1": aggregate["edge"]["f1"],
                "association_precision": aggregate["association"]["precision"],
                "false_loop_merge_rate": aggregate["association"]["false_loop_merge_rate"],
                "connected_component_mean_signed_error": aggregate["invariants"]["connected_component_mean_signed_error"],
                "cycle_rank_mean_signed_error": aggregate["invariants"]["cycle_rank_mean_signed_error"],
            }
            for metric, value in values.items():
                writer.writerow((method, metric, value, summary["selected_grid_index"], aggregate["world_seed_replays"]))

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.55), constrained_layout=True)
    x = np.arange(len(METHOD_ORDER))
    width = 0.35
    axes[0].bar(x - width / 2, [a["node"]["f1"] for a in aggregates], width, color="#376996", label="Node F1")
    axes[0].bar(x + width / 2, [a["edge"]["f1"] for a in aggregates], width, color="#2A9D8F", label="Edge F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Structural graph F1")
    axes[0].set_title("Graph fidelity")
    axes[0].legend(frameon=False, fontsize=8)

    precision = np.asarray([a["association"]["precision"] for a in aggregates])
    false_rate = np.asarray([a["association"]["false_loop_merge_rate"] for a in aggregates])
    axes[1].bar(x - width / 2, precision, width, color="#7B6D8D", label="Merge precision")
    axes[1].bar(x + width / 2, false_rate, width, color="#E76F51", label="False-loop rate")
    axes[1].axhline(0.98, color="#17212B", linestyle="--", linewidth=1.1, label="Precision gate")
    axes[1].axhline(0.01, color="#E76F51", linestyle=":", linewidth=1.1, label="False-loop gate")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Causal association")
    axes[1].legend(frameon=False, fontsize=7)

    components = np.asarray([a["invariants"]["connected_component_mean_signed_error"] for a in aggregates])
    cycles = np.asarray([a["invariants"]["cycle_rank_mean_signed_error"] for a in aggregates])
    axes[2].bar(x - width / 2, components, width, color="#E49B32", label="Components")
    axes[2].bar(x + width / 2, cycles, width, color="#4C78A8", label="Cycle rank")
    axes[2].axhline(0.0, color="#17212B", linewidth=0.8)
    axes[2].axhline(0.25, color="#2A9D8F", linestyle="--", linewidth=1.0)
    axes[2].axhline(-0.25, color="#2A9D8F", linestyle="--", linewidth=1.0, label="GSE bias limit")
    axes[2].set_ylabel("Mean signed count error")
    axes[2].set_title("Topology invariants")
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.set_xticks(x, METHOD_LABELS, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(destination / f"{FIGURE_ID}_source.json", source)
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all predeclared methods and their unique validation-selected configurations; no world, seed or metric cherry-picking",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in evidence_sources},
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
