#!/usr/bin/env python3
"""Publish diagnostic-only GSE geometry-uncertainty evidence from sealed C09."""

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


EXPECTED_RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
FIGURE_ID = "gse_uncertainty_analysis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise RuntimeError(f"empty uncertainty source: {path.name}")
    return records


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    gate = load_json(run_dir / "metrics/perception_gate.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or gate.get("passed") is not True
        or gate.get("strict_test_worlds_read") != 0
        or gate.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("uncertainty analysis requires the sealed perception PASS")

    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    calibration = run_dir / "artifacts/calibration"
    source_paths = [
        run_dir / "RUN_STATE.json",
        run_dir / "metrics/summary.json",
        run_dir / "metrics/perception_gate.json",
    ]
    seed_data: dict[int, dict] = {}
    for seed in (0, 1, 2):
        seed_summary_path = calibration / f"seed{seed}_summary.json"
        curve_path = calibration / f"seed{seed}_uncertainty_diagnostic_curve.jsonl"
        source_paths.extend((seed_summary_path, curve_path))
        seed_summary = load_json(seed_summary_path)
        diagnostic = seed_summary.get("uncertainty_diagnostic", {})
        curve = _jsonl(curve_path)
        if (
            seed_summary.get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1"
            or seed_summary.get("strict_test_worlds_read") != 0
            or seed_summary.get("mtare_worlds_read") != 0
            or diagnostic.get("selection_effect") != "NONE"
            or diagnostic.get("curve_file") != curve_path.name
            or diagnostic.get("curve_records") != len(curve)
            or diagnostic.get("bins") != len(curve)
            or sum(int(row["count"]) for row in curve) != diagnostic.get("frames")
        ):
            raise RuntimeError(f"seed {seed} uncertainty diagnostic is incomplete or selection-bearing")
        seed_data[seed] = {"summary": diagnostic, "curve": curve}
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"uncertainty source is absent from the run seal: {relative}")

    names = tuple(f"{FIGURE_ID}.{suffix}" for suffix in ("png", "pdf", "svg", "csv")) + (
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("uncertainty-analysis destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "seed",
                "bin_index",
                "count",
                "uncertainty_min",
                "uncertainty_mean",
                "uncertainty_max",
                "predicted_variance_mean",
                "geometry_residual_mean",
                "absolute_calibration_gap",
                "event_error_rate",
            )
        )
        fields = (
            "bin_index",
            "count",
            "uncertainty_min",
            "uncertainty_mean",
            "uncertainty_max",
            "predicted_variance_mean",
            "geometry_residual_mean",
            "absolute_calibration_gap",
            "event_error_rate",
        )
        for seed, data in seed_data.items():
            for row in data["curve"]:
                writer.writerow((seed, *(row[name] for name in fields)))

    colors = ("#376996", "#E76F51", "#2A9D8F")
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True)
    for seed, color in zip((0, 1, 2), colors, strict=True):
        curve = seed_data[seed]["curve"]
        uncertainty = [row["uncertainty_mean"] for row in curve]
        axes[0, 0].plot(
            uncertainty,
            [row["predicted_variance_mean"] for row in curve],
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"seed {seed} predicted",
        )
        axes[0, 0].plot(
            uncertainty,
            [row["geometry_residual_mean"] for row in curve],
            color=color,
            linewidth=1.6,
            label=f"seed {seed} observed",
        )
        axes[0, 1].plot(
            uncertainty,
            np.asarray([row["event_error_rate"] for row in curve]) * 100.0,
            marker="o",
            markersize=3.0,
            color=color,
            linewidth=1.25,
            label=f"seed {seed}",
        )

    axes[0, 0].set_title("Geometry residual-scale calibration")
    axes[0, 0].set_xlabel("Predicted uncertainty")
    axes[0, 0].set_ylabel("Normalized loss / predicted variance")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].set_title("Transfer to structural-event reliability")
    axes[0, 1].set_xlabel("Predicted geometry uncertainty")
    axes[0, 1].set_ylabel("Event error rate (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    seeds = np.arange(3)
    gaps = [seed_data[seed]["summary"]["weighted_absolute_calibration_gap"] for seed in (0, 1, 2)]
    axes[1, 0].bar(seeds, gaps, color=colors)
    axes[1, 0].set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axes[1, 0].set_title("Geometry calibration gap")
    axes[1, 0].set_ylabel("Weighted absolute gap")

    width = 0.34
    geometry_corr = [seed_data[seed]["summary"]["geometry_residual_pearson"] for seed in (0, 1, 2)]
    event_corr = [seed_data[seed]["summary"]["event_error_pearson"] for seed in (0, 1, 2)]
    axes[1, 1].bar(seeds - width / 2, np.asarray([np.nan if value is None else value for value in geometry_corr]), width, color="#7B6D8D", label="Geometry residual")
    axes[1, 1].bar(seeds + width / 2, np.asarray([np.nan if value is None else value for value in event_corr]), width, color="#E49B32", label="Event error")
    axes[1, 1].axhline(0.0, color="#17212B", linewidth=0.8)
    axes[1, 1].set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axes[1, 1].set_ylim(-1.0, 1.0)
    axes[1, 1].set_title("Uncertainty-error correlation")
    axes[1, 1].set_ylabel("Pearson correlation")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("GSE-Graph diagnostic uncertainty audit on C09 validation", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(
        destination / f"{FIGURE_ID}_source.json",
        {
            "schema_version": "gse_uncertainty_analysis_source_v1",
            "definition": "predicted squared geometry uncertainty versus the exact masked normalized Smooth-L1 geometry residual",
            "selection_effect": "NONE",
            "seeds": {str(seed): data["summary"] for seed, data in seed_data.items()},
        },
    )
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all ten equal-frequency bins for all three seeds; diagnostic only and excluded from checkpoint, threshold and gate selection",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in source_paths},
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
