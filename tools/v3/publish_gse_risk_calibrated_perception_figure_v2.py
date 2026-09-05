#!/usr/bin/env python3
"""Publish paper-ready risk-calibrated GSE perception evidence."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
EXPECTED_STATUS = "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"
FIGURE_ID = "gse_perception_validation"


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
    summary = load_json(run_dir / "metrics/summary.json")
    gate = load_json(run_dir / "metrics/corrected_perception_gate.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or gate.get("passed") is not True
        or gate.get("strict_test_worlds_read") != 0
        or gate.get("mtare_worlds_read") != 0
        or summary.get("c10_worlds_read") != 0
        or summary.get("model_updates") != 0
        or summary.get("optimizer_steps") != 0
    ):
        raise RuntimeError("perception paper figure requires the sealed validation PASS")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    sources = (
        run_dir / "RUN_STATE.json",
        run_dir / "metrics/summary.json",
        run_dir / "metrics/corrected_perception_gate.json",
        run_dir / "artifacts/risk_calibrated_slope_c09/summary.json",
        run_dir / "artifacts/risk_calibrated_slope_c09/risk_calibration.json",
    )
    for path in sources:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"paper source is absent from the run seal: {relative}")

    names = tuple(
        f"{FIGURE_ID}.{suffix}"
        for suffix in ("png", "pdf", "svg", "csv")
    ) + (f"{FIGURE_ID}_source.json", f"{FIGURE_ID}_provenance.json", f"{FIGURE_ID}_sha256.txt")
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("perception figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    event = gate["event_gate"]
    geometry = gate["geometry_gate"]
    association = gate["association_gate"]
    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("section", "seed_or_field", "metric", "value"))
        for row in event["per_seed"]:
            writer.writerow(("event", row["seed"], "gse_calibrated_macro_f1", row["gse_calibrated_macro_f1"]))
            writer.writerow(("event", row["seed"], "exit_only_macro_f1", row["exit_only_macro_f1"]))
        for field, value in geometry["per_field_relative_improvement"].items():
            writer.writerow(("geometry", field, "relative_mae_improvement", value))
        for row in association["per_seed"]:
            writer.writerow(("association", row["seed"], "place_precision", row["place_precision"]))
            writer.writerow(("association", row["seed"], "place_false_accept_rate", row["place_false_accept_rate"]))
            writer.writerow(("association", row["seed"], "exit_precision", row["exit_precision"]))
            writer.writerow(("association", row["seed"], "exit_false_accept_rate", row["exit_false_accept_rate"]))

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    seeds = np.arange(3)
    width = 0.34
    axes[0].bar(seeds - width / 2, [row["exit_only_macro_f1"] for row in event["per_seed"]], width, color="#9AA5B1", label="Exit-only M1D")
    axes[0].bar(seeds + width / 2, [row["gse_calibrated_macro_f1"] for row in event["per_seed"]], width, color="#2A9D8F", label="GSE-Graph (ours)")
    axes[0].set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Five-event macro-F1")
    axes[0].set_title("Structural events")
    axes[0].legend(frameon=False, fontsize=8)

    field_order = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    labels = ("Width", "Height", "Slope", "Curvature")
    improvements = [geometry["per_field_relative_improvement"][field] for field in field_order]
    colors = ["#376996" if value >= 0 else "#E76F51" for value in improvements]
    axes[1].bar(np.arange(4), np.asarray(improvements) * 100.0, color=colors)
    axes[1].axhline(10.0, color="#2A9D8F", linestyle="--", linewidth=1.2, label="10% macro gate")
    axes[1].axhline(-5.0, color="#E76F51", linestyle=":", linewidth=1.2, label="−5% regression limit")
    axes[1].set_xticks(np.arange(4), labels, rotation=20)
    axes[1].set_ylabel("Relative MAE improvement (%)")
    axes[1].set_title("Continuous geometry")
    axes[1].legend(frameon=False, fontsize=7)

    place_precision = [row["place_precision"] for row in association["per_seed"]]
    exit_precision = [row["exit_precision"] for row in association["per_seed"]]
    axes[2].bar(seeds - width / 2, place_precision, width, color="#7B6D8D", label="Place association")
    axes[2].bar(seeds + width / 2, exit_precision, width, color="#E49B32", label="Exit association")
    axes[2].axhline(0.98, color="#17212B", linestyle="--", linewidth=1.2, label="0.98 safety gate")
    axes[2].set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axes[2].set_ylim(0.94, 1.002)
    axes[2].set_ylabel("Causal merge precision")
    axes[2].set_title("Past-only association")
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(destination / f"{FIGURE_ID}_source.json", gate)
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all three predeclared seeds and all four fields; slope residual scale selected only on C07-C08 by fixed-grid parent-level maximin improvement",
            "slope_residual_scale": load_json(run_dir / "artifacts/risk_calibrated_slope_c09/risk_calibration.json")["residual_scale"],
            "c09_used_for_scale_selection": False,
            "c10_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in sources},
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
