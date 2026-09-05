#!/usr/bin/env python3
"""Publish fair GSE-versus-M1D exit geometry diagnostics from sealed C09 evidence."""

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
FIGURE_ID = "gse_exit_token_validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


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
        raise RuntimeError("exit-token comparison requires the sealed perception PASS")
    m1d_path = run_dir / "artifacts/exit_only_baseline/summary.json"
    m1d = load_json(m1d_path)
    if m1d.get("status") != "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1":
        raise RuntimeError("exit-only comparison source is not PASS")
    calibration = run_dir / "artifacts/calibration"
    seed_paths = [calibration / f"seed{seed}_summary.json" for seed in (0, 1, 2)]
    gse = [load_json(path) for path in seed_paths]
    if any(item.get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1" for item in gse):
        raise RuntimeError("GSE token comparison requires all three calibrated seeds")
    if [int(item["seed"]) for item in m1d["per_seed"]] != [0, 1, 2]:
        raise RuntimeError("M1D token comparison seed identity drift")

    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    source_paths = [run_dir / "RUN_STATE.json", run_dir / "metrics/summary.json", run_dir / "metrics/perception_gate.json", m1d_path, *seed_paths]
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"exit-token source is absent from the run seal: {relative}")

    rows = []
    for seed in (0, 1, 2):
        gse_direction = gse[seed]["exit_tokens"]["direction_at_selected_threshold"]
        gse_count = gse[seed]["exit_tokens"]["count_at_selected_threshold"]
        baseline = m1d["per_seed"][seed]
        if float(gse_direction["matching_tolerance_deg"]) != 20.0:
            raise RuntimeError("GSE and M1D direction tolerance contract drift")
        rows.append(
            {
                "seed": seed,
                "gse_direction_f1": float(gse_direction["f1"]),
                "m1d_direction_f1": float(baseline["direction"]["f1"]),
                "gse_angular_error_deg": _optional_float(gse_direction["mean_matched_angular_error_deg"]),
                "m1d_angular_error_deg": _optional_float(baseline["direction"]["mean_matched_angular_error_deg"]),
                "gse_count_exact": float(gse_count["exact_accuracy"]),
                "m1d_count_exact": float(baseline["count"]["exact_accuracy"]),
                "gse_count_mae": float(gse_count["mean_absolute_error"]),
                "m1d_count_mae": float(baseline["count"]["mean_absolute_error"]),
            }
        )

    names = tuple(f"{FIGURE_ID}.{suffix}" for suffix in ("png", "pdf", "svg", "csv")) + (
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("exit-token figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), constrained_layout=True)
    seeds = np.arange(3)
    width = 0.34
    panels = (
        ("direction_f1", "Exit-direction set F1", (0.0, 1.0)),
        ("angular_error_deg", "Matched direction error (deg)", None),
        ("count_exact", "Exit-count exact accuracy", (0.0, 1.0)),
        ("count_mae", "Exit-count MAE", None),
    )
    for axis, (metric, title, ylim) in zip(axes.flat, panels, strict=True):
        baseline_values = [np.nan if row[f"m1d_{metric}"] is None else row[f"m1d_{metric}"] for row in rows]
        gse_values = [np.nan if row[f"gse_{metric}"] is None else row[f"gse_{metric}"] for row in rows]
        axis.bar(seeds - width / 2, baseline_values, width, color="#9AA5B1", label="Exit-only M1D")
        axis.bar(seeds + width / 2, gse_values, width, color="#2A9D8F", label="GSE-Graph")
        axis.set_xticks(seeds, [f"seed {seed}" for seed in seeds])
        axis.set_title(title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("C09 exit geometry under the same 20° matching contract", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(destination / f"{FIGURE_ID}_source.json", {"schema_version": "gse_exit_token_comparison_source_v1", "matching_tolerance_deg": 20.0, "rows": rows})
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all three paired seeds; identical 20-degree direction matching; GSE uses its validation-selected presence threshold",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in source_paths},
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text("".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest), encoding="utf-8")
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
