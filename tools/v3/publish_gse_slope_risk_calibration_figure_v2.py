#!/usr/bin/env python3
"""Publish the slope residual risk-calibration ablation from sealed evidence."""

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


RUN_ID = "gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
RUN_STATUS = "PASS_GSE_RISK_CALIBRATED_PERCEPTION_VALIDATION_V2"
CORRECTIVE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
FIGURE_ID = "gse_slope_risk_calibration"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_files(run_dir: Path, expected_status: str) -> tuple[Path, dict[str, str]]:
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != expected_status or summary.get("overall_status") != expected_status:
        raise RuntimeError(f"paper source is not a completed PASS: {run_dir}")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    records = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        source = PROJECT_ROOT / relative
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError(f"paper source seal drift: {relative}")
        records[relative] = expected
    actual = {
        str(path.relative_to(PROJECT_ROOT))
        for path in run_dir.rglob("*")
        if path.is_file() and path != seal
    }
    if set(records) != actual:
        raise RuntimeError(f"paper source seal coverage drift: {run_dir}")
    return seal, records


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _selection_curve(rows: list[dict[str, np.ndarray]]) -> list[dict[str, float]]:
    parents = sorted(set(np.asarray(rows[0]["parent_id"]).astype(str).tolist()))
    if len(rows) != 3 or len(parents) != 20:
        raise RuntimeError("risk figure requires three seeds and twenty C07-C08 parents")
    curve = []
    for index in range(101):
        scale = index / 100.0
        parent_values = []
        for parent in parents:
            repeated = []
            for row in rows:
                parent_id = np.asarray(row["parent_id"]).astype(str)
                mask = parent_id == parent
                target = row["target_slope_deg"][mask]
                prior = row["five_frame_prior_slope_deg"][mask]
                raw = row["corrected_slope_deg"][mask]
                baseline = float(np.mean(np.abs(prior - target)))
                calibrated = float(np.mean(np.abs(prior + scale * (raw - prior) - target)))
                repeated.append((baseline - calibrated) / baseline)
            parent_values.append(float(np.mean(repeated)))
        curve.append(
            {
                "residual_scale": scale,
                "worst_parent_relative_improvement": min(parent_values),
                "mean_parent_relative_improvement": float(np.mean(parent_values)),
            }
        )
    return curve


def _world_rows(raw_summary: dict, calibrated_summary: dict) -> list[dict[str, object]]:
    output = []
    for parent in sorted(row["parent_id"] for row in raw_summary["seeds"][0]["per_world"]):
        raw = [next(item for item in seed["per_world"] if item["parent_id"] == parent) for seed in raw_summary["seeds"]]
        calibrated = [next(item for item in seed["per_world"] if item["parent_id"] == parent) for seed in calibrated_summary["seeds"]]
        prior = float(np.mean([item["five_frame_prior_mae_deg"] for item in raw]))
        raw_mae = float(np.mean([item["corrected_mae_deg"] for item in raw]))
        calibrated_mae = float(np.mean([item["corrected_mae_deg"] for item in calibrated]))
        output.append(
            {
                "parent_id": parent,
                "five_frame_prior_mae_deg": prior,
                "raw_full_residual_mae_deg": raw_mae,
                "risk_calibrated_mae_deg": calibrated_mae,
                "raw_relative_improvement": (prior - raw_mae) / prior,
                "risk_calibrated_relative_improvement": (prior - calibrated_mae) / prior,
            }
        )
    if len(output) != 10:
        raise RuntimeError("risk figure requires all ten C09 parents")
    return output


def publish(run_dir: Path, destination: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(PROJECT_ROOT)
    destination.relative_to(PROJECT_ROOT)
    if run_dir.name != RUN_ID:
        raise RuntimeError("risk figure source run identity drift")
    run_seal, run_records = _sealed_files(run_dir, RUN_STATUS)
    corrective_seal, corrective_records = _sealed_files(CORRECTIVE, "PASS_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1R")
    summary = load_json(run_dir / "metrics/summary.json")
    calibration = load_json(run_dir / "artifacts/risk_calibrated_slope_c09/risk_calibration.json")
    raw_summary = load_json(run_dir / "artifacts/raw_full_residual_c09/summary.json")
    calibrated_summary = load_json(run_dir / "artifacts/risk_calibrated_slope_c09/summary.json")
    if (
        summary.get("c10_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
        or summary.get("model_updates") != 0
        or calibration.get("c09_worlds_read_during_selection") != 0
        or calibration.get("residual_scale") != 0.89
    ):
        raise RuntimeError("risk figure isolation or selected-scale contract drift")
    selection_paths = [CORRECTIVE / f"artifacts/models/seed{seed}/selection_outputs.npz" for seed in range(3)]
    sources = [
        run_dir / "RUN_STATE.json",
        run_dir / "metrics/summary.json",
        run_dir / "metrics/corrected_perception_gate.json",
        run_dir / "artifacts/risk_calibrated_slope_c09/risk_calibration.json",
        run_dir / "artifacts/raw_full_residual_c09/summary.json",
        run_dir / "artifacts/risk_calibrated_slope_c09/summary.json",
        *selection_paths,
    ]
    for path in sources:
        relative = str(path.relative_to(PROJECT_ROOT))
        expected = run_records.get(relative, corrective_records.get(relative))
        if expected != _sha256(path):
            raise RuntimeError(f"risk figure source is not sealed: {relative}")

    curve = _selection_curve([_load_npz(path) for path in selection_paths])
    selected = max(curve, key=lambda row: (row["worst_parent_relative_improvement"], row["mean_parent_relative_improvement"], -row["residual_scale"]))
    if selected["residual_scale"] != calibration["residual_scale"]:
        raise RuntimeError("published calibration curve does not reproduce the formal selected scale")
    worlds = _world_rows(raw_summary, calibrated_summary)
    suffixes = (".png", ".pdf", ".svg", ".csv", "_source.json", "_provenance.json", "_sha256.txt")
    targets = [destination / f"{FIGURE_ID}{suffix}" for suffix in suffixes]
    if any(path.exists() for path in targets):
        raise RuntimeError("risk-calibration figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("section", "identity", "residual_scale", "worst_parent_relative_improvement", "mean_parent_relative_improvement", "raw_relative_improvement", "risk_calibrated_relative_improvement"))
        for row in curve:
            writer.writerow(("selection_curve", "C07-C08", row["residual_scale"], row["worst_parent_relative_improvement"], row["mean_parent_relative_improvement"], "", ""))
        for row in worlds:
            writer.writerow(("c09_world", row["parent_id"], calibration["residual_scale"], "", "", row["raw_relative_improvement"], row["risk_calibrated_relative_improvement"]))

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.75), constrained_layout=True)
    scale = np.asarray([row["residual_scale"] for row in curve])
    worst = np.asarray([row["worst_parent_relative_improvement"] for row in curve]) * 100.0
    mean = np.asarray([row["mean_parent_relative_improvement"] for row in curve]) * 100.0
    axes[0].plot(scale, mean, color="#376996", label="Mean parent gain")
    axes[0].plot(scale, worst, color="#E49B32", label="Worst parent gain")
    axes[0].axvline(calibration["residual_scale"], color="#17212B", linestyle="--", label="Selected 0.89")
    axes[0].set_xlabel("Learned residual scale")
    axes[0].set_ylabel("C07–C08 relative MAE improvement (%)")
    axes[0].set_title("Selection uses development worlds only")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(10)
    width = 0.38
    raw = np.asarray([row["raw_relative_improvement"] for row in worlds]) * 100.0
    risk = np.asarray([row["risk_calibrated_relative_improvement"] for row in worlds]) * 100.0
    axes[1].bar(x - width / 2, raw, width, color="#9AA5B1", label="Full residual")
    axes[1].bar(x + width / 2, risk, width, color="#2A9D8F", label="Risk-calibrated")
    axes[1].axhline(-5.0, color="#E76F51", linestyle=":", linewidth=1.4, label="−5% safety limit")
    axes[1].set_xticks(x, [f"S{index:02d}" for index in range(1, 11)], rotation=35)
    axes[1].set_ylabel("C09 relative slope-MAE improvement (%)")
    axes[1].set_title("All unseen development topologies")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(destination / f"{FIGURE_ID}_source.json", {"schema_version": "gse_slope_risk_calibration_figure_source_v2", "selection_curve": curve, "c09_worlds": worlds, "selected": calibration})
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v2",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(PROJECT_ROOT)),
            "source_run_seal_sha256": _sha256(run_seal),
            "corrective_training_seal_sha256": _sha256(corrective_seal),
            "selection_rule": calibration["selection_rule"],
            "c09_used_for_scale_selection": False,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sources},
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text("".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in sorted(targets) if path != manifest), encoding="utf-8")
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
