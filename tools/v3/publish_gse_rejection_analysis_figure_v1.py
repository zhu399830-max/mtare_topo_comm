#!/usr/bin/env python3
"""Publish GSE validation rejection/association behavior from sealed evidence."""

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
FIGURE_ID = "gse_rejection_analysis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise RuntimeError(f"empty rejection-analysis source: {path.name}")
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
        raise RuntimeError("rejection analysis requires the sealed perception PASS")

    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    calibration = run_dir / "artifacts/calibration"
    source_paths = [run_dir / "RUN_STATE.json", run_dir / "metrics/summary.json", run_dir / "metrics/perception_gate.json"]
    seed_data: dict[int, dict] = {}
    for seed in (0, 1, 2):
        seed_summary_path = calibration / f"seed{seed}_summary.json"
        event_path = calibration / f"seed{seed}_event_rejection_curve.jsonl"
        place_path = calibration / f"seed{seed}_place_association_curve.jsonl"
        exit_path = calibration / f"seed{seed}_exit_descriptor_curve.jsonl"
        paths = (seed_summary_path, event_path, place_path, exit_path)
        source_paths.extend(paths)
        seed_summary = load_json(seed_summary_path)
        if (
            seed_summary.get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1"
            or seed_summary.get("strict_test_worlds_read") != 0
            or seed_summary.get("mtare_worlds_read") != 0
        ):
            raise RuntimeError(f"seed {seed} calibration is not a clean PASS")
        seed_data[seed] = {
            "summary": seed_summary,
            "event": _jsonl(event_path),
            "place": _jsonl(place_path),
            "exit": _jsonl(exit_path),
        }
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"rejection-analysis source is absent from the run seal: {relative}")

    names = tuple(f"{FIGURE_ID}.{suffix}" for suffix in ("png", "pdf", "svg", "csv")) + (
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("rejection-analysis destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "section", "threshold", "accepted", "precision_or_f1", "recall", "false_accept_rate"))
        for seed, data in seed_data.items():
            for row in data["event"]:
                writer.writerow((seed, "event", row["threshold"], row["accepted_structural_events"], row["macro_f1"], row["structural_recall"], ""))
            for section in ("place", "exit"):
                for row in data[section]:
                    writer.writerow((seed, section, row["threshold"], row["accepted"], row["precision"], row["recall"], row["false_accept_rate"]))

    colors = ("#376996", "#E76F51", "#2A9D8F")
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True)
    for seed, color in zip((0, 1, 2), colors, strict=True):
        data = seed_data[seed]
        event = sorted(data["event"], key=lambda row: int(row["accepted_structural_events"]))
        axes[0, 0].plot([row["accepted_structural_events"] for row in event], [row["macro_f1"] for row in event], color=color, linewidth=1.25, label=f"seed {seed}")
        selected_event = data["summary"]["event"]["rejection_selection"]
        axes[0, 0].scatter(selected_event["accepted_structural_events"], selected_event["macro_f1"], marker="*", s=65, color=color, edgecolor="white", linewidth=0.4, zorder=5)
        for axis, section, selection_path in (
            (axes[0, 1], "place", ("place_association", "selection")),
            (axes[1, 0], "exit", ("exit_tokens", "descriptor_association", "selection")),
        ):
            curve = sorted(data[section], key=lambda row: int(row["accepted"]))
            axis.plot([row["accepted"] for row in curve], [row["precision"] for row in curve], color=color, linewidth=1.15)
            selected = data["summary"]
            for key in selection_path:
                selected = selected[key]
            axis.scatter(selected["accepted"], selected["precision"], marker="*", s=65, color=color, edgecolor="white", linewidth=0.4, zorder=5)

    axes[0, 0].set_title("Event rejection")
    axes[0, 0].set_xlabel("Accepted structural predictions")
    axes[0, 0].set_ylabel("Five-event macro-F1")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis, title in ((axes[0, 1], "Place association"), (axes[1, 0], "Exit association")):
        axis.axhline(0.98, color="#17212B", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Accepted causal matches")
        axis.set_ylabel("Precision")
        axis.set_ylim(0.94, 1.002)

    seeds = np.arange(3)
    width = 0.34
    place_far = [seed_data[seed]["summary"]["place_association"]["selection"]["false_accept_rate"] for seed in (0, 1, 2)]
    exit_far = [seed_data[seed]["summary"]["exit_tokens"]["descriptor_association"]["selection"]["false_accept_rate"] for seed in (0, 1, 2)]
    axes[1, 1].bar(seeds - width / 2, np.asarray(place_far) * 100.0, width, color="#7B6D8D", label="Place")
    axes[1, 1].bar(seeds + width / 2, np.asarray(exit_far) * 100.0, width, color="#E49B32", label="Exit")
    axes[1, 1].axhline(1.0, color="#17212B", linestyle="--", linewidth=1.0, label="1% limit")
    axes[1, 1].set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axes[1, 1].set_ylabel("False accepts among merges (%)")
    axes[1, 1].set_title("Selected association risk")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("GSE-Graph validation-only rejection and causal association", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    compact_source = {
        "schema_version": "gse_rejection_analysis_source_v1",
        "seeds": {
            str(seed): {
                "event_selection": data["summary"]["event"]["rejection_selection"],
                "place_selection": data["summary"]["place_association"]["selection"],
                "exit_selection": data["summary"]["exit_tokens"]["descriptor_association"]["selection"],
                "event_curve_records": len(data["event"]),
                "place_curve_records": len(data["place"]),
                "exit_curve_records": len(data["exit"]),
            }
            for seed, data in seed_data.items()
        },
    }
    write_json(destination / f"{FIGURE_ID}_source.json", compact_source)
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "complete validation curves for all three seeds; stars are the frozen selected operating points",
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
