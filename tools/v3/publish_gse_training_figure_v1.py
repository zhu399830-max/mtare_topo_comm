#!/usr/bin/env python3
"""Publish paper-ready GSE learning curves from the sealed three-seed run."""

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


EXPECTED_RUN_ID = "gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
EXPECTED_STATUS = "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
FIGURE_ID = "gse_training_curves"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sealed_files(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (run_dir / "artifacts/evidence_sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        result[relative] = expected
    return result


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    if run_dir.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected GSE training source run")
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
        or len(summary.get("seeds", ())) != 3
    ):
        raise RuntimeError("GSE training source is not a completed three-seed PASS")

    sealed = _sealed_files(run_dir)
    source_paths = [run_dir / "metrics/summary.json", run_dir / "RUN_STATE.json"]
    histories: dict[int, list[dict]] = {}
    for seed in (0, 1, 2):
        path = run_dir / f"artifacts/models/seed{seed}/epoch_metrics.jsonl"
        metrics_path = run_dir / f"artifacts/models/seed{seed}/best_validation_metrics.json"
        child_summary = run_dir / f"artifacts/models/seed{seed}/summary.json"
        source_paths.extend((path, metrics_path, child_summary))
        records = _read_jsonl(path)
        if not records or [int(record["epoch"]) for record in records] != list(range(1, len(records) + 1)):
            raise RuntimeError(f"seed {seed} epoch history is incomplete or noncontiguous")
        seed_summary = load_json(child_summary)
        if (
            seed_summary.get("seed") != seed
            or seed_summary.get("epochs_completed") != len(records)
            or seed_summary.get("strict_test_worlds_read") != 0
            or seed_summary.get("mtare_worlds_read") != 0
        ):
            raise RuntimeError(f"seed {seed} summary does not match its learning curve")
        histories[seed] = records
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"paper source is absent from the seal or drifted: {relative}")

    names = (
        f"{FIGURE_ID}.png",
        f"{FIGURE_ID}.pdf",
        f"{FIGURE_ID}.svg",
        f"{FIGURE_ID}.csv",
        f"{FIGURE_ID}_summary.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("paper figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "seed",
                "epoch",
                "train_total_loss",
                "validation_total_loss",
                "event_macro_f1",
                "axis_mean_error_deg",
                "association_top1_precision",
                "width_mae_m",
                "height_mae_m",
                "slope_mae_deg",
                "curvature_mae_per_m",
            )
        )
        for seed, records in histories.items():
            for record in records:
                validation = record["validation"]
                geometry = validation["geometry_mae"]
                writer.writerow(
                    (
                        seed,
                        record["epoch"],
                        record["train_loss"]["total"],
                        validation["loss"]["total"],
                        validation["event"]["macro_f1"],
                        validation["axis"]["mean_angular_error_deg"],
                        validation["association_retrieval"]["top1_same_identity_precision"],
                        geometry["width_m"],
                        geometry["height_m"],
                        geometry["slope_deg"],
                        geometry["curvature_per_m"],
                    )
                )

    colors = ("#376996", "#E76F51", "#2A9D8F")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.1), constrained_layout=True)
    panels = (
        ("validation.loss.total", "Validation multitask loss", "Loss"),
        ("validation.event.macro_f1", "Five-class structural event recognition", "Macro-F1"),
        ("validation.axis.mean_angular_error_deg", "Local structure-axis estimation", "Mean error (deg)"),
        (
            "validation.association_retrieval.top1_same_identity_precision",
            "Place-descriptor retrieval",
            "Top-1 identity precision",
        ),
    )

    def nested(record: dict, dotted: str) -> float:
        value: object = record
        for name in dotted.split("."):
            value = value[name]  # type: ignore[index]
        return float(value)

    for axis, (field, title, ylabel) in zip(axes.flat, panels, strict=True):
        for seed, color in zip((0, 1, 2), colors, strict=True):
            records = histories[seed]
            epochs = [int(record["epoch"]) for record in records]
            values = [nested(record, field) for record in records]
            axis.plot(epochs, values, marker="o", markersize=2.6, linewidth=1.35, color=color, label=f"seed {seed}")
            best_epoch = int(summary["seeds"][seed]["best_epoch"])
            axis.scatter(
                [best_epoch],
                [values[best_epoch - 1]],
                marker="*",
                s=65,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.suptitle("GSE-Graph training on 80 worlds; checkpoint selection on 10 validation worlds", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    best = []
    for item in summary["seeds"]:
        validation = item["best_validation"]
        best.append(
            {
                "seed": int(item["seed"]),
                "best_epoch": int(item["best_epoch"]),
                "validation_total_loss": float(validation["loss"]["total"]),
                "event_macro_f1": float(validation["event"]["macro_f1"]),
                "axis_mean_error_deg": float(validation["axis"]["mean_angular_error_deg"]),
                "association_top1_precision": float(validation["association_retrieval"]["top1_same_identity_precision"]),
                "geometry_mae": validation["geometry_mae"],
            }
        )
    aggregate = {}
    for field in ("validation_total_loss", "event_macro_f1", "axis_mean_error_deg", "association_top1_precision"):
        values = np.asarray([record[field] for record in best], dtype=np.float64)
        aggregate[field] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    write_json(destination / f"{FIGURE_ID}_summary.json", {"seeds": best, "aggregate": aggregate})
    seal = run_dir / "artifacts/evidence_sha256.txt"
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all epochs from all three predeclared seeds; stars mark validation-loss-selected checkpoints",
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
