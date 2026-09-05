#!/usr/bin/env python3
"""Publish a paper-ready overview from the sealed GSE dataset export."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
EVENT_LABELS = ("Corridor", "Junction", "Terminal", "Turn", "Geometry\ntransition")
FORBIDDEN_DUPLICATED_BYTES = 61_225_344_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    if run_dir.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected GSE dataset source run")
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    runner = load_json(run_dir / "metrics/runner_summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or runner.get("overall_status") != EXPECTED_STATUS
    ):
        raise RuntimeError("GSE dataset source is not a completed PASS")
    totals = summary["global_totals"]
    if (
        summary["split_totals"]["train"] != {
            "worlds": 80,
            "frames": 252430,
            "sequences": 188126,
            "references": 940630,
        }
        or summary["split_totals"]["validation"] != {
            "worlds": 10,
            "frames": 32678,
            "sequences": 24462,
            "references": 122310,
        }
        or totals.get("candidates") != 448338
        or totals.get("visible_tokens") != 448279
        or totals.get("visible_width_valid") != 438968
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("sealed GSE dataset counts drifted")

    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    source_paths = (
        run_dir / "metrics/summary.json",
        run_dir / "metrics/runner_summary.json",
        run_dir / "artifacts/shard_manifest.json",
        run_dir / "artifacts/world_dataset_summary.jsonl",
    )
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"source is missing from the seal or drifted: {relative}")

    names = (
        "gse_dataset_overview.png",
        "gse_dataset_overview.pdf",
        "gse_dataset_overview.svg",
        "gse_dataset_overview.csv",
        "gse_dataset_overview_summary.json",
        "gse_dataset_overview_provenance.json",
        "gse_dataset_overview_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("paper figure destination already exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    event_counts: Counter[int] = Counter()
    geometry_valid = 0
    association_valid = 0
    sequence_count = 0
    for shard in sorted((run_dir / "artifacts/dataset").glob("*/*.zarr")):
        group = zarr.open_group(str(shard), mode="r")
        events = np.asarray(group["event_index"][:], dtype=np.int64)
        event_counts.update(int(value) for value in events)
        geometry_valid += int(np.asarray(group["geometry_valid_mask"][:, 0]).sum())
        association_valid += int(np.asarray(group["association_valid_mask"][:]).sum())
        sequence_count += len(events)
    if sequence_count != 212588 or [event_counts[index] for index in range(5)] != [
        152574,
        29824,
        8394,
        2313,
        19483,
    ]:
        raise RuntimeError("dataset event arrays do not reproduce the sealed Teacher distribution")

    metrics = [
        ("geometry_width_height", geometry_valid, sequence_count),
        ("place_association", association_valid, sequence_count),
        ("exit_los_visibility", int(totals["visible_tokens"]), int(totals["candidates"])),
        ("exit_width", int(totals["visible_width_valid"]), int(totals["visible_tokens"])),
    ]
    csv_path = destination / "gse_dataset_overview.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record_type", "name", "numerator", "denominator", "fraction"))
        for index, label in enumerate(EVENT_LABELS):
            writer.writerow(("event", label.replace("\n", " "), event_counts[index], sequence_count, event_counts[index] / sequence_count))
        for name, numerator, denominator in metrics:
            writer.writerow(("supervision", name, numerator, denominator, numerator / denominator))
        for name, value in (
            ("duplicated_five_frame_raw", FORBIDDEN_DUPLICATED_BYTES),
            ("deduplicated_raw_arrays", int(totals["uncompressed_array_bytes"])),
            ("deduplicated_zarr_shards", int(totals["shard_bytes"])),
        ):
            writer.writerow(("storage", name, value, FORBIDDEN_DUPLICATED_BYTES, value / FORBIDDEN_DUPLICATED_BYTES))

    plt.rcParams.update({"font.size": 8.6, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.45), constrained_layout=True)
    colors = ("#376996", "#2A9D8F", "#E9C46A", "#E76F51", "#7B6D8D")
    events = np.asarray([event_counts[index] for index in range(5)])
    y = np.arange(5)
    axes[0].barh(y, events / 1000.0, color=colors)
    axes[0].set_yticks(y, EVENT_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Causal sequences (thousands)")
    axes[0].set_title("All five structure events are represented")
    axes[0].grid(axis="x", alpha=0.22)

    storage = np.asarray(
        [FORBIDDEN_DUPLICATED_BYTES, totals["uncompressed_array_bytes"], totals["shard_bytes"]],
        dtype=np.float64,
    ) / 1024**3
    bars = axes[1].bar(
        np.arange(3),
        storage,
        color=("#B8B8B8", "#376996", "#2A9D8F"),
        width=0.65,
    )
    axes[1].set_xticks(np.arange(3), ("Repeated\n5-frame", "Unique raw\narrays", "Stored\nZarr"))
    axes[1].set_ylabel("Payload (GiB)")
    axes[1].set_title("Unique-frame storage removes duplication")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, storage, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}", ha="center")

    rates = 100.0 * np.asarray([numerator / denominator for _, numerator, denominator in metrics])
    labels = ("Width/height", "Place identity", "Exit LOS", "Exit width")
    bars = axes[2].bar(np.arange(4), rates, color=("#376996", "#7B6D8D", "#2A9D8F", "#E9C46A"), width=0.65)
    axes[2].set_xticks(np.arange(4), labels, rotation=18, ha="right")
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Valid supervision (%)")
    axes[2].set_title("Missing labels remain explicit masks")
    axes[2].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, rates, strict=True):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.1f}", ha="center", fontsize=8)

    figure.suptitle(
        "GSE-Graph development dataset — 90 worlds, 285,108 unique LiDAR frames, 212,588 causal sequences",
        fontsize=11,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"gse_dataset_overview.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure_summary = {
        "event_counts": {EVENT_LABELS[index].replace("\n", " "): event_counts[index] for index in range(5)},
        "supervision": {name: {"valid": numerator, "total": denominator, "fraction": numerator / denominator} for name, numerator, denominator in metrics},
        "storage_bytes": {
            "forbidden_duplicated_five_frame": FORBIDDEN_DUPLICATED_BYTES,
            "deduplicated_raw_arrays": int(totals["uncompressed_array_bytes"]),
            "deduplicated_zarr_shards": int(totals["shard_bytes"]),
        },
    }
    write_json(destination / "gse_dataset_overview_summary.json", figure_summary)
    write_json(
        destination / "gse_dataset_overview_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": "gse_dataset_overview",
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all 80 train and 10 validation development worlds; no sample or outcome selection",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in source_paths},
        },
    )
    manifest = destination / "gse_dataset_overview_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {
        "figure_id": "gse_dataset_overview",
        "published_files": len(targets),
        "manifest_sha256": _sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
