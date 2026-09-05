#!/usr/bin/env python3
"""Render a paper figure only from the sealed PASS GSE exit-token audit."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate2_20260824_gse_exit_token_audit_v1r_seed0"
EXPECTED_STATUS = "PASS_GSE_EXIT_TOKEN_AUDIT_V1"
EVENTS = ("corridor", "junction", "terminal", "turn", "geometry_transition")
LABELS = ("Corridor", "Junction", "Terminal", "Turn", "Geometry\ntransition")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_entries(run_dir: Path) -> tuple[Path, dict[str, str]]:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    entries = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    return seal, entries


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    if run_dir.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected exit-token source run")
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    runner = load_json(run_dir / "metrics/runner_summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or runner.get("overall_status") != EXPECTED_STATUS
    ):
        raise RuntimeError("exit-token source is not a completed sealed PASS")
    totals = summary["totals"]
    if (
        totals.get("world_count") != 90
        or totals.get("observation_count") != 212588
        or totals.get("candidate_count") != 448338
        or totals.get("nonincident_candidate_count") != 0
        or totals.get("zero_visible_node_event_count") != 0
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("exit-token source contract mismatch")
    seal, entries = _seal_entries(run_dir)
    source_paths = (
        run_dir / "metrics/summary.json",
        run_dir / "artifacts/world_exit_token_summary.jsonl",
    )
    for source_path in source_paths:
        relative = str(source_path.relative_to(root))
        if entries.get(relative) != _sha256(source_path):
            raise RuntimeError("source evidence is absent from its seal or drifted")

    stems = (
        "gse_exit_token_audit.png",
        "gse_exit_token_audit.pdf",
        "gse_exit_token_audit.svg",
        "gse_exit_token_audit.csv",
        "gse_exit_token_audit_summary.json",
        "gse_exit_token_audit_provenance.json",
        "gse_exit_token_audit_sha256.txt",
    )
    targets = [destination / name for name in stems]
    if any(path.exists() for path in targets):
        raise RuntimeError("paper figure destination already exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    candidate = np.asarray(
        [sum(summary["candidate_counts"][split].get(event, 0) for split in ("train", "validation")) for event in EVENTS],
        dtype=np.float64,
    )
    visible = np.asarray(
        [sum(summary["visible_counts"][split].get(event, 0) for split in ("train", "validation")) for event in EVENTS],
        dtype=np.float64,
    )
    width_valid = np.asarray(
        [sum(summary["width_valid_counts"][split].get(event, 0) for split in ("train", "validation")) for event in EVENTS],
        dtype=np.float64,
    )
    event_observations: Counter[str] = Counter()
    world_rows = 0
    with (run_dir / "artifacts/world_exit_token_summary.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            world = json.loads(line)
            event_observations.update(world["event_observation_counts"])
            world_rows += 1
    if world_rows != 90 or sum(event_observations.values()) != 212588:
        raise RuntimeError("sealed per-world observation distribution is incomplete")
    observation_counts = np.asarray([event_observations[event] for event in EVENTS], dtype=np.float64)
    tokens_per_observation = candidate / observation_counts
    visible_rate = 100.0 * visible / candidate
    width_rate = 100.0 * width_valid / candidate

    csv_path = destination / "gse_exit_token_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("event", "observations", "candidates", "tokens_per_observation", "visible_fraction", "width_valid_fraction")
        )
        for index, event in enumerate(EVENTS):
            writer.writerow(
                (
                    event,
                    int(observation_counts[index]),
                    int(candidate[index]),
                    float(tokens_per_observation[index]),
                    float(visible[index] / candidate[index]),
                    float(width_valid[index] / candidate[index]),
                )
            )

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), constrained_layout=True)
    x = np.arange(len(EVENTS))
    axes[0].bar(x, tokens_per_observation, color="#376996", width=0.68)
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Directed exit tokens / observation")
    axes[0].set_ylim(0.0, max(4.0, float(tokens_per_observation.max()) + 0.35))
    axes[0].set_title("Physical structure controls token cardinality")
    axes[0].grid(axis="y", alpha=0.22)
    for index, value in enumerate(tokens_per_observation):
        axes[0].text(index, value + 0.07, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    width = 0.36
    axes[1].bar(x - width / 2, visible_rate, width, label="LOS visible", color="#2A9D8F")
    axes[1].bar(x + width / 2, width_rate, width, label="Width valid", color="#E9C46A")
    axes[1].set_xticks(x, LABELS)
    axes[1].set_ylabel("Qualified candidates (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title("Native-mesh supervision remains explicitly masked")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="lower left")

    figure.suptitle(
        "Incident-only geometric exit teacher — 90 development worlds, 448,338 candidates",
        fontsize=11,
    )
    figure.text(
        0.5,
        -0.01,
        "0 nonincident candidates · 0 junction/terminal observations without a visible exit · different physical edges never collapse",
        ha="center",
        fontsize=8.5,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"gse_exit_token_audit.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)
    shutil.copyfile(run_dir / "metrics/summary.json", destination / "gse_exit_token_audit_summary.json")

    provenance_path = destination / "gse_exit_token_audit_provenance.json"
    write_json(
        provenance_path,
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": "gse_exit_token_audit",
            "source_run": str(run_dir.relative_to(root)),
            "source_run_id": EXPECTED_RUN_ID,
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all 80 train and 10 validation development worlds; no result-based selection",
            "manual_value_entry": False,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "generator": "tools/v3/publish_gse_exit_token_audit_figure_v1.py",
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {
                str(path.relative_to(run_dir)): _sha256(path) for path in source_paths
            },
        },
    )
    manifest_path = destination / "gse_exit_token_audit_sha256.txt"
    published = sorted(path for path in targets if path != manifest_path)
    manifest_path.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in published),
        encoding="utf-8",
    )
    return {
        "figure_id": "gse_exit_token_audit",
        "published_files": len(targets),
        "manifest_sha256": _sha256(manifest_path),
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
