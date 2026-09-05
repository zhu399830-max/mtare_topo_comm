#!/usr/bin/env python3
"""Append the missing vector/provenance evidence to the published Teacher figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


FIGURE_ID = "gse_teacher_distribution"
SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_mesh_teacher_distribution_v1r_seed0"
SOURCE_STATUS = "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1R"
EVENT_NAMES = ("corridor", "junction", "terminal", "turn", "geometry_transition")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(path: Path, root: Path) -> int:
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        source = root / relative
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError(f"published figure bundle drift: {relative}")
        checked += 1
    return checked


def _render_svg(summary: dict, destination: Path) -> None:
    counts = summary["event_counts"]
    missing = summary["missing_continuous_geometry_by_event"]
    labels = ["corridor", "junction", "terminal", "turn", "geometry\ntransition"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].bar(labels, [counts[name] for name in EVENT_NAMES], color=colors)
    axes[0].set_ylabel("five-frame training sequences")
    axes[0].set_title("GSE structural-event teacher distribution")
    axes[0].tick_params(axis="x", labelrotation=18)
    for index, name in enumerate(EVENT_NAMES):
        axes[0].text(index, counts[name], f"{counts[name]:,}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(labels, [missing[name] for name in EVENT_NAMES], color=colors)
    axes[1].set_ylabel("masked width/height targets")
    axes[1].set_title("Non-unique or incomplete mesh cross-sections")
    axes[1].tick_params(axis="x", labelrotation=18)
    for index, name in enumerate(EVENT_NAMES):
        axes[1].text(index, missing[name], f"{missing[name]:,}", ha="center", va="bottom", fontsize=8)
    figure.suptitle(
        f"Train-only native-mesh teacher proof — geometry valid {summary['geometry_valid_fraction']:.2%}"
    )
    figure.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def repair(destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    destination = destination.resolve()
    destination.relative_to(root)
    state = load_json(SOURCE_RUN / "RUN_STATE.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != SOURCE_STATUS:
        raise RuntimeError("Teacher distribution source is not the required sealed PASS")
    source_seal = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    _verify_manifest(source_seal, root)
    old_manifest = destination / f"{FIGURE_ID}_sha256.txt"
    checked = _verify_manifest(old_manifest, root)
    svg = destination / f"{FIGURE_ID}.svg"
    if svg.exists():
        raise RuntimeError("Teacher distribution SVG already exists; refusing overwrite")
    summary_path = destination / f"{FIGURE_ID}_summary.json"
    source_summary = SOURCE_RUN / "metrics/summary.json"
    if _sha256(summary_path) != _sha256(source_summary):
        raise RuntimeError("published summary no longer matches sealed source")
    _render_svg(load_json(summary_path), svg)

    provenance_path = destination / f"{FIGURE_ID}_provenance.json"
    provenance = load_json(provenance_path)
    source_generator = root / provenance["generator"]
    provenance.update(
        {
            "generator_sha256": _sha256(source_generator),
            "vector_svg_added": True,
            "bundle_correction": "append missing SVG and generator identity; experimental values unchanged",
            "bundle_correction_tool": str(Path(__file__).resolve().relative_to(root)),
            "bundle_correction_tool_sha256": _sha256(Path(__file__).resolve()),
            "pre_correction_manifest_sha256": _sha256(old_manifest),
        }
    )
    write_json(provenance_path, provenance)
    files = sorted(
        path
        for path in destination.glob(f"{FIGURE_ID}*")
        if path.is_file() and path != old_manifest
    )
    old_manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in files),
        encoding="utf-8",
    )
    return {
        "figure_id": FIGURE_ID,
        "pre_correction_entries_verified": checked,
        "published_files": len(files) + 1,
        "svg_sha256": _sha256(svg),
        "manifest_sha256": _sha256(old_manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/gse_graph",
    )
    args = parser.parse_args()
    print(json.dumps(repair(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
