#!/usr/bin/env python3
"""Generate the paper method overview as PNG, PDF and editable SVG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


FIGURE_ID = "gse_method_overview"
LEGACY_MANIFEST_SHA256 = "3524d9e4f91caaf1a0f210e20a201c1236c1ffce799ab10c866f956d01ce35a9"

METHOD_SOURCE = {
    "schema_version": "gse_method_diagram_source_v1",
    "coordinate_system": {"xlim": [0.0, 13.2], "ylim": [0.0, 5.2]},
    "nodes": [
        {"id": "causal_lidar", "kind": "pipeline", "xy": [0.25, 2.55], "width": 2.05, "height": 1.35, "title": "5-frame causal LiDAR", "detail": "range + valid mask\nno pose or GT input", "color": "#376996"},
        {"id": "geometry_encoder", "kind": "pipeline", "xy": [2.80, 2.55], "width": 2.05, "height": 1.35, "title": "Geometry encoder", "detail": "circular range image\ncausal temporal fusion", "color": "#376996"},
        {"id": "semantic_observation", "kind": "pipeline", "xy": [5.35, 2.55], "width": 2.05, "height": 1.35, "title": "Semantic observation", "detail": "event · axis · width/height\nslope · curvature · exit tokens", "color": "#2A9D8F"},
        {"id": "online_graph", "kind": "pipeline", "xy": [7.90, 2.55], "width": 2.05, "height": 1.35, "title": "Online graph update", "detail": "event-triggered nodes\nuncertain matches stay provisional", "color": "#7B6D8D"},
        {"id": "exploration", "kind": "pipeline", "xy": [10.45, 2.55], "width": 2.05, "height": 1.35, "title": "Exploration interface", "detail": "untraversed exits + verified edges\nunchanged M-TARE local planner", "color": "#3A7D44"},
        {"id": "learning_supervision", "kind": "support", "xy": [2.80, 0.48], "width": 4.60, "height": 1.15, "title": "Learning supervision", "detail": "procedural TNG event identity · native-mesh cross-section\nspline axis/slope/curvature · revisit and hard-negative identities", "color": "#E49B32"},
        {"id": "reliability_contract", "kind": "support", "xy": [8.00, 0.48], "width": 4.50, "height": 1.15, "title": "Reliability contract", "detail": "validation-calibrated confidence · ≥0.98 merge precision\nambiguous loop closure rejected · edge only after physical traversal", "color": "#7B6D8D"},
    ],
    "edges": [
        {"source": "causal_lidar", "target": "geometry_encoder", "relation": "causal_input"},
        {"source": "geometry_encoder", "target": "semantic_observation", "relation": "typed_prediction"},
        {"source": "semantic_observation", "target": "online_graph", "relation": "event_and_association_update"},
        {"source": "online_graph", "target": "exploration", "relation": "sparse_graph_interface"},
        {"source": "learning_supervision", "target": "geometry_encoder", "relation": "training_only"},
        {"source": "reliability_contract", "target": "online_graph", "relation": "validated_update_contract"},
    ],
    "claims": {
        "learned_component_output": "typed and interpretable GeometricSemanticObservation",
        "edge_creation": "physical traversal evidence only",
        "unchanged_component": "M-TARE local planner, obstacle avoidance, and control",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(axis, xy, width, height, title, detail, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.25,
        edgecolor=color,
        facecolor=color + "18",
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height * 0.68, title, ha="center", va="center", fontsize=10.2, weight="bold", color="#17212B")
    axis.text(xy[0] + width / 2, xy[1] + height * 0.31, detail, ha="center", va="center", fontsize=7.8, color="#34495E", linespacing=1.25)


def _arrow(axis, start, end, color="#5B6570"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=color))


def publish(destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    destination = destination.resolve()
    destination.relative_to(root)
    names = (
        f"{FIGURE_ID}.png",
        f"{FIGURE_ID}.pdf",
        f"{FIGURE_ID}.svg",
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("method figure destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(13.2, 5.2))
    axis.set_xlim(*METHOD_SOURCE["coordinate_system"]["xlim"])
    axis.set_ylim(*METHOD_SOURCE["coordinate_system"]["ylim"])
    axis.axis("off")
    nodes = {node["id"]: node for node in METHOD_SOURCE["nodes"]}
    for node in METHOD_SOURCE["nodes"]:
        _box(axis, tuple(node["xy"]), node["width"], node["height"], node["title"], node["detail"], node["color"])
    for edge in METHOD_SOURCE["edges"][:4]:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        _arrow(
            axis,
            (source["xy"][0] + source["width"] + 0.06, source["xy"][1] + source["height"] / 2),
            (target["xy"][0] - 0.08, target["xy"][1] + target["height"] / 2),
        )

    axis.text(0.25, 4.55, "GSE-Graph: learned geometry semantics directly determine sparse online topology", fontsize=15, weight="bold", color="#17212B")
    axis.text(0.25, 4.20, "The learned component ends at a typed, interpretable observation; graph edges require real traversal evidence.", fontsize=9.4, color="#4B5563")

    _arrow(axis, (4.85, 1.66), (4.05, 2.48), nodes["learning_supervision"]["color"])
    _arrow(axis, (10.25, 1.66), (9.55, 2.48), nodes["reliability_contract"]["color"])

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    source_path = destination / f"{FIGURE_ID}_source.json"
    write_json(source_path, METHOD_SOURCE)
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "kind": "method diagram; no experimental values",
            "method_contract": "docs/GSE_GRAPH_RESEARCH_PLAN_V1.md",
            "method_contract_sha256": _sha256(root / "docs/GSE_GRAPH_RESEARCH_PLAN_V1.md"),
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "machine_readable_source": str(source_path.relative_to(root)),
            "machine_readable_source_sha256": _sha256(source_path),
            "manual_value_entry": False,
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def repair_missing_source(destination: Path) -> dict:
    """Complete the legacy method bundle without changing its rendered figure files."""
    root = PROJECT_ROOT.resolve()
    destination = destination.resolve()
    destination.relative_to(root)
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    source_path = destination / f"{FIGURE_ID}_source.json"
    provenance_path = destination / f"{FIGURE_ID}_provenance.json"
    if source_path.exists() or _sha256(manifest) != LEGACY_MANIFEST_SHA256:
        raise RuntimeError("legacy method bundle is not the exact repair target")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"legacy method bundle drift: {relative}")
    write_json(source_path, METHOD_SOURCE)
    write_json(
        provenance_path,
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "kind": "method diagram; no experimental values",
            "method_contract": "docs/GSE_GRAPH_RESEARCH_PLAN_V1.md",
            "method_contract_sha256": _sha256(root / "docs/GSE_GRAPH_RESEARCH_PLAN_V1.md"),
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "machine_readable_source": str(source_path.relative_to(root)),
            "machine_readable_source_sha256": _sha256(source_path),
            "manual_value_entry": False,
            "repair": "added the missing structured method source; rendered PNG/PDF/SVG were hash-verified and unchanged",
            "supersedes_manifest_sha256": LEGACY_MANIFEST_SHA256,
        },
    )
    retained = [
        destination / f"{FIGURE_ID}.png",
        destination / f"{FIGURE_ID}.pdf",
        destination / f"{FIGURE_ID}.svg",
        source_path,
        provenance_path,
    ]
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(retained)),
        encoding="utf-8",
    )
    return {
        "figure_id": FIGURE_ID,
        "repaired": True,
        "rendered_files_changed": False,
        "manifest_sha256": _sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    parser.add_argument("--repair-missing-source", action="store_true")
    args = parser.parse_args()
    result = repair_missing_source(args.destination) if args.repair_missing_source else publish(args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
