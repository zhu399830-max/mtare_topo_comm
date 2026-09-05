#!/usr/bin/env python3
"""Publish the active primitive-relation structural-graph method diagram."""

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


FIGURE_ID = "primitive_relation_method_overview"

METHOD_SOURCE = {
    "schema_version": "primitive_relation_method_diagram_source_v1",
    "coordinate_system": {"xlim": [0.0, 15.2], "ylim": [0.0, 6.2]},
    "nodes": [
        {
            "id": "causal_input",
            "kind": "online",
            "xy": [0.25, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "5 causal\nLiDAR scans",
            "detail": "16 × 720 range + valid\nrelative odometry only",
            "color": "#386FA4",
        },
        {
            "id": "encoder",
            "kind": "online",
            "xy": [2.75, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "Causal geometry\nencoder",
            "detail": "circular range features\nregistered temporal fusion",
            "color": "#386FA4",
        },
        {
            "id": "primitive_set",
            "kind": "learned",
            "xy": [5.25, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "Swept primitive\nset",
            "detail": "axis · endpoints · section\nwidth/height · taper · curve",
            "color": "#2A9D8F",
        },
        {
            "id": "relation_set",
            "kind": "learned",
            "xy": [7.75, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "Physical relation\nset",
            "detail": "endpoint attachment\ndisconnected overlap\ntemporal ID · uncertainty",
            "color": "#2A9D8F",
        },
        {
            "id": "persistent_graph",
            "kind": "graph",
            "xy": [10.25, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "Persistent\nstructural graph",
            "detail": "nodes + ports from relations\nambiguous matches\nstay provisional",
            "color": "#7B6D8D",
        },
        {
            "id": "exploration",
            "kind": "execution",
            "xy": [12.75, 2.85],
            "width": 2.05,
            "height": 1.65,
            "title": "Traversal-verified\nexploration graph",
            "detail": "edge only after traversal\nunchanged M-TARE local stack",
            "color": "#3A7D44",
        },
        {
            "id": "construction_supervisor",
            "kind": "training_only",
            "xy": [2.75, 0.50],
            "width": 7.05,
            "height": 1.25,
            "title": "Training only: construction-program supervisor",
            "detail": "paired ellipse / rounded-rectangle / C1-mixed tunnels  →  ray provenance\nprimitive parameters  →  ports, overlap and temporal correspondence",
            "color": "#E49B32",
        },
        {
            "id": "reliability_contract",
            "kind": "contract",
            "xy": [10.25, 0.50],
            "width": 4.55,
            "height": 1.25,
            "title": "Online reliability contract",
            "detail": "no construction identity in forward pass\nreject uncertain merge · commit edge by execution",
            "color": "#7B6D8D",
        },
    ],
    "edges": [
        {"source": "causal_input", "target": "encoder", "relation": "causal_input"},
        {"source": "encoder", "target": "primitive_set", "relation": "set_decode"},
        {"source": "primitive_set", "target": "relation_set", "relation": "compose"},
        {"source": "relation_set", "target": "persistent_graph", "relation": "graph_update"},
        {"source": "persistent_graph", "target": "exploration", "relation": "traversal_commit"},
        {"source": "construction_supervisor", "target": "primitive_set", "relation": "training_only"},
        {"source": "construction_supervisor", "target": "relation_set", "relation": "training_only"},
        {"source": "reliability_contract", "target": "persistent_graph", "relation": "online_contract"},
    ],
    "claims": {
        "student_input": "five causal range images, valid masks, and relative odometry",
        "learned_output": "explicit swept tunnel primitives and physical composition relations",
        "graph_rule": "relations propose structure; physical traversal alone commits a global edge",
        "unchanged_stack": "M-TARE localization, local planning, avoidance, and control",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(axis, node: dict) -> None:
    x, y = node["xy"]
    width = node["width"]
    height = node["height"]
    color = node["color"]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            linewidth=1.35,
            edgecolor=color,
            facecolor=color + "18",
        )
    )
    axis.text(
        x + width / 2,
        y + height * 0.70,
        node["title"],
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color="#17212B",
    )
    axis.text(
        x + width / 2,
        y + height * 0.27,
        node["detail"],
        ha="center",
        va="center",
        fontsize=7.0,
        color="#34495E",
        linespacing=1.27,
    )


def _arrow(axis, start, end, color="#5B6570", style="-") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            linestyle=style,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


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
        raise RuntimeError("primitive-relation method figure exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(15.2, 6.2))
    axis.set_xlim(*METHOD_SOURCE["coordinate_system"]["xlim"])
    axis.set_ylim(*METHOD_SOURCE["coordinate_system"]["ylim"])
    axis.axis("off")
    nodes = {node["id"]: node for node in METHOD_SOURCE["nodes"]}
    for node in METHOD_SOURCE["nodes"]:
        _box(axis, node)

    online_ids = [
        "causal_input",
        "encoder",
        "primitive_set",
        "relation_set",
        "persistent_graph",
        "exploration",
    ]
    for source_id, target_id in zip(online_ids[:-1], online_ids[1:]):
        source = nodes[source_id]
        target = nodes[target_id]
        _arrow(
            axis,
            (source["xy"][0] + source["width"] + 0.05, source["xy"][1] + source["height"] / 2),
            (target["xy"][0] - 0.07, target["xy"][1] + target["height"] / 2),
        )

    _arrow(axis, (6.15, 1.78), (6.15, 2.78), nodes["construction_supervisor"]["color"], "--")
    _arrow(axis, (8.65, 1.78), (8.65, 2.78), nodes["construction_supervisor"]["color"], "--")
    _arrow(axis, (12.55, 1.78), (11.45, 2.78), nodes["reliability_contract"]["color"], "--")

    axis.text(
        0.25,
        5.55,
        "Primitive-Relation Structural Graph: physical composition learned before topology",
        fontsize=15.2,
        weight="bold",
        color="#17212B",
    )
    axis.text(
        0.25,
        5.15,
        "Construction identity supervises training only; online topology is inferred from partial causal LiDAR and verified by robot execution.",
        fontsize=9.6,
        color="#4B5563",
    )
    axis.text(0.25, 2.52, "ONLINE", fontsize=8.3, weight="bold", color="#4B5563")
    axis.text(0.25, 1.17, "SUPERVISION / CONTRACT", fontsize=8.3, weight="bold", color="#4B5563")

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    source_path = destination / f"{FIGURE_ID}_source.json"
    write_json(source_path, METHOD_SOURCE)
    contract = root / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md"
    generator = Path(__file__).resolve()
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "primitive_relation_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "kind": "method diagram; no experimental outcome values",
            "method_contract": str(contract.relative_to(root)),
            "method_contract_sha256": _sha256(contract),
            "generator": str(generator.relative_to(root)),
            "generator_sha256": _sha256(generator),
            "machine_readable_source": str(source_path.relative_to(root)),
            "machine_readable_source_sha256": _sha256(source_path),
            "supporting_runs": [
                "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0",
                "gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0",
                "gate3_20260830_primitive_relation_model_readiness_v1r_seed0",
            ],
            "manual_value_entry": False,
            "outcome_values": False,
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    retained = [path for path in targets if path != manifest]
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(retained)),
        encoding="utf-8",
    )
    return {
        "figure_id": FIGURE_ID,
        "published_files": len(targets),
        "manifest_sha256": _sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
