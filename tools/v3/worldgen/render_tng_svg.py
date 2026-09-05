#!/usr/bin/env python3
"""Render an auditable two-view SVG for one topology-parent JSON artifact."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _transform(
    first: float,
    second: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_first, max_first, min_second, max_second = bounds
    panel_x, panel_y, panel_w, panel_h = panel
    first_span = max(max_first - min_first, 1e-9)
    second_span = max(max_second - min_second, 1e-9)
    scale = min(panel_w / first_span, panel_h / second_span)
    used_w = first_span * scale
    used_h = second_span * scale
    x = panel_x + (panel_w - used_w) / 2 + (first - min_first) * scale
    y = panel_y + (panel_h - used_h) / 2 + (max_second - second) * scale
    return x, y


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    nodes = {
        node["id"]: tuple(float(value) for value in node["position"])
        for node in payload["nodes"]
    }
    if not nodes:
        raise ValueError("input graph has no nodes")

    x_values = [position[0] for position in nodes.values()]
    y_values = [position[1] for position in nodes.values()]
    z_values = [position[2] for position in nodes.values()]
    xy_bounds = (min(x_values), max(x_values), min(y_values), max(y_values))
    xz_bounds = (min(x_values), max(x_values), min(z_values), max(z_values))
    top_panel = (60.0, 155.0, 610.0, 650.0)
    elevation_panel = (750.0, 155.0, 610.0, 650.0)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1420" height="900" viewBox="0 0 1420 900">',
        '<rect width="1420" height="900" fill="#fbfcfe"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#152238}.title{font-size:24px;font-weight:700}.sub{font-size:14px}.panel{font-size:18px;font-weight:700}.axis{font-size:12px;fill:#4b5563}.node{fill:#ffffff;stroke:#12355b;stroke-width:1.4}.label{font-size:9px;fill:#334155}.rgtg{stroke:#276fbf;stroke-width:3;stroke-linecap:round}.ctg{stroke:#d1495b;stroke-width:4;stroke-dasharray:9 6;stroke-linecap:round}.frame{fill:none;stroke:#94a3b8;stroke-width:1}</style>',
        f'<text x="60" y="48" class="title">TNG topology-parent smoke test: {html.escape(payload["topology_parent_id"])}</text>',
        f'<text x="60" y="78" class="sub">seed={payload["topology_seed"]} | nodes={payload["stats"]["node_count"]} | edges={payload["stats"]["edge_count"]} | cycle rank={payload["stats"]["cycle_rank"]} | sampled minimum clearance={payload["stats"]["sampled_min_nonincident_edge_distance"]:.3f} m</text>',
        '<text x="60" y="105" class="sub">Blue solid: RGTG spanning tree. Red dashed: CTG connector. Coordinates and distances are metres. This is a graph preview, not tunnel mesh geometry.</text>',
        '<text x="60" y="140" class="panel">Top view (X–Y)</text>',
        '<text x="750" y="140" class="panel">Elevation (X–Z)</text>',
        '<rect x="60" y="155" width="610" height="650" class="frame"/>',
        '<rect x="750" y="155" width="610" height="650" class="frame"/>',
    ]

    for panel_name, bounds, panel, dimensions in (
        ("xy", xy_bounds, top_panel, (0, 1)),
        ("xz", xz_bounds, elevation_panel, (0, 2)),
    ):
        for edge in payload["edges"]:
            start = nodes[edge["u"]]
            end = nodes[edge["v"]]
            x1, y1 = _transform(start[dimensions[0]], start[dimensions[1]], bounds, panel)
            x2, y2 = _transform(end[dimensions[0]], end[dimensions[1]], bounds, panel)
            css_class = "ctg" if edge["kind"] == "CTG" else "rgtg"
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{css_class}"><title>{html.escape(edge["id"])} {edge["kind"]}: {html.escape(edge["u"])}–{html.escape(edge["v"])}</title></line>'
            )
        for node_id, position in sorted(nodes.items()):
            x, y = _transform(position[dimensions[0]], position[dimensions[1]], bounds, panel)
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" class="node"><title>{html.escape(node_id)}: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m</title></circle>'
            )
            lines.append(
                f'<text x="{x + 6:.2f}" y="{y - 5:.2f}" class="label">{html.escape(node_id)}</text>'
            )

    lines.extend(
        [
            f'<text x="60" y="833" class="axis">X: {xy_bounds[0]:.2f} to {xy_bounds[1]:.2f} m; Y: {xy_bounds[2]:.2f} to {xy_bounds[3]:.2f} m</text>',
            f'<text x="750" y="833" class="axis">X: {xz_bounds[0]:.2f} to {xz_bounds[1]:.2f} m; Z: {xz_bounds[2]:.2f} to {xz_bounds[3]:.2f} m</text>',
            f'<text x="60" y="868" class="axis">Canonical SHA-256: {html.escape(payload["canonical_hash"])}</text>',
            '</svg>',
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
