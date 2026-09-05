#!/usr/bin/env python3
"""Render three deterministic raster projections from an OBJ mesh without GUI deps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import zlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    scanlines = bytearray()
    row_bytes = width * 3
    for y in range(height):
        scanlines.append(0)
        scanlines.extend(pixels[y * row_bytes : (y + 1) * row_bytes])
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )
    with path.open("xb") as stream:
        stream.write(payload)


def _line(
    pixels: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        for oy in range(-thickness, thickness + 1):
            for ox in range(-thickness, thickness + 1):
                x, y = x0 + ox, y0 + oy
                if 0 <= x < width and 0 <= y < height:
                    offset = (y * width + x) * 3
                    pixels[offset : offset + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def main() -> int:
    args = parse_args()
    vertices: list[tuple[float, float, float]] = []
    with args.obj.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError("OBJ contains no vertices")
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    node_positions = {
        node["id"]: tuple(float(value) for value in node["position"])
        for node in topology["nodes"]
    }

    width, height = 1620, 900
    pixels = bytearray([248, 250, 252] * width * height)
    panels = (
        (40, 70, 500, 770, lambda p: (p[0], p[1], p[2])),
        (560, 70, 500, 770, lambda p: (p[0], p[2], p[1])),
        (1080, 70, 500, 770, lambda p: (0.82 * p[0] - 0.52 * p[1], p[2] + 0.22 * p[0] + 0.22 * p[1], p[0] + p[1])),
    )
    for panel_x, panel_y, panel_w, panel_h, projection in panels:
        projected = [projection(vertex) for vertex in vertices]
        min_x = min(point[0] for point in projected)
        max_x = max(point[0] for point in projected)
        min_y = min(point[1] for point in projected)
        max_y = max(point[1] for point in projected)
        scale = min(
            (panel_w - 30) / max(max_x - min_x, 1e-9),
            (panel_h - 30) / max(max_y - min_y, 1e-9),
        )
        used_w = (max_x - min_x) * scale
        used_h = (max_y - min_y) * scale

        def screen(point3: tuple[float, float, float]) -> tuple[int, int]:
            point = projection(point3)
            x = panel_x + (panel_w - used_w) / 2 + (point[0] - min_x) * scale
            y = panel_y + (panel_h - used_h) / 2 + (max_y - point[1]) * scale
            return int(round(x)), int(round(y))

        for x in range(panel_x, panel_x + panel_w):
            for y in (panel_y, panel_y + panel_h - 1):
                offset = (y * width + x) * 3
                pixels[offset : offset + 3] = bytes((148, 163, 184))
        for y in range(panel_y, panel_y + panel_h):
            for x in (panel_x, panel_x + panel_w - 1):
                offset = (y * width + x) * 3
                pixels[offset : offset + 3] = bytes((148, 163, 184))

        depth_min = min(point[2] for point in projected)
        depth_span = max(max(point[2] for point in projected) - depth_min, 1e-9)
        for vertex, point in zip(vertices, projected):
            x, y = screen(vertex)
            shade = (point[2] - depth_min) / depth_span
            color = (
                int(116 + 45 * shade),
                int(76 + 38 * shade),
                int(43 + 25 * shade),
            )
            for oy in (0, 1):
                for ox in (0, 1):
                    px, py = x + ox, y + oy
                    if 0 <= px < width and 0 <= py < height:
                        offset = (py * width + px) * 3
                        pixels[offset : offset + 3] = bytes(color)

        for edge in topology["edges"]:
            color = (209, 73, 91) if edge["kind"] == "CTG" else (39, 111, 191)
            _line(
                pixels,
                width,
                height,
                screen(node_positions[edge["u"]]),
                screen(node_positions[edge["v"]]),
                color,
                thickness=1,
            )
        for position in node_positions.values():
            x, y = screen(position)
            for oy in range(-3, 4):
                for ox in range(-3, 4):
                    if ox * ox + oy * oy <= 9 and 0 <= x + ox < width and 0 <= y + oy < height:
                        offset = ((y + oy) * width + (x + ox)) * 3
                        pixels[offset : offset + 3] = bytes((255, 255, 255))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_png(args.output, width, height, pixels)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "vertex_count": len(vertices),
                "panels_left_to_right": ["top_xy", "elevation_xz", "oblique_3d"],
                "topology_parent_id": topology["topology_parent_id"],
                "source": "actual_obj_vertices_with_tng_overlay",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
