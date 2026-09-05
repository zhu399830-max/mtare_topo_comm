#!/usr/bin/env python3
"""Synthetic deterministic contract for the locked scikit-image sidecar."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import skimage
from skimage.measure import marching_cubes


def _mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.round(vertices, 8), dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def main() -> int:
    spacing = 0.1
    coordinates = (np.arange(65, dtype=np.float32) - 32.0) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    sdf = np.sqrt(x * x + y * y + z * z, dtype=np.float32) - np.float32(2.0)
    first = marching_cubes(sdf, level=0.0, spacing=(spacing, spacing, spacing), allow_degenerate=False)
    second = marching_cubes(sdf, level=0.0, spacing=(spacing, spacing, spacing), allow_degenerate=False)
    vertices, faces = first[0], first[1]
    edges = _edge_incidence(faces)
    report = {
        "schema_version": "gate4_meshing_sidecar_contract_v1",
        "skimage_version": skimage.__version__,
        "algorithm": "skimage.measure.marching_cubes",
        "field_shape": list(sdf.shape),
        "spacing_m": spacing,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "degenerate_triangle_count": int(sum(len(set(map(int, face))) != 3 for face in faces)),
        "watertight_edge_incidence": bool(all(count == 2 for count in edges.values())),
        "deterministic_hash_equal": _mesh_hash(vertices, faces) == _mesh_hash(second[0], second[1]),
        "mesh_hash": _mesh_hash(vertices, faces),
    }
    report["status"] = "PASS" if (
        report["vertex_count"] > 0
        and report["triangle_count"] > 0
        and report["degenerate_triangle_count"] == 0
        and report["watertight_edge_incidence"]
        and report["deterministic_hash_equal"]
    ) else "FAIL"
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
