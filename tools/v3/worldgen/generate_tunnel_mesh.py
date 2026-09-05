#!/usr/bin/env python3
"""Generate one audited tunnel mesh and USD from one TNG topology parent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.data.worldgen import (
    SeedBundle,
    TunnelMeshParameters,
    generate_tunnel_mesh,
    mesh_stats,
    tng_from_dict,
    write_obj,
    write_usda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    topology_payload = json.loads(args.topology.read_text(encoding="utf-8"))
    graph = tng_from_dict(topology_payload)
    seed_payload = topology_payload.get("seed_bundle")
    if not isinstance(seed_payload, dict):
        raise ValueError("topology artifact must include seed_bundle")
    seed_bundle = SeedBundle(**seed_payload)
    parameters = TunnelMeshParameters.from_dict(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    mesh = generate_tunnel_mesh(graph, seed_bundle, parameters)
    args.output_dir.mkdir(parents=True)
    obj_path = args.output_dir / "collision_render_mesh.obj"
    usd_path = args.output_dir / "isaac_stage.usda"
    validation_path = args.output_dir / "mesh_validation.json"
    write_obj(mesh, obj_path)
    write_usda(mesh, usd_path)
    stats = mesh_stats(mesh)
    navigation_grade = parameters.navigation_audit_enabled
    validation = {
        "schema_version": (
            "tunnel_mesh_validation_v2"
            if navigation_grade
            else "tunnel_mesh_validation_v1"
        ),
        "status": "PASS",
        "scope": (
            "single_navigation_grade_geometry_no_dataset_no_training"
            if navigation_grade
            else "single_geometry_smoke_no_dataset_no_training"
        ),
        "topology_parent_id": graph.topology_parent_id,
        "topology_hash": graph.canonical_hash(),
        "topology_cycle_rank": graph.stats()["cycle_rank"],
        "geometry_variant_id": parameters.geometry_variant_id,
        "seed_bundle": asdict(seed_bundle),
        "parameters": asdict(parameters),
        "mesh": stats,
        "navigation_audit": mesh.navigation_audit,
        "files": {
            "obj": obj_path.name,
            "obj_sha256": _sha256(obj_path),
            "usd": usd_path.name,
            "usd_sha256": _sha256(usd_path),
        },
        "limitations": (
            [
                "robot probes certify the frozen centerline corridor, not dynamic controller execution",
                "junction floor triangles still require Isaac physics contact rollout",
                "no surface noise, clutter, material randomization, or LiDAR capture",
            ]
            if navigation_grade
            else [
                "voxel boundary is intentionally blocky",
                "graph/mesh checks do not replace robot-footprint planner rollout",
                "no surface noise, clutter, material randomization, or LiDAR capture",
            ]
        ),
    }
    with validation_path.open("x", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
