#!/usr/bin/env python3
"""Generate one fixed-seed Cano world bundle without modifying upstream source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from subt_proc_gen.graph import Node
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenParams,
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
)
from subt_proc_gen.serialization import WorldInfo
from subt_proc_gen.tunnel import (
    GrownTunnelGenerationParams,
    Tunnel,
    TunnelNetwork,
    TunnelNetworkParams,
)


EXPECTED_COMMIT = "b6c77621187404b4dfab1249c7a1b40f63ad9ab3"
EXPECTED_REMOTE = "https://github.com/LorenzoCanoAn/procedural-subt-gen"
SOURCE_ROOT = Path(
    "/home/zeng-workstation/mtare_topo_comm/external/procedural-subt-gen"
)
ENTRYPOINT_HASH = "7c765d160b21198d611c297b160664aaffc3c12c9d18c36fc9851a9d182113a4"
MODEL_SDF_TEXT = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="tunnel_network">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></collision>
      <visual name="visual"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
    </link>
  </model>
</sdf>
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _source_precheck() -> dict[str, Any]:
    imported = Path(sys.modules[TunnelNetwork.__module__].__file__).resolve()
    expected_import_root = (SOURCE_ROOT / "src").resolve()
    snapshot = {
        "remote": _git("remote", "get-url", "origin"),
        "commit": _git("rev-parse", "HEAD"),
        "tracked_clean": not bool(_git("status", "--porcelain=v1")),
        "entrypoint_sha256": _sha256(SOURCE_ROOT / "scripts/generate_environments.py"),
        "imported_tunnel_module": str(imported),
        "imported_from_fixed_checkout": imported.is_relative_to(expected_import_root),
    }
    expected = {
        "remote": EXPECTED_REMOTE,
        "commit": EXPECTED_COMMIT,
        "tracked_clean": True,
        "entrypoint_sha256": ENTRYPOINT_HASH,
        "imported_from_fixed_checkout": True,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            raise RuntimeError(f"source precheck failed for {key}: {snapshot[key]!r}")
    return snapshot


def _normalize_result(result: Any) -> tuple[bool, Tunnel | None, str]:
    if isinstance(result, tuple) and len(result) == 2:
        success, tunnel = result
        valid = isinstance(success, (bool, np.bool_)) and (
            (bool(success) and isinstance(tunnel, Tunnel))
            or (not bool(success) and tunnel is None)
        )
        return bool(success) if valid else False, tunnel if valid else None, "tuple_v1"
    return False, None, f"unexpected_{type(result).__name__}"


def _node_xyz(node: Node) -> np.ndarray:
    values = np.asarray(node.xyz, dtype=float).reshape(-1)
    if values.shape != (3,):
        raise ValueError(f"upstream node xyz must flatten to 3 values, got {values.shape}")
    return values


def _stable_graph(network: TunnelNetwork) -> tuple[dict[str, Any], dict[Node, str]]:
    network.compute_node_types()
    ordered_nodes = sorted(
        network.nodes,
        key=lambda node: tuple(_node_xyz(node).tolist()),
    )
    node_ids = {node: f"node_{index:04d}" for index, node in enumerate(ordered_nodes)}
    tunnels = sorted(network.tunnels, key=lambda tunnel: tunnel.tunnel_id)
    edge_tunnels: dict[tuple[str, str], set[int]] = {}
    for tunnel in tunnels:
        for first, second in zip(tunnel.nodes[:-1], tunnel.nodes[1:]):
            key = tuple(sorted((node_ids[first], node_ids[second])))
            edge_tunnels.setdefault(key, set()).add(int(tunnel.tunnel_id))
    edges = [
        {
            "id": f"edge_{index:04d}",
            "node_ids": list(pair),
            "tunnel_ids": sorted(tunnel_ids),
        }
        for index, (pair, tunnel_ids) in enumerate(sorted(edge_tunnels.items()))
    ]
    nodes = []
    for node in ordered_nodes:
        incident = sorted(
            int(tunnel.tunnel_id) for tunnel in network._tunnels_of_node[node]
        )
        nodes.append(
            {
                "id": node_ids[node],
                "xyz": _node_xyz(node).tolist(),
                "degree": int(len(network.connected_nodes(node))),
                "node_type": network.get_node_type(node).name,
                "incident_tunnel_ids": incident,
            }
        )
    tunnel_records = [
        {
            "id": int(tunnel.tunnel_id),
            "type": tunnel.tunnel_type.name,
            "node_ids": [node_ids[node] for node in tunnel.nodes],
            "endpoint_node_ids": [node_ids[tunnel.nodes[0]], node_ids[tunnel.nodes[-1]]],
        }
        for tunnel in tunnels
    ]
    intersections = [
        {
            "node_id": node_ids[node],
            "node_type": network.get_node_type(node).name,
            "incident_tunnel_ids": sorted(
                int(tunnel.tunnel_id) for tunnel in network._tunnels_of_node[node]
            ),
        }
        for node in sorted(network.intersections, key=lambda item: node_ids[item])
    ]
    adjacency = {node_id: set() for node_id in node_ids.values()}
    for edge in edges:
        first, second = edge["node_ids"]
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            unseen = adjacency[current] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    cycle_rank = len(edges) - len(nodes) + components
    graph = {
        "schema_version": "cano_audited_graph_v1",
        "coordinate_frame": "cano_world",
        "units": "expected_m_not_independently_certified",
        "nodes": nodes,
        "edges": edges,
        "tunnels": tunnel_records,
        "intersections": intersections,
        "statistics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "tunnel_count": len(tunnel_records),
            "intersection_count": len(intersections),
            "connected_components": components,
            "cycle_rank": cycle_rank,
        },
    }
    return graph, node_ids


def _spline_records(network: TunnelNetwork, node_ids: dict[Node, str]) -> dict[str, Any]:
    records = []
    for tunnel in sorted(network.tunnels, key=lambda item: item.tunnel_id):
        distances, points, tangents = tunnel.spline.discretize(0.5)
        records.append(
            {
                "tunnel_id": int(tunnel.tunnel_id),
                "type": tunnel.tunnel_type.name,
                "endpoint_node_ids": [
                    node_ids[tunnel.nodes[0]],
                    node_ids[tunnel.nodes[-1]],
                ],
                "sample_resolution_expected_m": 0.5,
                "distances": np.asarray(distances).reshape(-1).astype(float).tolist(),
                "points": np.asarray(points).astype(float).tolist(),
                "tangents": np.asarray(tangents).astype(float).tolist(),
            }
        )
    return {
        "schema_version": "cano_audited_splines_v1",
        "coordinate_frame": "cano_world",
        "tunnels": records,
    }


def _per_tunnel_axis(mesh_generator: TunnelNetworkMeshGenerator) -> np.ndarray:
    rows = []
    for tunnel in sorted(mesh_generator.tunnels, key=lambda item: item.tunnel_id):
        radius = float(mesh_generator.ptcl_params_of_tunnel(tunnel).radius)
        interior_points = mesh_generator.aps_of_tunnel(tunnel)
        interior_vectors = mesh_generator.avs_of_tunnel(tunnel)
        if len(interior_points):
            rows.append(
                np.column_stack(
                    (
                        interior_points,
                        interior_vectors,
                        np.full(len(interior_points), radius),
                        np.ones(len(interior_points)),
                        np.full(len(interior_points), tunnel.tunnel_id),
                    )
                )
            )
        for intersection in sorted(
            mesh_generator.intersections,
            key=lambda node: tuple(_node_xyz(node).tolist()),
        ):
            associated = mesh_generator._aps_avs_of_intersections[intersection]
            if tunnel not in associated or not len(associated[tunnel]):
                continue
            values = associated[tunnel]
            rows.append(
                np.column_stack(
                    (
                        values[:, :3],
                        values[:, 3:6],
                        np.full(len(values), radius),
                        np.full(len(values), 2.0),
                        np.full(len(values), tunnel.tunnel_id),
                    )
                )
            )
    return np.vstack(rows) if rows else np.zeros((0, 9), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--grown", required=True, type=int)
    parser.add_argument("--connector", required=True, type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    if args.grown != 3 or args.connector != 1:
        raise ValueError("this approved smoke is frozen to 3 grown and 1 connector")
    output.mkdir(parents=True, exist_ok=False)

    source_before = _source_precheck()
    np.random.seed(args.seed)
    random.seed(args.seed)
    Node.set_global_counter(0)
    Tunnel.counter = 0
    fta_distance = float(np.random.uniform(-2.0, -1.0))
    network_params = TunnelNetworkParams.from_defaults()
    network_params.min_distance_between_intersections = 30
    network_params.collision_distance = 10
    network = TunnelNetwork(params=network_params)
    audit: dict[str, Any] = {
        "schema_version": "cano_generation_return_audit_v1",
        "seed_controls": {
            "numpy_random_seed": args.seed,
            "python_random_seed": args.seed,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "node_counter_reset": 0,
            "tunnel_counter_reset": 0,
        },
        "requested": {"grown": args.grown, "connector": args.connector},
        "calls": [],
    }
    GrownTunnelGenerationParams._random_distance_range = (100, 300)
    GrownTunnelGenerationParams._random_horizontal_tendency_range_deg = (-40, 40)
    GrownTunnelGenerationParams._random_horizontal_noise_range_deg = (-30, 30)
    GrownTunnelGenerationParams._random_min_segment_length_fraction_range = (0.05, 0.05)
    GrownTunnelGenerationParams._random_max_segment_length_fraction_range = (0.10, 0.10)

    for index in range(args.grown):
        params = GrownTunnelGenerationParams.random()
        result = network.add_random_grown_tunnel(params=params, n_trials=100)
        success, tunnel, return_schema = _normalize_result(result)
        audit["calls"].append(
            {
                "kind": "grown",
                "index": index,
                "success": success,
                "return_schema": return_schema,
                "tunnel_id": int(tunnel.tunnel_id) if tunnel is not None else None,
            }
        )
        if not success:
            audit["overall_status"] = "FAIL_GROWN_RETURN"
            _write_json(output / "generation_audit.json", audit)
            return 3

    for index in range(args.connector):
        result = network.add_random_connector_tunnel(n_trials=100)
        success, tunnel, return_schema = _normalize_result(result)
        audit["calls"].append(
            {
                "kind": "connector",
                "index": index,
                "success": success,
                "return_schema": return_schema,
                "tunnel_id": int(tunnel.tunnel_id) if tunnel is not None else None,
            }
        )
        if not success:
            audit["overall_status"] = "FAIL_CONNECTOR_RETURN"
            _write_json(output / "generation_audit.json", audit)
            return 4

    audit["overall_status"] = "PASS_ALL_REQUESTED_TUNNELS_CONFIRMED"
    audit["succeeded"] = {"grown": args.grown, "connector": args.connector}
    _write_json(output / "generation_audit.json", audit)

    graph, node_ids = _stable_graph(network)
    graph["generation"] = {
        "requested_grown": args.grown,
        "successful_grown": args.grown,
        "requested_connector": args.connector,
        "successful_connector": args.connector,
    }
    _write_json(output / "graph.json", graph)
    _write_json(output / "splines.json", _spline_records(network, node_ids))

    pointcloud_params = TunnelNetworkPtClGenParams.random()
    mesh_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_params.fta_distance = fta_distance
    mesh_generator = TunnelNetworkMeshGenerator(
        network,
        ptcl_gen_params=pointcloud_params,
        meshing_params=mesh_params,
    )
    mesh_generator.compute_all()
    mesh_path = output / "mesh.obj"
    mesh_generator.save_mesh(str(mesh_path))
    axis = _per_tunnel_axis(mesh_generator)
    np.savetxt(output / "axis.txt", axis, fmt="%.9g")
    np.savetxt(output / "fta_dist.txt", np.array([fta_distance]), fmt="%.17g")
    (output / "model.sdf").write_text(
        MODEL_SDF_TEXT.format(mesh=mesh_path), encoding="utf-8"
    )
    world_info = WorldInfo.from_TunnelNetworkMeshGenerator(
        mesh_generator, spline_res=0.5
    )
    world_info.save_json(str(output / "world_info.json"))

    source_after = _source_precheck()
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "metadata.json"
    )
    metadata = {
        "schema_version": "cano_audited_bundle_metadata_v1",
        "role": "TEMPORARY_SMOKE_FAILURE_OR_ACCEPTANCE_EVIDENCE_NOT_DATASET",
        "formal_dataset_world_count": 0,
        "seed": args.seed,
        "random_controls": audit["seed_controls"],
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_before == source_after,
        "generation": audit,
        "parameters": {
            "grown": args.grown,
            "connector": args.connector,
            "grown_trials_per_call": 100,
            "connector_trials_per_call": 100,
            "collision_distance_expected_m": 10,
            "min_intersection_distance_expected_m": 30,
            "fta_distance_expected_m": fta_distance,
            "spline_resolution_expected_m": 0.5,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "upstream_core_modified": False,
        },
        "claim_boundary": (
            "The adapter exports and audits one fixed-seed upstream-generated candidate. "
            "It does not repair mesh topology, certify scale, create a dataset world, "
            "run Isaac/LiDAR, label samples, or train a model."
        ),
    }
    _write_json(output / "metadata.json", metadata)
    print(json.dumps({"status": "BUNDLE_EMITTED_FOR_STRICT_AUDIT", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
