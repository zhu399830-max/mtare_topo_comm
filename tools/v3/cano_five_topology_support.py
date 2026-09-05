"""Cano-backed deterministic topology templates for the five-world CPU pilot."""

from __future__ import annotations

import random
import hashlib
import json
from typing import Any

import numpy as np

from generate_cano_audited_bundle import (
    _per_tunnel_axis,
    _source_precheck,
    _spline_records,
    _stable_graph,
)
from subt_proc_gen.graph import Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from subt_proc_gen.param_classes import (
    IntersectionPtClGenParams,
    IntersectionPtClType,
    PerlinParams,
    TunnelNetworkMeshGenParams,
    TunnelNetworkPtClGenParams,
    TunnelPtClGenParams,
)
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork, TunnelType


def _reset(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    random.seed(seed)
    Node.set_global_counter(0)
    Tunnel.counter = 0
    return np.random.default_rng(seed)


def _jittered_nodes(
    coordinates: dict[str, tuple[float, float, float]], rng: np.random.Generator
) -> dict[str, Node]:
    result = {}
    for name, coordinate in coordinates.items():
        value = np.asarray(coordinate, dtype=np.float64)
        if name not in {"center", "origin"}:
            scale = np.asarray((0.7, 0.7, 0.25), dtype=np.float64)
            value = value + rng.normal(0.0, scale)
        result[name] = Node(value)
    return result


def topology_identity_preview(parent_id: str, topology_seed: int) -> str:
    """Hash the deterministic pre-upstream template without constructing a world."""

    coordinates, tunnel_specs = _blueprint(parent_id)
    rng = np.random.default_rng(topology_seed)
    preview = {}
    for name, coordinate in coordinates.items():
        value = np.asarray(coordinate, dtype=np.float64)
        if name not in {"center", "origin"}:
            value = value + rng.normal(0.0, np.asarray((0.7, 0.7, 0.25)))
        preview[name] = np.round(value, 12).tolist()
    payload = {
        "coordinates": preview,
        "tunnels": [(name, nodes, kind.name) for name, nodes, kind in tunnel_specs],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def template_polyline_length_m(parent_id: str) -> float:
    """Return the unjittered sum of tunnel polyline lengths for scope checks."""

    coordinates, tunnel_specs = _blueprint(parent_id)
    total = 0.0
    for _, node_names, _ in tunnel_specs:
        points = np.asarray([coordinates[name] for name in node_names], dtype=np.float64)
        total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return total


def _blueprint(parent_id: str) -> tuple[dict[str, tuple[float, float, float]], list[tuple[str, list[str], TunnelType]]]:
    if parent_id == "P01_straight_turn":
        return (
            {
                "a": (-150, -5, 0),
                "b": (-90, -5, 0),
                "c": (-30, 0, 0),
                "d": (30, 35, 0),
                "e": (75, 95, 0),
                "f": (100, 155, 0),
            },
            [("grown_0", ["a", "b", "c", "d", "e", "f"], TunnelType.grown)],
        )
    if parent_id == "P02_branch_deadend":
        return (
            {
                "center": (0, 0, 0),
                "a1": (50, 0, 0), "a2": (112, 2, 0),
                "b1": (-25, 43, 0), "b2": (-58, 100, 0),
                "c1": (-27, -42, 0), "c2": (-61, -99, 0),
            },
            [
                ("grown_0", ["center", "a1", "a2"], TunnelType.grown),
                ("grown_1", ["center", "b1", "b2"], TunnelType.grown),
                ("grown_2", ["center", "c1", "c2"], TunnelType.grown),
            ],
        )
    if parent_id == "P03_loop_bottleneck":
        return (
            {
                "center": (0, 0, 0),
                "a1": (38, 0, 0), "a2": (78, 2, 0),
                "b1": (-18, 32, 0), "b2": (-42, 70, 0),
                "c1": (-20, -32, 0), "c2": (-46, -72, 0),
                "ab1": (58, 35, 0), "ab2": (18, 60, 0),
            },
            [
                ("grown_0", ["center", "a1", "a2"], TunnelType.grown),
                ("grown_1", ["center", "b1", "b2"], TunnelType.grown),
                ("grown_2", ["center", "c1", "c2"], TunnelType.grown),
                ("connector_0", ["a2", "ab1", "ab2", "b2"], TunnelType.connector),
            ],
        )
    if parent_id == "P04_chamber_multiexit":
        return (
            {
                "center": (0, 0, 0),
                "e1": (40, 0, 0), "e2": (82, 0, 0),
                "w1": (-40, 0, 0), "w2": (-82, 0, 0),
                "n1": (0, 40, 0), "n2": (0, 82, 0),
                "s1": (0, -40, 0), "s2": (0, -82, 0),
            },
            [
                ("grown_0", ["center", "e1", "e2"], TunnelType.grown),
                ("grown_1", ["center", "w1", "w2"], TunnelType.grown),
                ("grown_2", ["center", "n1", "n2"], TunnelType.grown),
                ("grown_3", ["center", "s1", "s2"], TunnelType.grown),
            ],
        )
    if parent_id == "P05_slope_multiheight":
        return (
            {
                "center": (0, 0, 0),
                "a1": (50, 0, 10), "a2": (110, 2, 25),
                "b1": (-25, 43, -6), "b2": (-58, 100, -12),
                "c1": (-27, -42, 7), "c2": (-61, -99, 16),
            },
            [
                ("grown_0", ["center", "a1", "a2"], TunnelType.grown),
                ("grown_1", ["center", "b1", "b2"], TunnelType.grown),
                ("grown_2", ["center", "c1", "c2"], TunnelType.grown),
            ],
        )
    raise ValueError(f"unknown parent id: {parent_id}")


def build_parent(parent_id: str, topology_seed: int) -> tuple[TunnelNetwork, dict[str, Node], dict[str, Any]]:
    rng = _reset(topology_seed)
    coordinates, tunnel_specs = _blueprint(parent_id)
    nodes = _jittered_nodes(coordinates, rng)
    network = TunnelNetwork(initial_node=False)
    construction_records = []
    for name, node_names, tunnel_type in tunnel_specs:
        tunnel = Tunnel([nodes[node_name] for node_name in node_names], tunnel_type=tunnel_type)
        returned = network.add_tunnel(tunnel)
        success = returned is tunnel and tunnel in network.tunnels
        construction_records.append(
            {
                "name": name,
                "type": tunnel_type.name,
                "node_names": node_names,
                "tunnel_id": int(tunnel.tunnel_id),
                "success": bool(success),
            }
        )
        if not success:
            raise RuntimeError(f"{parent_id}: upstream add_tunnel did not retain {name}")
    return network, nodes, {
        "parent_id": parent_id,
        "topology_seed": topology_seed,
        "construction_records": construction_records,
        "all_constructions_succeeded": all(item["success"] for item in construction_records),
    }


def canonical_exports(network: TunnelNetwork) -> tuple[dict[str, Any], dict[str, Any]]:
    graph, node_ids = _stable_graph(network)
    splines = _spline_records(network, node_ids)
    return graph, splines


def build_mesh_generator(
    network: TunnelNetwork,
    nodes: dict[str, Node],
    parent_id: str,
    geometry_seed: int,
) -> tuple[TunnelNetworkMeshGenerator, dict[str, Any]]:
    rng = _reset(geometry_seed)
    ordered_tunnels = sorted(network.tunnels, key=lambda item: item.tunnel_id)
    radii = [5.0] * len(ordered_tunnels)
    if parent_id == "P03_loop_bottleneck":
        radii = [5.5, 5.0, 3.0, 4.5]
    tunnel_params = {}
    tunnel_records = []
    for tunnel, radius in zip(ordered_tunnels, radii):
        noise = float(rng.uniform(0.22, 0.38))
        tunnel_params[tunnel] = TunnelPtClGenParams(
            dist_between_circles=0.5,
            n_points_per_circle=30,
            radius=radius,
            noise_multiplier=noise,
            perlin_params=PerlinParams.from_defaults(),
        )
        tunnel_records.append(
            {"tunnel_id": int(tunnel.tunnel_id), "radius_m": radius, "noise_multiplier": noise}
        )
    intersection_params = {}
    if parent_id == "P04_chamber_multiexit":
        intersection_params[nodes["center"]] = IntersectionPtClGenParams(
            radius=15.0,
            ptcl_type=IntersectionPtClType.spherical_cavity,
            perlin_params=PerlinParams.from_defaults(),
            points_per_sm=3,
            noise_multiplier=0.25,
        )
    pointcloud_params = TunnelNetworkPtClGenParams.from_defaults(
        pre_set_tunnel_params=tunnel_params,
        pre_set_intersection_params=intersection_params,
    )
    fta_distance = float(rng.uniform(-2.0, -1.0))
    mesh_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_params.fta_distance = fta_distance
    generator = TunnelNetworkMeshGenerator(
        network, ptcl_gen_params=pointcloud_params, meshing_params=mesh_params
    )
    return generator, {
        "geometry_seed": geometry_seed,
        "fta_distance_m": fta_distance,
        "tunnel_parameters": tunnel_records,
        "intersection_mode": "spherical_cavity" if parent_id == "P04_chamber_multiexit" else "default_no_cavity",
    }


__all__ = [
    "_per_tunnel_axis",
    "_source_precheck",
    "build_parent",
    "canonical_exports",
    "build_mesh_generator",
    "template_polyline_length_m",
    "topology_identity_preview",
]
