"""Typed construction-program supervision for primitive-relation learning.

The archived Cano assets retain topology, one centreline per physical tunnel,
and one radius per tunnel.  A *primitive* is deliberately defined per graph
edge rather than per tunnel: connector tunnels attach to the interior of grown
tunnels, so tunnel-level endpoint relations cannot represent the construction.

This module is train-free.  It reconstructs the edge-level swept primitives
and their endpoint composition graph, and explicitly audits which provenance
needed by a future LiDAR Teacher is (and is not) present in archived assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.topology.continuous_trajectory import (
    polyline_between,
    project_to_polyline,
)


@dataclass(frozen=True)
class PrimitiveEndpoint:
    primitive_id: str
    endpoint_index: int
    node_id: str
    xyz_m: tuple[float, float, float]
    composition_anchor_xyz_m: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "endpoint_index": self.endpoint_index,
            "node_id": self.node_id,
            "xyz_m": list(self.xyz_m),
            "composition_anchor_xyz_m": list(
                self.xyz_m if self.composition_anchor_xyz_m is None else self.composition_anchor_xyz_m
            ),
        }


@dataclass(frozen=True)
class SweptPrimitive:
    primitive_id: str
    source_edge_id: str
    source_tunnel_id: str
    centerline_xyz_m: np.ndarray
    radius_m: float
    endpoint_projection_error_m: tuple[float, float]
    endpoints: tuple[PrimitiveEndpoint, PrimitiveEndpoint]

    def __post_init__(self) -> None:
        points = np.asarray(self.centerline_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("primitive centerline must have shape [N>=2,3]")
        if not np.isfinite(points).all():
            raise ValueError("primitive centerline must be finite")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
            raise ValueError("primitive centerline contains a degenerate segment")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("primitive radius must be finite and positive")
        errors = np.asarray(self.endpoint_projection_error_m, dtype=np.float64)
        if errors.shape != (2,) or np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("endpoint projection errors must be two finite nonnegative values")
        object.__setattr__(self, "centerline_xyz_m", points)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.centerline_xyz_m, axis=0), axis=1).sum())

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "primitive_type": "swept_circle_archived_cano",
            "source_edge_id": self.source_edge_id,
            "source_tunnel_id": self.source_tunnel_id,
            "centerline_xyz_m": self.centerline_xyz_m.astype(float).tolist(),
            "radius_m": float(self.radius_m),
            "length_m": self.length_m,
            "endpoint_projection_error_m": list(self.endpoint_projection_error_m),
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
        }


@dataclass(frozen=True)
class EndpointComposition:
    node_id: str
    member_endpoints: tuple[PrimitiveEndpoint, ...]
    anchor_xyz_m: tuple[float, float, float] | None = None

    @property
    def degree(self) -> int:
        return len(self.member_endpoints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": "endpoint_union",
            "node_id": self.node_id,
            "degree": self.degree,
            "anchor_xyz_m": None if self.anchor_xyz_m is None else list(self.anchor_xyz_m),
            "member_endpoints": [value.as_dict() for value in self.member_endpoints],
        }


@dataclass(frozen=True)
class NodeDegreeMismatch:
    node_id: str
    declared_degree: int
    edge_incidence_degree: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "declared_degree": self.declared_degree,
            "edge_incidence_degree": self.edge_incidence_degree,
        }


@dataclass(frozen=True)
class PrimitiveConstructionGraph:
    coordinate_frame: str
    primitives: tuple[SweptPrimitive, ...]
    compositions: tuple[EndpointComposition, ...]
    endpoint_attachment_mode: str = "coincident_axis"
    node_degree_source: str = "declared"
    node_degree_mismatches: tuple[NodeDegreeMismatch, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "primitive_construction_graph_v1",
            "coordinate_frame": self.coordinate_frame,
            "primitive_definition": "one spline-clipped swept primitive per physical graph edge",
            "endpoint_attachment_mode": self.endpoint_attachment_mode,
            "node_degree_source": self.node_degree_source,
            "node_degree_mismatches": [value.as_dict() for value in self.node_degree_mismatches],
            "primitives": [value.as_dict() for value in self.primitives],
            "composition_operations": [value.as_dict() for value in self.compositions],
        }


@dataclass(frozen=True)
class ArchivedSupervisionAudit:
    edge_count: int
    primitive_count: int
    composition_count: int
    maximum_endpoint_projection_error_m: float
    edge_level_primitives_unique: bool
    endpoint_compositions_complete: bool
    mesh_triangle_provenance_present: bool
    lidar_hit_provenance_present: bool
    construction_supervision_complete: bool
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_count": self.edge_count,
            "primitive_count": self.primitive_count,
            "composition_count": self.composition_count,
            "maximum_endpoint_projection_error_m": self.maximum_endpoint_projection_error_m,
            "edge_level_primitives_unique": self.edge_level_primitives_unique,
            "endpoint_compositions_complete": self.endpoint_compositions_complete,
            "mesh_triangle_provenance_present": self.mesh_triangle_provenance_present,
            "lidar_hit_provenance_present": self.lidar_hit_provenance_present,
            "construction_supervision_complete": self.construction_supervision_complete,
            "failure_reasons": list(self.failure_reasons),
        }


def _exactly_one(values: Sequence[Any], description: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{description} must contain exactly one value")
    return values[0]


def build_primitive_construction_graph(
    graph: Mapping[str, Any],
    splines: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    maximum_endpoint_projection_error_m: float = 0.01,
    endpoint_attachment_mode: str = "coincident_axis",
    node_degree_source: str = "declared",
) -> PrimitiveConstructionGraph:
    """Recover an edge-level primitive graph without collapsing tunnel identity."""

    node_xyz = {
        str(row["id"]): np.asarray(row["xyz"], dtype=np.float64)
        for row in graph.get("nodes", ())
    }
    spline_by_tunnel = {
        str(row["tunnel_id"]): np.asarray(row["points"], dtype=np.float64)
        for row in splines.get("tunnels", ())
    }
    radius_by_tunnel = {
        str(row["tunnel_id"]): float(row["radius_m"])
        for row in geometry.get("tunnels", ())
    }
    if not node_xyz or not spline_by_tunnel or not radius_by_tunnel:
        raise ValueError("graph, spline and geometry records must be nonempty")
    if set(spline_by_tunnel) != set(radius_by_tunnel):
        raise ValueError("spline and geometry tunnel identities differ")

    if endpoint_attachment_mode not in {"coincident_axis", "free_space_overlap"}:
        raise ValueError("endpoint_attachment_mode must be coincident_axis or free_space_overlap")
    if node_degree_source not in {"declared", "edge_incidence"}:
        raise ValueError("node_degree_source must be declared or edge_incidence")
    primitives: list[SweptPrimitive] = []
    endpoints_by_node: dict[str, list[PrimitiveEndpoint]] = {key: [] for key in node_xyz}
    seen_edges: set[str] = set()
    for edge in sorted(graph.get("edges", ()), key=lambda row: str(row["id"])):
        edge_id = str(edge["id"])
        if edge_id in seen_edges:
            raise ValueError(f"duplicate graph edge identity: {edge_id}")
        seen_edges.add(edge_id)
        tunnel_id = str(_exactly_one(edge.get("tunnel_ids", ()), f"edge {edge_id} tunnel_ids"))
        if tunnel_id not in spline_by_tunnel:
            raise ValueError(f"edge {edge_id} references an unknown tunnel {tunnel_id}")
        node_ids = tuple(str(value) for value in edge.get("node_ids", ()))
        if len(node_ids) != 2 or node_ids[0] == node_ids[1]:
            raise ValueError(f"edge {edge_id} must connect two distinct nodes")
        if any(value not in node_xyz for value in node_ids):
            raise ValueError(f"edge {edge_id} references an unknown node")
        line = spline_by_tunnel[tunnel_id]
        projections = tuple(project_to_polyline(node_xyz[value], line) for value in node_ids)
        errors = tuple(float(value.error_m) for value in projections)
        if endpoint_attachment_mode == "coincident_axis":
            if max(errors) > float(maximum_endpoint_projection_error_m) + 1e-12:
                raise ValueError(
                    f"edge {edge_id} endpoint projection exceeds contract: {max(errors):.9f}m"
                )
        elif any(error > radius_by_tunnel[tunnel_id] + 1e-12 for error in errors):
            raise ValueError(
                f"edge {edge_id} composition anchor lies outside declared tunnel free space: "
                f"error={max(errors):.9f}m radius={radius_by_tunnel[tunnel_id]:.9f}m"
            )
        clipped = polyline_between(line, projections[0].arc_m, projections[1].arc_m)
        primitive_id = f"primitive:{edge_id}"
        endpoints = tuple(
            PrimitiveEndpoint(
                primitive_id=primitive_id,
                endpoint_index=index,
                node_id=node_id,
                xyz_m=tuple(float(x) for x in clipped[0 if index == 0 else -1]),
                composition_anchor_xyz_m=tuple(float(x) for x in node_xyz[node_id]),
            )
            for index, node_id in enumerate(node_ids)
        )
        primitive = SweptPrimitive(
            primitive_id=primitive_id,
            source_edge_id=edge_id,
            source_tunnel_id=tunnel_id,
            centerline_xyz_m=clipped,
            radius_m=radius_by_tunnel[tunnel_id],
            endpoint_projection_error_m=errors,
            endpoints=endpoints,  # type: ignore[arg-type]
        )
        primitives.append(primitive)
        for endpoint in endpoints:
            endpoints_by_node[endpoint.node_id].append(endpoint)

    if len(primitives) != len(graph.get("edges", ())):
        raise RuntimeError("one-to-one edge primitive construction failed")
    compositions = tuple(
        EndpointComposition(
            node_id=node_id,
            member_endpoints=tuple(
                sorted(values, key=lambda value: (value.primitive_id, value.endpoint_index))
            ),
            anchor_xyz_m=tuple(float(x) for x in node_xyz[node_id]),
        )
        for node_id, values in sorted(endpoints_by_node.items())
        if values
    )
    declared_degree = {str(row["id"]): int(row["degree"]) for row in graph["nodes"]}
    degree_mismatches: list[NodeDegreeMismatch] = []
    for composition in compositions:
        if composition.degree != declared_degree[composition.node_id]:
            mismatch = NodeDegreeMismatch(
                node_id=composition.node_id,
                declared_degree=declared_degree[composition.node_id],
                edge_incidence_degree=composition.degree,
            )
            degree_mismatches.append(mismatch)
            if node_degree_source == "declared":
                raise ValueError(
                    f"node {composition.node_id} endpoint degree {composition.degree} "
                    f"differs from graph degree {declared_degree[composition.node_id]}"
                )
    return PrimitiveConstructionGraph(
        coordinate_frame=str(graph.get("coordinate_frame", "unknown")),
        primitives=tuple(primitives),
        compositions=compositions,
        endpoint_attachment_mode=endpoint_attachment_mode,
        node_degree_source=node_degree_source,
        node_degree_mismatches=tuple(degree_mismatches),
    )


def obj_has_face_provenance(path: Path) -> bool:
    """Return whether an OBJ partitions faces by object/group/material identity."""

    face_partitions: set[str] = set()
    current = ""
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith(("g ", "o ", "usemtl ")):
                current = line
            elif line.startswith("f "):
                face_partitions.add(current)
                if len(face_partitions) > 1 or (face_partitions and "" not in face_partitions):
                    return True
    return False


def audit_archived_supervision(
    construction: PrimitiveConstructionGraph,
    graph: Mapping[str, Any],
    *,
    mesh_obj_path: Path,
    lidar_array_names: Sequence[str],
) -> ArchivedSupervisionAudit:
    """Fail closed when archived mesh/range data cannot identify primitive hits."""

    primitive_ids = [value.primitive_id for value in construction.primitives]
    endpoint_count = sum(value.degree for value in construction.compositions)
    edge_unique = len(primitive_ids) == len(set(primitive_ids)) == len(graph.get("edges", ()))
    compositions_complete = endpoint_count == 2 * len(construction.primitives)
    mesh_provenance = obj_has_face_provenance(mesh_obj_path)
    lidar_names = {str(value) for value in lidar_array_names}
    lidar_provenance = bool(
        {"primitive_hit_id", "triangle_id"} & lidar_names
        or {"surface_id", "geometry_id"} & lidar_names
    )
    reasons: list[str] = []
    if not edge_unique:
        reasons.append("edge_to_primitive_mapping_is_not_one_to_one")
    if not compositions_complete:
        reasons.append("endpoint_composition_does_not_cover_every_primitive_endpoint")
    if not mesh_provenance:
        reasons.append("poisson_obj_has_no_per_face_primitive_or_surface_identity")
    if not lidar_provenance:
        reasons.append("archived_lidar_has_no_triangle_surface_or_primitive_hit_identity")
    maximum_error = max(
        error
        for primitive in construction.primitives
        for error in primitive.endpoint_projection_error_m
    )
    return ArchivedSupervisionAudit(
        edge_count=len(graph.get("edges", ())),
        primitive_count=len(construction.primitives),
        composition_count=len(construction.compositions),
        maximum_endpoint_projection_error_m=float(maximum_error),
        edge_level_primitives_unique=edge_unique,
        endpoint_compositions_complete=compositions_complete,
        mesh_triangle_provenance_present=mesh_provenance,
        lidar_hit_provenance_present=lidar_provenance,
        construction_supervision_complete=not reasons,
        failure_reasons=tuple(reasons),
    )


__all__ = [
    "ArchivedSupervisionAudit",
    "EndpointComposition",
    "NodeDegreeMismatch",
    "PrimitiveConstructionGraph",
    "PrimitiveEndpoint",
    "SweptPrimitive",
    "audit_archived_supervision",
    "build_primitive_construction_graph",
    "obj_has_face_provenance",
]
