"""Load the immutable P1a construction documents for P1b materialization."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition,
    PrimitiveConstructionGraph,
    PrimitiveEndpoint,
)
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive


def load_p1a_realized_construction(
    document: Mapping[str, Any],
) -> tuple[PrimitiveConstructionGraph, tuple[SweptSuperellipsePrimitive, ...]]:
    """Decode only the relation-bearing portion of a sealed P1a document.

    P1b does not reconstruct topology from a world file.  It consumes the
    exact construction and realized operands that P1a placed beside each
    sensor shard, preserving primitive ordering and endpoint identity.
    """

    if document.get("schema_version") != "primitive_relation_realized_construction_v1":
        raise ValueError("unsupported P1a construction schema")
    base = document.get("base_construction")
    realized_rows = document.get("realized_primitives")
    if not isinstance(base, Mapping) or not isinstance(realized_rows, list):
        raise ValueError("P1a construction document is incomplete")
    if base.get("schema_version") != "primitive_construction_graph_v1":
        raise ValueError("unsupported base construction schema")

    compositions: list[EndpointComposition] = []
    for row in base.get("composition_operations", ()):
        endpoints = tuple(
            PrimitiveEndpoint(
                primitive_id=str(value["primitive_id"]),
                endpoint_index=int(value["endpoint_index"]),
                node_id=str(value["node_id"]),
                xyz_m=tuple(float(x) for x in value["xyz_m"]),
                composition_anchor_xyz_m=tuple(
                    float(x) for x in value.get("composition_anchor_xyz_m", value["xyz_m"])
                ),
            )
            for value in row["member_endpoints"]
        )
        anchor = row.get("anchor_xyz_m")
        compositions.append(
            EndpointComposition(
                node_id=str(row["node_id"]),
                member_endpoints=endpoints,
                anchor_xyz_m=None if anchor is None else tuple(float(x) for x in anchor),
            )
        )

    realized = tuple(
        SweptSuperellipsePrimitive(
            primitive_id=str(row["primitive_id"]),
            centerline_xyz_m=np.asarray(row["centerline_xyz_m"], dtype=np.float64),
            endpoint_half_axes_m=tuple(tuple(float(x) for x in pair) for pair in row["endpoint_half_axes_m"]),
            endpoint_shape_exponent=tuple(float(x) for x in row["endpoint_shape_exponent"]),
        )
        for row in realized_rows
    )
    identities = tuple(value.primitive_id for value in realized)
    expected = tuple(str(row["primitive_id"]) for row in base.get("primitives", ()))
    if not identities or identities != expected or len(identities) != len(set(identities)):
        raise ValueError("realized primitive identity/order differs from base construction")
    referenced = {
        endpoint.primitive_id
        for composition in compositions
        for endpoint in composition.member_endpoints
    }
    if not referenced.issubset(set(identities)):
        raise ValueError("composition references an absent realized primitive")

    construction = PrimitiveConstructionGraph(
        coordinate_frame=str(base.get("coordinate_frame", "world")),
        primitives=(),
        compositions=tuple(compositions),
        endpoint_attachment_mode=str(base.get("endpoint_attachment_mode", "free_space_overlap")),
        node_degree_source=str(base.get("node_degree_source", "edge_incidence")),
    )
    return construction, realized


__all__ = ["load_p1a_realized_construction"]
