import json

import numpy as np
import pytest

from mtare_topo.teacher.gse_construction_paths_v2 import construction_incident_paths
from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition, PrimitiveConstructionGraph, PrimitiveEndpoint, SweptPrimitive,
)


def document():
    endpoints = (
        PrimitiveEndpoint("p", 0, "a", (0., 0., 0.), (0., 1., 0.)),
        PrimitiveEndpoint("p", 1, "b", (2., 0., 0.), (2., 0., 0.)),
    )
    primitive = SweptPrimitive("p", "edge", "tunnel", np.array([[0., 0., 0.], [2., 0., 0.]]),
                               2., (1., 0.), endpoints)
    graph = PrimitiveConstructionGraph("cano_world", (primitive,), tuple(
        EndpointComposition(e.node_id, (e,), e.composition_anchor_xyz_m) for e in endpoints
    ), endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
    # Real producer serializer, not a hand-invented replacement base schema.
    return json.loads(json.dumps({
        "schema_version": "primitive_relation_realized_construction_v1",
        "base_construction": graph.as_dict(),
        "realized_primitives": [{"primitive_id": "p", "centerline_xyz_m": [[0, 0, 0], [2, 0, 0]],
                                 "endpoint_half_axes_m": [[2, 2], [2, 2]],
                                 "endpoint_shape_exponent": [2, 2]}],
    }))


def test_producer_serializer_roundtrip_preserves_offset_and_reverse():
    d = document()
    before = json.dumps(d, sort_keys=True)
    a, b = construction_incident_paths(d)
    np.testing.assert_array_equal(a["paths"][0]["points_world_m"], [[0, 1, 0], [0, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(b["paths"][0]["points_world_m"], [[2, 0, 0], [0, 0, 0]])
    assert a["paths"][0]["endpoint_key"] == ("p", 0)
    assert b["paths"][0]["endpoint_key"] == ("p", 1)
    assert json.dumps(d, sort_keys=True) == before
    assert not a["paths"][0]["points_world_m"].flags.writeable


@pytest.mark.parametrize("fault", ["frame", "alias", "missing_groups", "outer_schema", "base_schema",
                                  "realized_id", "degree", "endpoint_type", "endpoint_position",
                                  "anchor_position", "duplicate_incidence"])
def test_source_defects_are_not_silently_repaired(fault):
    d = document(); base = d["base_construction"]
    group = base["composition_operations"][0]; member = group["member_endpoints"][0]
    if fault == "frame": base["coordinate_frame"] = "world"
    elif fault == "alias": base["compositions"] = base.pop("composition_operations")
    elif fault == "missing_groups": del base["composition_operations"]
    elif fault == "outer_schema": d["schema_version"] = "invented"
    elif fault == "base_schema": base["schema_version"] = "invented"
    elif fault == "realized_id": d["realized_primitives"][0]["primitive_id"] = "other"
    elif fault == "degree": group["degree"] = 2
    elif fault == "endpoint_type": member["endpoint_index"] = "0"
    elif fault == "endpoint_position": member["xyz_m"][2] = 1
    elif fault == "anchor_position": member["composition_anchor_xyz_m"][2] = 1
    elif fault == "duplicate_incidence": group["member_endpoints"].append(dict(member)); group["degree"] = 2
    with pytest.raises(ValueError): construction_incident_paths(d)
