from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from mtare_topo.data.primitive_frame_support import PrimitiveFrameSupport, summarize_primitive_frame_support, visible_primitive_targets_from_frame_support
from mtare_topo.data.primitive_relation_targets import primitive_relation_targets_from_frame_support
from mtare_topo.data.primitive_relation_storage import pack_primitive_relation_targets
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry
from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveEndpoint, EndpointComposition, PrimitiveConstructionGraph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, SweptSuperellipseProvenanceField
from mtare_topo.teacher.gse_supported_construction_teacher import window_supported_targets, construction_source_conflicts, build_task_targets


def fixture(degree=3):
    directions = [(1.,0.,0.), (-1.,0.,0.), (0.,1.,0.), (0.,-1.,0.)]
    operands = tuple(SweptSuperellipsePrimitive(f"p{i}", np.array([[0.,0.,0.], np.array(directions[i])*2]),
                    ((1.,1.),(1.,1.)), (2.,2.)) for i in range(degree))
    field = SweptSuperellipseProvenanceField(operands, spacing_m=.025)
    members = tuple(PrimitiveEndpoint(f"p{i}", 0, "node", (0.,0.,0.), (0.,0.,0.)) for i in range(degree))
    graph = PrimitiveConstructionGraph("world", (), (EndpointComposition("node", members, (0.,0.,0.)),))
    support = PrimitiveFrameSupport(np.full(degree, 3, np.int32), np.zeros(degree,np.int32), np.full(degree,80,np.int32), np.zeros((degree,90),np.uint8))
    return graph, field, (support,)*5


def generate(graph, field, supports, **kwargs):
    return window_supported_targets(construction=graph, field=field, frame_supports=supports,
        current_sensor_xyz_m=np.zeros(3), current_yaw_deg=0., **kwargs)


def test_supported_t_junction_and_full_candidate_membership():
    result = generate(*fixture())
    g = result["regions"][0]
    assert g["event_target"] == "junction" and g["center_valid"]
    assert sum(g["directional_member_valid"]) == 3
    assert [g["directional_member_target"][i] for i in (0,2,4)] == [1,1,1]
    assert not result["observation_label_complete"] and not result["physical_connectivity_certified"]


def test_occluded_third_arm_unknown_not_corridor():
    graph, field, supports = fixture()
    support = PrimitiveFrameSupport(np.array([3,3,0]), np.array([0,0,-1]), np.array([80,80,-1]), np.zeros((3,90),np.uint8))
    result = generate(graph, field, (support,)*5)["regions"][0]
    assert result["center_valid"] and result["event_target"] is None
    assert not result["event_valid"] and sum(result["directional_member_valid"]) == 2
    assert not result["members"][2]["supported"]


def test_two_supported_arms_only_corridor_if_construction_degree_two():
    assert generate(*fixture(2))["regions"][0]["event_target"] == "corridor"


def test_terminal_without_cap_return_kept_unknown_even_interval_reaches_end():
    result = generate(*fixture(1))["regions"][0]
    assert result["center_valid"] and not result["event_valid"]
    assert result["event_unknown_reason"] == "TERMINAL_CAP_RETURN_NOT_VERIFIED"


def test_no_local_support_does_not_invent_center():
    graph, field, supports = fixture()
    support = replace(supports[0], minimum_sample_index=np.full(3,10))
    result = generate(graph, field, (support,)*5)["regions"][0]
    assert not result["center_valid"] and not any(result["directional_member_valid"])


def test_interval_gap_is_explicit_proxy_not_continuous_observation_claim():
    graph, field, supports = fixture(1)
    graph = replace(graph, compositions=(replace(graph.compositions[0], anchor_xyz_m=(1.,0.,0.)),))
    low = replace(supports[0], minimum_sample_index=np.array([0]), maximum_sample_index=np.array([0]))
    high = replace(supports[0], minimum_sample_index=np.array([80]), maximum_sample_index=np.array([80]))
    g = generate(graph, field, (low,low,low,high,high))["regions"][0]
    assert g["center_valid"] and g["interval_proxy_not_continuous_visibility_proof"]
    assert g["members"][0]["projected_native_axis_index"] == 40


def test_numeric_equal_projection_abstains():
    graph, field, supports = fixture(1)
    graph = replace(graph, compositions=(replace(graph.compositions[0], anchor_xyz_m=(.0125,0.,0.)),))
    g = generate(graph, field, supports)["regions"][0]
    assert not g["center_valid"]
    assert "ANCHOR_PROJECTION_NUMERIC_TIE" in g["members"][0]["unknown_reasons"]


def test_two_instances_negative_only_for_supported_other_instance_direction():
    graph, field, supports = fixture(4)
    first = replace(graph.compositions[0], member_endpoints=graph.compositions[0].member_endpoints[:2])
    second = EndpointComposition("second", tuple(replace(m,node_id="second") for m in graph.compositions[0].member_endpoints[2:]), (0.,0.,0.))
    graph = replace(graph, compositions=(first,second))
    result = generate(graph,field,supports)
    assert len(result["regions"]) == 2
    assert result["regions"][0]["directional_member_valid"][4]
    assert result["regions"][0]["directional_member_target"][4] == 0
    assert not result["regions"][0]["directional_member_valid"][5]


def test_source_nonincident_ambiguity_not_angular_overlap():
    graph, field, supports = fixture(4)
    first = replace(graph.compositions[0], member_endpoints=graph.compositions[0].member_endpoints[:2])
    second = EndpointComposition("second", tuple(replace(m,node_id="second") for m in graph.compositions[0].member_endpoints[2:]), (0.,0.,0.))
    graph = replace(graph, compositions=(first,second))
    assert construction_source_conflicts(graph, field.primitive_ids, [(0,1),(0,2)]) == ((0,2),)
    result = generate(graph,field,supports,nonincident_source_pairs=((0,2),))
    assert all(not x["event_valid"] for x in result["regions"])
    assert "NONINCIDENT_SOURCE_AMBIGUITY" in result["regions"][0]["members"][0]["unknown_reasons"]


def test_old_target_node_is_not_an_argument():
    assert "target_node" not in inspect.signature(window_supported_targets).parameters
    assert "target_node" not in inspect.signature(build_task_targets).parameters


def test_node_identity_rename_does_not_change_geometric_labels():
    graph, field, supports = fixture()
    renamed = replace(graph, compositions=(replace(graph.compositions[0], node_id="different",
                      member_endpoints=tuple(replace(m,node_id="different") for m in graph.compositions[0].member_endpoints)),))
    a, b = generate(graph,field,supports), generate(renamed,field,supports)
    del a["regions"][0]["construction_node_id_teacher_only"]; del b["regions"][0]["construction_node_id_teacher_only"]
    assert a == b


def test_duplicate_node_ids_rejected_even_with_disjoint_members():
    graph, field, supports = fixture(4)
    members = graph.compositions[0].member_endpoints
    first = EndpointComposition("node", members[:2], (0.,0.,0.))
    second = EndpointComposition("node", members[2:], (0.,0.,0.))
    with pytest.raises(ValueError, match="unique"):
        generate(replace(graph, compositions=(first, second)), field, supports)


def test_member_node_id_must_match_its_composition():
    graph, field, supports = fixture()
    members = (replace(graph.compositions[0].member_endpoints[0], node_id="other"),) + graph.compositions[0].member_endpoints[1:]
    with pytest.raises(ValueError, match="member node ownership"):
        generate(replace(graph, compositions=(replace(graph.compositions[0], member_endpoints=members),)), field, supports)


@pytest.mark.parametrize("degree", [0, 1, 4, True, 3.0])
def test_declared_object_degree_cannot_disagree_with_incidence(degree):
    graph, field, supports = fixture()
    old = graph.compositions[0]
    malformed = SimpleNamespace(node_id=old.node_id, anchor_xyz_m=old.anchor_xyz_m,
                                member_endpoints=old.member_endpoints, degree=degree)
    with pytest.raises(ValueError, match="degree"):
        generate(replace(graph, compositions=(malformed,)), field, supports)


def test_invisible_bad_composition_cannot_escape_ownership_validation():
    graph, field, supports = fixture(2)
    first = EndpointComposition("node", graph.compositions[0].member_endpoints[:1], (0.,0.,0.))
    second = EndpointComposition("other", graph.compositions[0].member_endpoints[1:], (0.,0.,0.))
    support = PrimitiveFrameSupport(np.array([3,0]), np.array([0,-1]), np.array([80,-1]), np.zeros((2,90),np.uint8))
    with pytest.raises(ValueError, match="member node ownership"):
        generate(replace(graph, compositions=(first, second)), field, (support,)*5)


def test_empty_invisible_composition_is_not_silently_dropped():
    graph, field, supports = fixture()
    with pytest.raises(ValueError, match="degree"):
        generate(replace(graph, compositions=graph.compositions + (EndpointComposition("empty", (), (0.,0.,0.)),)), field, supports)


def test_same_primitive_two_endpoints_remain_independent_incidence():
    graph, field, supports = fixture(1)
    first = graph.compositions[0]
    second = EndpointComposition("other", (PrimitiveEndpoint("p0",1,"other",(2.,0.,0.),(2.,0.,0.)),), (2.,0.,0.))
    rows = generate(replace(graph, compositions=(first,second)), field, supports)["regions"]
    assert len(rows) == 2
    assert rows[0]["members"][0]["teacher_direction_token"] == 0
    assert rows[1]["members"][0]["teacher_direction_token"] == 1


@pytest.mark.parametrize("error", ["missing_anchor", "duplicate_owner", "four_frames", "field_spacing", "support_range"])
def test_invalid_contract_refused(error):
    graph, field, supports = fixture()
    if error == "missing_anchor": graph = replace(graph,compositions=(replace(graph.compositions[0],anchor_xyz_m=None),))
    if error == "duplicate_owner": graph = replace(graph,compositions=graph.compositions*2)
    if error == "four_frames": supports = supports[:4]
    if error == "field_spacing": field = SweptSuperellipseProvenanceField(tuple(x.primitive for x in field.operands), spacing_m=.05)
    if error == "support_range": supports = (replace(supports[0],maximum_sample_index=np.full(3,999)),)*5
    with pytest.raises(ValueError): generate(graph,field,supports)


def task_fixture():
    graph, field, _ = fixture(1)
    sensor = {"range_m":np.full((5,16,720),50.,np.float32), "valid_mask":np.zeros((5,16,720),np.uint8),
              "primitive_membership_code":np.zeros((5,16,720),np.uint16), "sensor_xyz_m":np.zeros((5,3)), "yaw_deg":np.zeros(5)}
    sensor["range_m"][:,8,0] = 1.; sensor["valid_mask"][:,8,0] = 1; sensor["primitive_membership_code"][:,8,0] = 1
    sources = ((),(0,))
    supports = tuple(summarize_primitive_frame_support(range_m=sensor["range_m"][i],primitive_membership_code=sensor["primitive_membership_code"][i],
        source_sets=sources,field=field,sensor_xyz_m=sensor["sensor_xyz_m"][i],yaw_deg=0.) for i in range(5))
    visible = visible_primitive_targets_from_frame_support(field=field,frame_supports=supports,current_sensor_xyz_m=np.zeros(3),current_yaw_deg=0.,maximum_slots=32)
    relation = primitive_relation_targets_from_frame_support(construction=graph,primitive_ids=field.primitive_ids,frame_supports=supports,maximum_slots=32)
    packed = pack_primitive_relation_targets(relation)
    teacher = {"primitive_index":visible.primitive_index[None],"primitive_mask":visible.mask[None],"frame_row":np.arange(5)[None],"source_global_sequence_index":np.array([9])}
    for k in ("axis_control_current_sensor_m","endpoint_half_axes_m","endpoint_shape_exponent","support_ray_count","temporal_visibility"):
        teacher[k] = getattr(visible,k)[None]
    for k in ("endpoint_neighbor","disconnected_overlap_packed"): teacher[k] = getattr(packed,k)[None]
    odometry = causal_relative_odometry(sensor["sensor_xyz_m"],sensor["yaw_deg"])
    teacher["relative_translation_current_sensor_m"] = odometry.translation_current_sensor_m[None]
    teacher["relative_yaw_current_sensor_deg"] = odometry.yaw_current_sensor_deg[None]
    base = graph.as_dict(); base["primitives"] = [{"primitive_id":"p0"}]
    construction = {"schema_version":"primitive_relation_realized_construction_v1", "base_construction":base,"realized_primitives":[x.primitive.as_dict() for x in field.operands]}
    return dict(teacher=teacher,sensor=sensor,sensor_frame_rows=np.arange(5),construction=construction,codebook={"primitive_ids":["p0"],"source_sets":[[],[0]]})


def test_full_task_reconstructs_old_targets_from_raw_without_models():
    result = build_task_targets(**task_fixture())
    assert result["old_teacher_all_fields_reconstructed_exactly"]
    assert result["counts"]["unique_sensor_frames"] == 5 and result["counts"]["visible_fragments"] == 1
    assert not result["capacity_ready"] and result["counts"]["events"]["terminal"] == 0


@pytest.mark.parametrize("degree", [2, 0, -1, True, 1.0, None])
def test_json_degree_drift_rejected_before_loader_discards_field(degree):
    value = task_fixture()
    record = value["construction"]["base_construction"]["composition_operations"][0]
    if degree is None:
        del record["degree"]
    else:
        record["degree"] = degree
    with pytest.raises(ValueError, match="JSON degree"):
        build_task_targets(**value)


@pytest.mark.parametrize("key", ["axis_control_current_sensor_m", "support_ray_count", "relative_translation_current_sensor_m"])
def test_old_target_drift_is_not_silently_repaired(key):
    value = task_fixture(); value["teacher"][key] = value["teacher"][key] + 1
    with pytest.raises(ValueError,match="differs"): build_task_targets(**value)


def test_causal_rows_and_valid_source_parity():
    value = task_fixture(); value["sensor"]["valid_mask"][0,8,0] = 0
    with pytest.raises(ValueError,match="source-code"): build_task_targets(**value)
    value = task_fixture(); value["teacher"]["frame_row"][0] = [0,1,2,4,3]
    with pytest.raises(ValueError,match="noncausal"): build_task_targets(**value)


@pytest.mark.parametrize("fault", ["negative_range", "nonbinary_valid", "float_codebook", "unsigned_reverse", "duplicate_source", "duplicate_sequence"])
def test_raw_entry_rejects_invalid_contract_before_target_generation(fault):
    value = task_fixture()
    if fault == "negative_range": value["sensor"]["range_m"][0,8,0] = -4
    if fault == "nonbinary_valid": value["sensor"]["valid_mask"][0,8,0] = 2
    if fault == "float_codebook": value["codebook"]["source_sets"][1] = [.9]
    if fault == "unsigned_reverse": value["teacher"]["frame_row"] = np.array([[4,3,2,1,0]],np.uint32)
    if fault == "duplicate_source": value["codebook"]["source_sets"].append([0])
    if fault == "duplicate_sequence": value["teacher"]["source_global_sequence_index"] = np.array([9.])
    with pytest.raises(ValueError): build_task_targets(**value)
