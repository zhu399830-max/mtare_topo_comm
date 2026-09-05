import numpy as np

from mtare_topo.data.c08_causal_replay import (
    frame_contract,
    objective_role,
    traversal_index_for_arcs,
    validate_m1d_checkpoint_identity,
)


def test_frozen_m1d_checkpoint_identity_uses_exact_trainer_mode():
    validate_m1d_checkpoint_identity({"seed": 0, "mode": "M1D"}, 0)
    for invalid in ({"seed": 0, "mode": "m1d"}, {"seed": 1, "mode": "M1D"}):
        try:
            validate_m1d_checkpoint_identity(invalid, 0)
        except ValueError as error:
            assert "checkpoint identity mismatch" in str(error)
        else:
            raise AssertionError(f"invalid checkpoint identity was accepted: {invalid}")


def test_traversal_boundaries_belong_to_new_traversal():
    traversals = [
        {"route_start_m": 0.0, "route_end_m": 3.0},
        {"route_start_m": 3.0, "route_end_m": 7.0},
    ]
    assert traversal_index_for_arcs([0.0, 2.0, 3.0, 6.0], traversals).tolist() == [0, 0, 1, 1]


def test_terminal_first_role_contract_and_tunnel_restriction():
    graph = {"nodes": [
        {"id": "j", "degree": 3, "xyz": [1, 0, 0], "incident_tunnel_ids": [1, 2, 3]},
        {"id": "t", "degree": 1, "xyz": [2, 0, 0], "incident_tunnel_ids": [2]},
    ]}
    assert objective_role([0, 0, 0], 2, graph)[0] == "terminal"
    assert objective_role([0, 0, 0], 1, graph)[0] == "junction"


def test_frame_contract_preserves_evaluator_identity_only():
    traversals = [{
        "route_start_m": 0.0, "route_end_m": 3.0, "edge_id": "e0", "tunnel_id": "1"
    }]
    graph = {"nodes": []}
    frames = frame_contract(
        xyz_m=np.array([[0, 0, 2], [2, 0, 2.0]]),
        tangent_world=np.array([[1, 0, 0], [1, 0, 0.0]]),
        route_arc_m=np.array([0.0, 2.0]), traversals=traversals, graph=graph,
        fta_distance_m=-1.2,
    )
    assert np.allclose(frames[0]["sensor_xyz_m"], [0, 0, 1.8])
    assert frames[1]["edge_id"] == "e0"
    assert frames[1]["objective_role"] == "interior"


def test_frame_contract_keeps_teacher_graph_and_sensor_coordinates_distinct():
    traversals = [{
        "route_start_m": 0.0, "route_end_m": 3.0, "edge_id": "e0", "tunnel_id": "1"
    }]
    graph = {"nodes": [{"id":"j","degree":3,"xyz":[0,0,0],"incident_tunnel_ids":[1]}]}
    frames = frame_contract(
        teacher_axis_xyz_m=np.array([[0.0,0.0,0.0]]),
        graph_axis_xyz_m=np.array([[0.1,0.0,0.0]]),
        sensor_xyz_m=np.array([[0.1,0.0,1.0]]),
        tangent_world=np.array([[1.0,0.0,0.0]]),
        route_arc_m=np.array([0.0]),traversals=traversals,graph=graph,fta_distance_m=-1.0,
    )
    np.testing.assert_array_equal(frames[0]["teacher_axis_xyz_m"],[0.0,0.0,0.0])
    np.testing.assert_array_equal(frames[0]["graph_axis_xyz_m"],[0.1,0.0,0.0])
    np.testing.assert_array_equal(frames[0]["axis_xyz_m"],frames[0]["graph_axis_xyz_m"])
    np.testing.assert_array_equal(frames[0]["sensor_xyz_m"],[0.1,0.0,1.0])
    assert frames[0]["objective_role"] == "junction"


def test_frame_contract_rejects_partial_or_mixed_coordinate_interfaces():
    common = dict(
        tangent_world=np.array([[1.0,0.0,0.0]]),route_arc_m=np.array([0.0]),
        traversals=[{"route_start_m":0.0,"route_end_m":1.0,"edge_id":"e","tunnel_id":1}],
        graph={"nodes":[]},fta_distance_m=-1.0,
    )
    try:
        frame_contract(
            xyz_m=np.array([[0.0,0.0,0.0]]),teacher_axis_xyz_m=np.array([[0.0,0.0,0.0]]),
            graph_axis_xyz_m=np.array([[0.0,0.0,0.0]]),sensor_xyz_m=np.array([[0.0,0.0,1.0]]),**common,
        )
    except ValueError as error:
        assert "either legacy" in str(error)
    else:
        raise AssertionError("mixed coordinate interface was accepted")
