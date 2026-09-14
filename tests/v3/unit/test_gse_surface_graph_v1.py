"""Ideal class-free software fixtures, not a dataset/teacher qualification."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest

from mtare_topo.semantics.gse_local_structure_v2 import SourceFrameV2
from mtare_topo.topology.gse_surface_graph_v1 import (
    SurfaceGraphConfigV1, KnownSurfacePoseV1, SurfaceSegmentedGraphV1, SurfaceExecutionStartV1,
)
from mtare_topo.topology.gse_segmented_graph_v1 import SegmentCoordinate
from tests.v3.unit.test_gse_surface_observation_v1 import anchor, opening, observation


IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def graph(stable=2):
    # Fixture values only. Neither1m arrival nor.25m tracking is calibrated.
    g = SurfaceSegmentedGraphV1(SurfaceGraphConfigV1(stable, .8, .5, .25, 1., .8, .8))
    g.begin_segment("segment:a", runtime_stream_key="stream", coordinate_kind="order_index")
    return g


def pose(step, xyz=(0., 0., 0.), rotation=IDENTITY, reference="map"):
    return KnownSurfacePoseV1(SegmentCoordinate("order_index", step), reference, rotation, tuple(float(v) for v in xyz))


def feed(g, step, robot, anchors=((0., 0., 0.),), count=1, stream="stream", continuous=True, queries=None):
    indices = range(len(anchors)) if queries is None else queries
    candidates = tuple(anchor(i, tuple(a - b for a, b in zip(xyz, robot))) for i, xyz in zip(indices, anchors))
    obs = observation(step, candidates, tuple(opening(i, tuple(indices)) for i in range(count)), stream)
    return g.update(obs, pose(step, robot), continuous=continuous)


def execution(g, query=0):
    obs = g._last_observation  # Synthetic receipt fixture, not a controller binding.
    return SurfaceExecutionStartV1(f"fixture-execution:{obs.runtime_stream_key}:{obs.decision_index}:{query}",
        obs.runtime_stream_key, obs.decision_index, query, obs.input_binding_sha256)


def depart(g, query=0):
    return g.depart(query, execution_start=execution(g, query))


def test_ideal_t_moving_boundary2_3_2_creates_one_fixed_anchor():
    g = graph()
    for i, (x, count) in enumerate(((-12., 2), (-8., 3), (0., 3), (8., 3), (12., 2))):
        feed(g, i, (x, 0., 0.), count=count)
    state = g.snapshot()
    assert len(state["nodes"]) == 1 and state["nodes"][0]["xyz_m"] == (0., 0., 0.)
    assert state["edges"] == []
    assert [r["opening_count"] for r in state["decision_trace"] if r["action"] == "observe"] == [2, 3, 3, 3, 2]


def test_roi_boundaries_or_closed_end_not_inferred_from_opening_count():
    g = graph()
    for i, count in enumerate((2, 3, 2, 0)):
        feed(g, i, (i, 0., 0.), anchors=(), count=count)  # No anchors; no false terminal even zero openings.
    assert not g.snapshot()["nodes"]
    closed = graph()
    feed(closed, 0, (0., 0., 0.), count=0); feed(closed, 1, (0., 0., 0.), count=0)
    assert len(closed.snapshot()["nodes"]) == 1
    with pytest.raises(ValueError):
        depart(closed)


def test_stacked_xy_and_multiple_visible_anchors_never_imply_edges():
    g = graph()
    positions = ((0., 0., 0.), (0., 0., 3.), (8., 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); feed(g, 1, (0., 0., 0.), positions)
    state = g.snapshot()
    assert len(state["nodes"]) == 3 and not state["edges"]
    assert state["nodes"][state["current_node"]]["xyz_m"] == (0., 0., 0.)


def abc(g):
    positions = ((0., 0., 0.), (5., 0., 0.), (10., 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); feed(g, 1, (0., 0., 0.), positions)
    depart(g)
    feed(g, 2, (2., 0., 0.), positions); feed(g, 3, (5., 0., 0.), positions)
    depart(g)
    feed(g, 4, (7., 0., 0.), positions); feed(g, 5, (10., 0., 0.), positions)


def test_abc_all_adjacent_edges_and_explicit_departure_states():
    g = graph(); abc(g)
    state = g.snapshot()
    assert [(e["from"], e["to"], e["length_m"]) for e in state["edges"]] == [(0, 1, 5.), (1, 2, 5.)]
    assert state["departure_states"] == {0: "traversed", 1: "traversed"}
    assert all(e["segment_id"] == "segment:a" for e in state["edges"])


def test_segment_end_abandons_pending_and_does_not_merge_same_position_next_visit():
    g = graph()
    feed(g, 0, (0., 0., 0.)); feed(g, 1, (0., 0., 0.)); depart(g)
    feed(g, 2, (3., 0., 0.))
    g.end_segment()
    g.begin_segment("segment:b", runtime_stream_key="stream:b", coordinate_kind="order_index")
    feed(g, 0, (0., 0., 0.), stream="stream:b"); feed(g, 1, (0., 0., 0.), stream="stream:b")
    state = g.snapshot()
    assert len(state["nodes"]) == 2 and not state["edges"]
    assert state["departure_states"] == {0: "abandoned_segment_end"}
    assert state["cross_visit_merge_implemented"] is False


def test_prefix_is_immutable_and_snapshot_mutation_cannot_rewrite_state():
    short, full = graph(), graph()
    for g in (short, full):
        feed(g, 0, (0., 0., 0.)); feed(g, 1, (0., 0., 0.))
    prefix, trace = short.snapshot(), short.immutable_trace
    assert prefix == full.snapshot()
    feed(full, 2, (1., 0., 0.)); full.end_segment()
    assert full.immutable_trace[:len(trace)] == trace
    prefix["nodes"][0]["xyz_m"] = (999., 0., 0.)
    assert short.snapshot()["nodes"][0]["xyz_m"] == (0., 0., 0.)


def test_query_permutation_does_not_supply_place_identity():
    g = graph()
    positions = ((0., 0., 0.), (0., 0., 3.))
    feed(g, 0, (0., 0., 0.), positions, queries=(3, 7))
    feed(g, 1, (0., 0., 0.), positions[::-1], queries=(22, 1))
    assert len(g.snapshot()["nodes"]) == 2


def test_ambiguous_close_anchors_are_not_forced_into_one_identity():
    g = graph()
    for step in range(3):
        feed(g, step, (0., 0., 0.), ((0., 0., 0.), (.1, 0., 0.)))
    assert not g.snapshot()["nodes"]
    assert all(d["action"] == "pending_ambiguous_short_track" for d in g.snapshot()["decision_trace"][-1]["anchor_decisions"])


def test_full_se3_pose_transforms_anchor_not_just_robot_position():
    # Nonzero roll,pitch,yaw proper rotation from explicit composition.
    a, b, c = .3, -.2, .7
    rx = np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    ry = np.array([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]])
    rz = np.array([[math.cos(c),-math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
    r = rz @ ry @ rx; origin = np.array([5., -2., 3.]); target = np.array([1., 4., 7.])
    g = graph(); local = tuple(float(x) for x in r.T @ (target - origin))
    for step in range(2):
        g.update(observation(step, (anchor(0,local),), ()),
                 pose(step, origin, tuple(tuple(float(x) for x in row) for row in r)))
    np.testing.assert_allclose(g.snapshot()["nodes"][0]["xyz_m"], target, atol=1e-12)


def test_discontinuous_trace_is_rejected_not_bridged():
    g = graph(stable=1)
    positions = ((0., 0., 0.), (5., 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); depart(g)
    feed(g, 1, (5., 0., 0.), positions, continuous=False)
    assert not g.snapshot()["edges"]
    assert g.snapshot()["departure_states"][0] == "failed_discontinuous_or_zero_length"


def test_unknown_frame_expires_short_track_and_cannot_fake_revisit_merge():
    g = graph()
    feed(g, 0, (0.,0.,0.)); feed(g, 1, (0.,0.,0.))
    feed(g, 2, (0.,0.,0.), anchors=(), count=0)
    feed(g, 3, (0.,0.,0.)); feed(g, 4, (0.,0.,0.))
    assert len(g.snapshot()["nodes"]) == 2


@pytest.mark.parametrize("case", ("before_begin", "repeat_segment", "step", "wrong_reference", "source_remap", "duplicate_source_order", "wrong_stream", "pose_binding"))
def test_invalid_binding_has_no_partial_mutation(case):
    g = graph(); feed(g, 0, (0.,0.,0.)); before = g.snapshot()
    with pytest.raises(ValueError):
        if case == "before_begin":
            g.end_segment(); before = g.snapshot(); feed(g, 1, (0.,0.,0.))
        elif case == "repeat_segment":
            g.end_segment(); before = g.snapshot()
            g.begin_segment("segment:a", runtime_stream_key="stream", coordinate_kind="order_index")
        elif case == "step":
            feed(g, 0, (0.,0.,0.))
        elif case == "wrong_reference":
            g.update(observation(1), pose(1, reference="another_map"))
        elif case == "source_remap":
            obs = observation(1)
            g.update(replace(obs, source_frames=(SourceFrameV2("frame:0", 1), *obs.source_frames[1:])), pose(1))
        elif case == "duplicate_source_order":
            obs = observation(1)
            g.update(replace(obs, source_frames=(SourceFrameV2("other", 1), *obs.source_frames[1:])), pose(1))
        elif case == "wrong_stream":
            g.update(observation(1, stream="other"), pose(1))
        else:
            g.update(observation(1), pose(2))
    assert g.snapshot() == before


def test_actual_seconds_must_be_explicit_and_order_does_not_become_seconds():
    g = SurfaceSegmentedGraphV1(graph().config)
    g.begin_segment("seconds", runtime_stream_key="stream", coordinate_kind="seconds")
    with pytest.raises(ValueError):
        g.update(observation(), KnownSurfacePoseV1(SegmentCoordinate("seconds", 10.), "map", IDENTITY, (0.,0.,0.)))
    g.update(replace(observation(), timestamp_s=10.), KnownSurfacePoseV1(SegmentCoordinate("seconds", 10.), "map", IDENTITY, (0.,0.,0.)))
    assert g.snapshot()["time_metrics_available"] is False
    assert "timestamp_s" not in json.dumps(g.snapshot())


@pytest.mark.parametrize("rotation", (((True,False,False),(False,True,False),(False,False,True)),
    ((1.,0.,0.),(0.,1.,0.),(0.,0.,-1.))))
def test_bool_rotation_and_reflection_rejected(rotation):
    with pytest.raises(ValueError):
        pose(0, rotation=rotation)


def test_unknown_opening_cannot_authorize_departure():
    g = graph()
    for step in range(2):
        g.update(observation(step, openings=(opening(traversability="unknown"),)), pose(step))
    before = g.snapshot()
    with pytest.raises(ValueError):
        g.depart(0)
    assert g.snapshot() == before


def test_overlapping_arrival_regions_do_not_authorize_stale_current_node_departure():
    g = graph()
    positions = ((0., 0., 0.), (1.5, 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); feed(g, 1, (0., 0., 0.), positions)
    decision = feed(g, 2, (.75, 0., 0.), positions)
    assert decision["arrival_action"] == "arrival_ambiguous_no_edge"
    before = g.snapshot()
    with pytest.raises(ValueError, match="unique currently observed arrival"):
        depart(g)
    assert g.snapshot() == before


def test_future_timestamp_rejected_before_any_graph_or_ledger_change():
    g = SurfaceSegmentedGraphV1(graph().config)
    g.begin_segment("actual-clock", runtime_stream_key="stream", coordinate_kind="seconds")
    before = g.snapshot()
    with pytest.raises(ValueError, match="actual observation timestamp"):
        g.update(replace(observation(), timestamp_s=11.),
                 KnownSurfacePoseV1(SegmentCoordinate("seconds", 10.), "map", IDENTITY, (0., 0., 0.)))
    assert g.snapshot() == before


def test_sparse_polyline_cannot_skip_known_intermediate_anchor():
    g = graph(stable=1)
    positions = ((0., 0., 0.), (5., 0., 0.), (10., 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); depart(g)
    prefix = g.immutable_trace
    for step, x in enumerate((3., 7., 10.), 1):
        feed(g, step, (x, 0., 0.), positions)
    state = g.snapshot()
    assert state["edges"] == []
    assert state["departure_states"] == {0: "unresolved_intermediate_anchor"}
    trace = state["unresolved_traces"][0]
    assert trace["intermediate_candidate_nodes"] == [1]
    assert [p["xyz_m"][0] for p in trace["poses"]] == [0., 3., 7., 10.]
    assert g.immutable_trace[:len(prefix)] == prefix


def test_polyline_does_not_flatten_stacked_anchor_into_intermediate():
    g = graph(stable=1)
    positions = ((0., 0., 0.), (5., 0., 3.), (10., 0., 0.))
    feed(g, 0, (0., 0., 0.), positions); depart(g)
    feed(g, 1, (10., 0., 0.), positions)
    assert [(e["from"], e["to"]) for e in g.snapshot()["edges"]] == [(0, 2)]


def test_unknown_prediction_can_record_independently_started_executed_trace():
    g = graph(stable=1)
    positions = ((0., 0., 0.), (5., 0., 0.))
    candidates = tuple(anchor(i, p) for i, p in enumerate(positions))
    unknown = replace(opening(links=(0, 1), traversability="unknown"), reachability_probabilities=(0., 0., 1.))
    g.update(observation(0, candidates, (unknown,)), pose(0))
    with pytest.raises(ValueError, match="external execution"):
        g.depart(0)
    depart(g)
    feed(g, 1, (5., 0., 0.), positions)
    edge = g.snapshot()["edges"][0]
    assert edge["departure_opening"]["traversability"] == "unknown"
    assert edge["departure_opening"]["reachability_probabilities"] == (0., 0., 1.)
    assert edge["external_execution_start"]["execution_key"].startswith("fixture-execution:")
    assert edge["kind"] == "recorded_traversed"


@pytest.mark.parametrize("field,value", [("decision_index", 99), ("runtime_stream_key", "other"),
    ("opening_query_index", 1), ("input_binding_sha256", "f" * 64)])
def test_execution_receipt_cannot_be_rebound(field, value):
    g = graph(stable=1); feed(g, 0, (0., 0., 0.))
    before = g.snapshot()
    with pytest.raises(ValueError, match="source mismatch"):
        g.depart(0, execution_start=replace(execution(g), **{field: value}))
    assert g.snapshot() == before
