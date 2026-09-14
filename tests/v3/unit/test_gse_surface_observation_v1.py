from dataclasses import replace
import pytest

from mtare_topo.semantics.gse_local_structure_v2 import SourceFrameV2
from mtare_topo.topology.gse_surface_observation_v1 import (
    SurfaceAnchorV1, SurfaceOpeningV1, SurfaceObservationV1,
)


def anchor(query=0, xyz=(0., 0., 0.)):
    return SurfaceAnchorV1(query, xyz, 1., (.01, .01, .01), True)


def opening(query=0, links=(0,), traversability="traversable"):
    return SurfaceOpeningV1(query, (10., 0., 0.), (1., 0., 0.), None, None, 1., True,
        (1., 0., 0.) if traversability == "traversable" else (0., 0., 1.), traversability,
        tuple(1. if i in links else 0. for i in range(32)), tuple(i in links for i in range(32)))


def observation(step=0, anchors=None, openings=None, stream="stream"):
    return SurfaceObservationV1(stream, step,
        tuple(SourceFrameV2(f"frame:{i}", i) for i in range(step, step + 5)), step + 4,
        "a" * 64, (anchor(),) if anchors is None else anchors,
        (opening(),) if openings is None else openings)


def test_independent_memberships_allow_multiple_anchors_and_unknown_sizes():
    value = observation(anchors=(anchor(0), anchor(1, (0., 0., 3.))), openings=(opening(links=(0, 1)),))
    assert sum(value.openings[0].anchor_relation_probability) == 2
    assert value.openings[0].width_m is value.openings[0].height_m is None


@pytest.mark.parametrize("probabilities", ((0., 0., 1.), (.5, .5, 0.), (.2, .7, .1)))
def test_predicted_traversability_cannot_contradict_unknown_blocked_or_tie(probabilities):
    with pytest.raises(ValueError, match="unique maximum"):
        replace(opening(), reachability_probabilities=probabilities)


def test_full32_64_capacity_and_unknown_geometry():
    obs = observation(anchors=tuple(anchor(i, (i, 0., 0.)) for i in range(32)),
                      openings=tuple(opening(i, (i % 32,)) for i in range(64)))
    assert len(obs.anchors) == 32 and len(obs.openings) == 64
    unknown = SurfaceAnchorV1(0, None, .5, None, False)
    op = SurfaceOpeningV1(0, None, None, None, None, .5, False, (0., 0., 1.), "unknown", (0.,) * 32, (False,) * 32)
    assert observation(anchors=(unknown,), openings=(op,)).openings[0].position_robot_m is None


@pytest.mark.parametrize("field,value", [("query_index", True), ("query_index", 32),
    ("position_robot_m", (float("nan"), 0., 0.)), ("position_robot_m", [0., 0., 0.]),
    ("uncertainty_m", (-1., 0., 0.)), ("existence_probability", True), ("numerically_supported", 1)])
def test_invalid_anchor(field, value):
    with pytest.raises(ValueError):
        replace(anchor(), **{field: value})


@pytest.mark.parametrize("field,value", [("query_index", 64), ("width_m", 0),
    ("height_m", float("inf")), ("direction_robot", (0., 0., 0.)),
    ("anchor_relation_valid", (1,) * 32), ("anchor_relation_probability", (0.,) * 33),
    ("reachability_probabilities", (.1, .1, .1)), ("traversability", "safe"),
    ("numerically_supported", False)])
def test_invalid_opening(field, value):
    with pytest.raises(ValueError):
        replace(opening(), **{field: value})


@pytest.mark.parametrize("case", ("future", "duplicate", "relation_missing", "query_duplicate", "wrongframe", "timestampbool", "hash"))
def test_observation_boundary(case):
    value = observation()
    with pytest.raises(ValueError):
        if case == "future":
            replace(value, current_source_index=3)
        elif case == "duplicate":
            replace(value, source_frames=(SourceFrameV2("dup", 0), SourceFrameV2("dup", 1), *value.source_frames[2:]))
        elif case == "relation_missing":
            replace(value, openings=(opening(links=(2,)),))
        elif case == "query_duplicate":
            replace(value, anchors=(anchor(), anchor()))
        elif case == "wrongframe":
            replace(value, coordinate_frame="world")
        elif case == "timestampbool":
            replace(value, timestamp_s=True)
        else:
            replace(value, input_binding_sha256="not-a-sha")
