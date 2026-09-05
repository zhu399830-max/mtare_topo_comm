"""Synthetic decision-free candidates; zero real observations or training."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import inspect

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes
from mtare_topo.topology.gse_region_candidates import (
    ObservationStamp, SensorToRobotExtrinsic, region_candidates_from_prediction,
)


def fixture():
    axes = torch.zeros((1, 32, 3, 3))
    for slot in range(31):
        axes[0, slot] = torch.tensor([[slot, 0., 0.], [slot, 1., 0.], [slot, 2., 0.]])
    valid = tokens_from_axes(axes).valid
    p = RegionPrediction(
        torch.arange(192, dtype=torch.float32).reshape(1, 64, 3) / 10,
        torch.arange(192, dtype=torch.float32).reshape(1, 64, 3) - 50,
        torch.linspace(-100, 100, 64)[None],
        torch.arange(4096, dtype=torch.float32).reshape(1, 64, 64) - 2000,
        torch.linspace(0, 1, 64)[None], valid, valid[:, :, None] & valid[:, None])
    stamp = ObservationStamp(3., "lidar", "base_link")
    # Nonzero roll/yaw and translation; not a truth/world pose.
    rz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    transform = np.eye(4); transform[:3, :3] = rz @ rx; transform[:3, 3] = [2., 3., 4.]
    extrinsic = SensorToRobotExtrinsic("lidar", "base_link", transform, "synthetic-calibration-only")
    return p, axes, stamp, extrinsic


def test_all_64_candidates_tokens_scores_and_unknowns_preserved():
    p, axes, stamp, extrinsic = fixture()
    out = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    assert len(out.candidates) == len(out.direction_tokens) == 64
    assert [q.observation_local_query_id for q in out.candidates] == list(range(64))
    assert [t.observation_local_token_id for t in out.direction_tokens] == list(range(64))
    assert out.stamp == stamp and out.extrinsic == extrinsic and out.decisions_made is False
    for i, candidate in enumerate(out.candidates):
        assert candidate.event_logits == tuple(p.event_logits[0, i].tolist())
        assert candidate.presence_logit == float(p.presence_logits[0, i])
        assert candidate.membership_logits == tuple(p.membership_logits[0, i].tolist())
        assert candidate.uncertainty_uncalibrated == float(p.uncertainty[0, i])
        assert candidate.numerical_query_supported == bool(p.query_supported[0, i])
        assert candidate.numerical_member_supported == tuple(p.member_supported[0, i].tolist())
    assert all(t.opening_width_m is None and t.opening_height_m is None for t in out.direction_tokens)
    assert all(t.direction_robot is None and not t.numerical_direction_supported for t in out.direction_tokens[-2:])
    assert all(t.direction_robot is not None for t in out.direction_tokens[:-2])


def test_coordinates_use_translation_for_centers_and_anchors_but_not_directions():
    p, axes, stamp, extrinsic = fixture()
    out = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    matrix = np.asarray(extrinsic.transformation_robot_from_sensor)
    tokens = tokens_from_axes(axes)
    expected_centers = p.centers_m[0].numpy() @ matrix[:3, :3].T + matrix[:3, 3]
    expected_anchors = tokens.positions_m[0].numpy() @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_array_equal([c.center_robot_m for c in out.candidates], expected_centers)
    np.testing.assert_array_equal([t.anchor_robot_m for t in out.direction_tokens], expected_anchors)
    np.testing.assert_array_equal(out.direction_tokens[0].direction_robot, matrix[:3, :3] @ tokens.tangents[0, 0].numpy())
    assert np.linalg.norm(out.direction_tokens[0].direction_robot) == pytest.approx(1.)


def test_same_frame_duplicate_centers_and_events_never_merge_or_select():
    p, axes, stamp, extrinsic = fixture()
    p.centers_m[:] = 0
    p.event_logits[:] = torch.tensor([0., 100., 0.])
    out = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    assert len(out.candidates) == 64
    assert len({q.center_robot_m for q in out.candidates}) == 1
    assert out.candidates[0].presence_logit == -100.
    assert out.candidates[-1].presence_logit == 100.
    assert not out.decisions_made


def test_slot_query_and_member_permutation_only_reorders_observation_slots():
    p, axes, stamp, extrinsic = fixture()
    old = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    slots = torch.arange(31, -1, -1)
    tokens = (slots[:, None] * 2 + torch.arange(2)).flatten()
    changed = RegionPrediction(
        p.centers_m[:, tokens], p.event_logits[:, tokens], p.presence_logits[:, tokens],
        p.membership_logits[:, tokens][:, :, tokens], p.uncertainty[:, tokens],
        p.query_supported[:, tokens], p.member_supported[:, tokens][:, :, tokens])
    new = region_candidates_from_prediction(changed, axes[:, slots], stamp, extrinsic)
    for index, old_index in enumerate(tokens.tolist()):
        a, b = new.candidates[index], old.candidates[old_index]
        assert a.center_robot_m == b.center_robot_m and a.event_logits == b.event_logits
        assert a.membership_logits == tuple(b.membership_logits[j] for j in tokens.tolist())
        assert new.direction_tokens[index].direction_robot == old.direction_tokens[old_index].direction_robot


def test_rigid_robot_reference_change_is_coordinate_only():
    p, axes, stamp, extrinsic = fixture()
    old = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    change = np.eye(4); change[:3, :3] = np.diag([-1., -1., 1.]); change[:3, 3] = [5., -2., 1.]
    transformed = SensorToRobotExtrinsic("lidar", "base_link", change @ np.asarray(extrinsic.transformation_robot_from_sensor), "synthetic-changed-frame")
    new = region_candidates_from_prediction(p, axes, stamp, transformed)
    for a, b in zip(old.candidates, new.candidates):
        np.testing.assert_allclose(b.center_robot_m, change[:3, :3] @ a.center_robot_m + change[:3, 3])
        assert a.event_logits == b.event_logits and a.membership_logits == b.membership_logits
    for a, b in zip(old.direction_tokens, new.direction_tokens):
        if a.direction_robot is not None:
            np.testing.assert_allclose(b.direction_robot, change[:3, :3] @ a.direction_robot)
        else:
            assert b.direction_robot is None


def test_input_unchanged_output_immutable_copied_and_repeat_exact():
    p, axes, stamp, extrinsic = fixture()
    old = deepcopy(p); old_axes = axes.clone(); old_extrinsic = deepcopy(extrinsic)
    out = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    assert out == region_candidates_from_prediction(p, axes, stamp, extrinsic)
    assert all(torch.equal(getattr(p, f.name), getattr(old, f.name)) for f in fields(p))
    assert torch.equal(axes, old_axes) and extrinsic == old_extrinsic
    p.event_logits.fill_(0); axes.fill_(0)
    assert out.candidates[0].event_logits == tuple(old.event_logits[0, 0].tolist())
    with pytest.raises(FrozenInstanceError):
        out.candidates[0].presence_logit = 0


@pytest.mark.parametrize("field", ["centers_m", "event_logits", "presence_logits", "membership_logits", "uncertainty"])
def test_nonfinite_prediction_rejected_even_for_unsupported_query(field):
    p, axes, stamp, extrinsic = fixture()
    getattr(p, field).flatten()[-1] = float("nan")
    with pytest.raises(ValueError, match="finite prediction"):
        region_candidates_from_prediction(p, axes, stamp, extrinsic)


@pytest.mark.parametrize("issue", ["wrong_shape", "wrong_dtype", "integer_mask", "query_mask", "member_mask", "wrong_axes", "uncertainty"])
def test_prediction_axes_shapes_masks_and_range_must_agree(issue):
    p, axes, stamp, extrinsic = fixture()
    if issue == "wrong_shape": p = replace(p, membership_logits=p.membership_logits[:, :, :63])
    if issue == "wrong_dtype": p = replace(p, centers_m=p.centers_m.double())
    if issue == "integer_mask": p = replace(p, query_supported=p.query_supported.int())
    if issue == "query_mask": p = replace(p, query_supported=~p.query_supported)
    if issue == "member_mask": p = replace(p, member_supported=~p.member_supported)
    if issue == "wrong_axes": axes[0, 0] = 0
    if issue == "uncertainty": p.uncertainty[0, 0] = 1.1
    with pytest.raises(ValueError):
        region_candidates_from_prediction(p, axes, stamp, extrinsic)


@pytest.mark.parametrize("issue", ["reflection", "scale", "bottom_row", "nan", "shape"])
def test_extrinsic_must_be_proper_finite_se3(issue):
    matrix = np.eye(4)
    if issue == "reflection": matrix[0, 0] = -1
    if issue == "scale": matrix[0, 0] = 2
    if issue == "bottom_row": matrix[3, 0] = 1
    if issue == "nan": matrix[0, 3] = np.nan
    if issue == "shape": matrix = matrix[:3]
    with pytest.raises(ValueError):
        SensorToRobotExtrinsic("lidar", "base_link", matrix, "synthetic")


@pytest.mark.parametrize("time", [True, float("nan"), float("inf"), "3"])
def test_invalid_timestamp(time):
    with pytest.raises(ValueError):
        ObservationStamp(time, "lidar", "base_link")


def test_frame_mismatch_missing_calibration_and_multi_observation_refused():
    p, axes, stamp, extrinsic = fixture()
    with pytest.raises(ValueError, match="frame mismatch"):
        region_candidates_from_prediction(p, axes, replace(stamp, sensor_frame="wrong"), extrinsic)
    with pytest.raises(ValueError):
        replace(extrinsic, calibration_reference="")
    with pytest.raises(ValueError, match="one observation"):
        region_candidates_from_prediction(p, axes.repeat(2, 1, 1, 1), stamp, extrinsic)


def test_no_teacher_threshold_or_graph_interface_and_all_unsupported_kept():
    assert tuple(inspect.signature(region_candidates_from_prediction).parameters) == (
        "prediction", "axes_current_sensor_m", "observation_stamp", "robot_from_sensor")
    p, axes, stamp, extrinsic = fixture()
    axes.zero_(); valid = tokens_from_axes(axes).valid
    p = replace(p, query_supported=valid, member_supported=valid[:, :, None] & valid[:, None])
    out = region_candidates_from_prediction(p, axes, stamp, extrinsic)
    assert len(out.candidates) == len(out.direction_tokens) == 64
    assert all(not q.numerical_query_supported for q in out.candidates)
    assert all(t.direction_robot is None for t in out.direction_tokens)
    assert not out.decisions_made


def test_matching_numeric_masks_do_not_certify_forward_identity_or_prediction_time():
    p, axes, stamp, extrinsic = fixture()
    other_axes = axes.clone(); other_axes[:, :31, :, 0] += 10
    assert torch.equal(tokens_from_axes(axes).valid, tokens_from_axes(other_axes).valid)
    # The producer must bind the correct arrays and time. There is deliberately
    # no identity inference from same-shaped, nondegenerate geometry here.
    out = region_candidates_from_prediction(p, other_axes, replace(stamp, timestamp_s=9.), extrinsic)
    assert out.stamp.timestamp_s == 9. and len(out.candidates) == 64
