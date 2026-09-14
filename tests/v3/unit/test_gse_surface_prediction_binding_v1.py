"""Synthetic owned-forward integration; no trained weights or sensor datasets."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationModelV1
from mtare_topo.semantics.gse_local_structure_v2 import SourceFrameV2
from mtare_topo.topology.gse_surface_prediction_binding_v1 import (
    SurfacePredictionBindingV1, SurfaceFramePacketV1, SurfaceRigidTransformV1,
    SurfaceForwardReadoutV1, frame_payload_sha256,
)
from mtare_topo.topology.gse_segmented_graph_v1 import SegmentCoordinate
from tests.v3.unit.test_gse_surface_graph_v1 import graph, pose, IDENTITY


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


I = SurfaceRigidTransformV1(IDENTITY, (0., 0., 0.))


def packets(step=0, supported=True):
    result = []
    for slot in range(5):
        xyz = torch.tensor([[-.1, -.1, -.1], [.1, -.1, .1], [-.1, .1, .1], [.1, .1, -.1]])
        p = SurfaceFramePacketV1(SourceFrameV2(f"frame:{step + slot}", step + slot), xyz,
            torch.zeros(4, 128), torch.full((4,), supported, dtype=torch.bool), torch.arange(4), I, "")
        result.append(replace(p, payload_sha256=frame_payload_sha256(p)))
    return tuple(result)


def binder(model=None):
    torch.manual_seed(71)
    return SurfacePredictionBindingV1((SurfaceRelationModelV1("C") if model is None else model).eval(),
        SurfaceForwardReadoutV1(.8, .8, .8, .8, .5, 10., 4096,
            supervision_policy='synthetic_all_heads_v1'))


def forward(b, step=0, **overrides):
    p = packets(step)
    kwargs = dict(packets=p, runtime_stream_key="stream", decision_index=step,
        current_source=p[-1].source, pose=pose(step), robot_from_current_sensor=I)
    kwargs.update(overrides)
    return b.forward(**kwargs)


class IdealSyntheticModel(SurfaceRelationModelV1):
    """Deterministic synthetic outputs after real random-weight forward.

    This is a plumbing fixture, explicitly not learned accuracy evidence.
    Center depends on actual input; no identity, teacher, pose arguments.
    """
    def __init__(self):
        super().__init__("C")

    def forward(self, xyz, context, valid, patches, *, sensor_token_index):
        out = super().forward(xyz, context, valid, patches, sensor_token_index=sensor_token_index)
        self.seen_xyz = xyz
        self.seen_layout = sensor_token_index
        self.seen_relation = patches.relation
        center = out.anchor_position_m.clone()
        center[:, 0] = xyz.mean(1)
        center[:, 1] = xyz.mean(1) + xyz.new_tensor([0., 0., 3.])
        presence = torch.full_like(out.anchor_presence_logits, -20.)
        presence[:, :2] = 20.
        direction = torch.zeros_like(out.opening_direction); direction[..., 0] = 1.
        reach = torch.zeros_like(out.reachability_logits); reach[..., 2] = 20.
        return replace(out, anchor_position_m=center, anchor_presence_logits=presence,
            anchor_uncertainty_m=torch.full_like(out.anchor_uncertainty_m, .01),
            opening_position_m=torch.ones_like(out.opening_position_m),
            opening_presence_logits=torch.full_like(out.opening_presence_logits, 20.),
            opening_direction=direction, opening_direction_valid=torch.ones_like(out.opening_direction_valid),
            opening_dimensions_m=torch.full_like(out.opening_dimensions_m, 2.),
            opening_dimension_evidence_logits=torch.full_like(out.opening_dimension_evidence_logits, -20.),
            opening_support_logits=torch.full_like(out.opening_support_logits, 20.),
            reachability_logits=reach,
            membership_logits=torch.full_like(out.membership_logits, 2.),
            membership_validity_logits=torch.full_like(out.membership_validity_logits, 20.))


def test_real_model_owned_forward_keeps32_64_raw_logits_and_proof():
    b = binder(); value = forward(b)
    raw = json.loads(value.raw_prediction_json); proof = json.loads(value.proof_json)
    assert len(value.observation.anchors) == 32 and len(value.observation.openings) == 64
    assert np.asarray(raw["membership_probability"]).shape == (64, 32)
    assert np.asarray(raw["membership_validity_logits"]).shape == (64, 32)
    assert np.asarray(raw["opening_dimension_evidence_probability"]).shape == (64, 2)
    assert proof["teacher_input"] is proof["model_absolute_pose_input"] is False
    assert proof["upstream_file_encoder_provenance_authenticated"] is False
    assert len(proof["frame_payload_sha256"]) == 5
    assert proof["actual_model_inputs"]["xyz"]["shape"] == [1, 20, 3]
    assert value.proof_sha256 == hashlib.sha256(value.proof_json.encode()).hexdigest()
    assert value.observation.input_binding_sha256 == value.proof_sha256
    assert proof["raw_prediction_sha256"] == hashlib.sha256(value.raw_prediction_json.encode()).hexdigest()


def test_real_model_repeat_is_deterministic():
    b = binder()
    assert forward(b) == forward(b)


@pytest.mark.parametrize("substitution", ("points", "context", "source", "layout", "mask"))
def test_same_shape_or_mask_does_not_authenticate_changed_source(substitution):
    b = binder(); p = list(packets())
    replacements = {
        "points": {"points_frame_sensor_m": p[0].points_frame_sensor_m + .01},
        "context": {"frozen_point_context": p[0].frozen_point_context + 1.},
        "source": {"source": SourceFrameV2("different-file", 0)},
        "layout": {"local_sensor_token_index": p[0].local_sensor_token_index + 1},
        "mask": {"valid": ~p[0].valid},
    }
    p[0] = replace(p[0], **replacements[substitution])
    with pytest.raises(ValueError, match="provenance differ"):
        forward(b, packets=tuple(p))


def test_explicit_current_stamp_rejects_future_frame_even_with_valid_hash():
    b = binder(); p = list(packets())
    p[-1] = replace(p[-1], source=SourceFrameV2("future", 9))
    p[-1] = replace(p[-1], payload_sha256=frame_payload_sha256(p[-1]))
    with pytest.raises(ValueError, match="future/source substitution"):
        forward(b, packets=tuple(p))


def test_source_type_rejected_before_attribute_access():
    p = list(packets()); p[0] = replace(p[0], source=True)
    with pytest.raises(ValueError, match="typed frame identities"):
        forward(binder(), packets=tuple(p))


def test_old_output_cannot_be_rebound_to_new_pose_or_foreign_wrapper():
    b = binder(); result = forward(b); g = graph()
    for changed in (pose(1), pose(0, (1., 0., 0.)), pose(0, reference="other")):
        with pytest.raises(ValueError, match="old output"):
            b.update_graph(result, g, changed)
    with pytest.raises(ValueError, match="owned forward"):
        b.update_graph(replace(result), g, pose(0))
    with pytest.raises(ValueError, match="owned forward"):
        binder().update_graph(result, g, pose(0))
    assert g.snapshot()["nodes"] == []


def test_full_extrinsic_transforms_geometry_but_rotated_uncertainty_unknown():
    r = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))  # proper3D permutation, not XY-only.
    extrinsic = SurfaceRigidTransformV1(r, (2., 4., 6.))
    b = binder(IdealSyntheticModel()); result = forward(b, robot_from_current_sensor=extrinsic)
    obs = result.observation
    np.testing.assert_allclose(obs.anchors[0].position_robot_m, (2., 4., 6.))
    np.testing.assert_allclose(obs.anchors[1].position_robot_m, (5., 4., 6.))
    np.testing.assert_allclose(obs.openings[0].position_robot_m, (3., 5., 7.))
    assert obs.openings[0].direction_robot == (0., 1., 0.)
    assert obs.anchors[0].uncertainty_m is None
    assert json.loads(result.proof_json)["robot_uncertainty_available"] is False
    assert json.loads(result.raw_prediction_json)["anchor_uncertainty_m"][0][0] > 0.
    g = graph(stable=1); b.update_graph(result, g, pose(0))
    assert not g.snapshot()["nodes"]  # Unknown rotated scale must not be fabricated.


def test_five_frame_motion_alignment_and_own_copies_reach_actual_forward():
    model = IdealSyntheticModel(); b = binder(model); p = list(packets())
    p[0] = replace(p[0], current_sensor_from_frame=SurfaceRigidTransformV1(IDENTITY, (1., 2., 3.)))
    p[0] = replace(p[0], payload_sha256=frame_payload_sha256(p[0]))
    original = p[0].points_frame_sensor_m.clone()
    result = forward(b, packets=tuple(p))
    torch.testing.assert_close(model.seen_xyz[0, :4], original + original.new_tensor([1., 2., 3.]))
    assert model.seen_layout[0].tolist() == [i + 180 * s for s in range(5) for i in range(4)]
    before = model.seen_xyz.clone(); p[0].points_frame_sensor_m.fill_(999.)
    torch.testing.assert_close(model.seen_xyz, before)
    assert result.proof_sha256 == hashlib.sha256(result.proof_json.encode()).hexdigest()


def test_owned_forward_to_graph_keeps_stacked_nodes_unknown_and_no_fake_edges():
    b = binder(IdealSyntheticModel()); g = graph()
    for step in range(3):
        result = forward(b, step)
        b.update_graph(result, g, pose(step))
    snapshot = g.snapshot()
    assert len(snapshot["nodes"]) == 2
    assert [n["xyz_m"] for n in snapshot["nodes"]] == [(0., 0., 0.), (0., 0., 3.)]
    assert snapshot["edges"] == []
    assert all(o.traversability == "unknown" for o in result.observation.openings)
    assert all(o.width_m is o.height_m is None for o in result.observation.openings)
    assert sum(result.observation.openings[0].anchor_relation_valid) == 32  # Not forced single membership.
    with pytest.raises(ValueError, match="external execution"):
        g.depart(0)


def test_empty_actual_input_remains_unknown_despite_zero_logits_half_probability():
    b = binder(); result = forward(b, packets=packets(supported=False))
    assert len(result.observation.anchors) == 32 and len(result.observation.openings) == 64
    assert all(a.position_robot_m is a.uncertainty_m is None for a in result.observation.anchors)
    assert all(not o.numerically_supported and o.traversability == "unknown" and not any(o.anchor_relation_valid)
               for o in result.observation.openings)
    g = graph(stable=1); b.update_graph(result, g, pose(0))
    assert not g.snapshot()["nodes"] and not g.snapshot()["edges"]


def test_model_input_mutation_is_rejected():
    class Mutator(IdealSyntheticModel):
        def forward(self, xyz, *args, **kwargs):
            result = super().forward(xyz, *args, **kwargs)
            xyz.add_(1.)
            return result
    with pytest.raises(ValueError, match="mutated model state or owned input"):
        forward(binder(Mutator()))


def test_training_mode_is_not_silently_changed_or_run():
    b = binder(); b.model.train()
    with pytest.raises(ValueError, match="explicit eval mode"):
        forward(b)


def test_explicit_seconds_and_nonidentity_current_alignment_rejected():
    b = binder()
    with pytest.raises(ValueError, match="timestamp does not match"):
        forward(b, timestamp_s=11., pose=replace(pose(0), coordinate=SegmentCoordinate("seconds", 10.)))
    p = list(packets())
    p[-1] = replace(p[-1], current_sensor_from_frame=SurfaceRigidTransformV1(IDENTITY, (1., 0., 0.)))
    p[-1] = replace(p[-1], payload_sha256=frame_payload_sha256(p[-1]))
    with pytest.raises(ValueError, match="current frame relative transform must be identity"):
        forward(b, packets=tuple(p))


def ray_packets(points, supported=True):
    xyz = torch.tensor(points, dtype=torch.float32)
    result = []
    for p in packets():
        p = replace(p, points_frame_sensor_m=xyz.clone(), frozen_point_context=torch.zeros(len(xyz), 128),
            valid=torch.full((len(xyz),), supported, dtype=torch.bool), local_sensor_token_index=torch.arange(len(xyz)))
        result.append(replace(p, payload_sha256=frame_payload_sha256(p)))
    return tuple(result)


def test_actual_owned_ray_obstacle_enters_gap_input_not_traversability():
    model = IdealSyntheticModel(); b = binder(model)
    p = ray_packets([[.125, .125, .125], [1.125, .125, .125], [2.125, .125, .125]])
    result = forward(b, packets=p)
    evidence = json.loads(result.raw_prediction_json)["observed_ray_evidence"]
    centers = np.asarray(evidence["patch_centers_m"])
    left = int(np.argmin(centers[:, 0])); right = int(np.argmax(centers[:, 0]))
    slot = evidence["neighbor_index"][left].index(right)
    assert evidence["gap_counts"][left][slot][1] >= 1  # Interior measured hit, not either endpoint surface.
    assert evidence["gap_defined"][left][slot]
    np.testing.assert_allclose(model.seen_relation[0, left, slot, -3:].cpu(),
        evidence["gap_fractions"][left][slot], rtol=1e-6)
    assert evidence["physical_connectivity"] is False
    assert all(o.traversability == "unknown" for o in result.observation.openings)  # Fixture prediction unchanged.
    proof = json.loads(result.proof_json)
    expected = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert proof["observed_ray_evidence_sha256"] == expected
    assert proof["grid_source_geometry_sha256"] == evidence["grid_source_geometry_sha256"]
    assert proof["grid_content_sha256"] == evidence["grid_content_sha256"]


def test_undefined_short_gap_is_explicit_unknown_not_zero_free():
    model = IdealSyntheticModel(); b = binder(model)
    # Adjacent.25m cells but different.5m patches: no intervening cell.
    result = forward(b, packets=ray_packets([[.375, .125, .125], [.625, .125, .125]]))
    e = json.loads(result.raw_prediction_json)["observed_ray_evidence"]
    assert e["message_neighbor_valid"][0][0]
    assert not e["gap_defined"][0][0] and e["gap_fractions"][0][0] == [0., 0., 0.]
    assert e["actual_ray_relation_features"][0][0] == [0., 0., 1.]
    assert model.seen_relation[0, 0, 0, -3:].tolist() == [0., 0., 1.]


def test_no_return_support_never_clears_grid_and_same_observation_repeats():
    b = binder()
    p = ray_packets([[2.125, .125, .125]], supported=False)
    first = forward(b, packets=p); second = forward(b, packets=p)
    a = json.loads(first.raw_prediction_json)["observed_ray_evidence"]
    other = json.loads(second.raw_prediction_json)["observed_ray_evidence"]
    assert a == other and first.proof_sha256 == second.proof_sha256
    assert a["grid_state_cell_counts"] == [80 ** 3, 0, 0]
    assert a["first_return_count"] == 0 and a["ignored_ray_count"] == 5
    assert a["gap_counts"] == [] and a["actual_ray_relation_features"] == []


def test_fixed_ray_cube_does_not_silently_truncate_larger_patch_roi():
    with pytest.raises(ValueError, match="fixed10m"):
        SurfaceForwardReadoutV1(.8, .8, .8, .8, .5, 11., 4096)
