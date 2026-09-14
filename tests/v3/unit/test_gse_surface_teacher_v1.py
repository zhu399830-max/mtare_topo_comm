"""Invented in-memory evidence only; no robot capability values inferred here."""
from dataclasses import replace
import pytest

from mtare_topo.teacher.gse_surface_teacher_v1 import (
    ContractScope, FiveFrameSupport, GroundPath, GroundRobotContract, Label,
    Observation, Obstruction, PhysicalReference, Truth, fuse_ground_path,
)


def fixtures():
    contract = GroundRobotContract(ContractScope.SYNTHETIC_ONLY, 1., 1., 1., 20., .2,
                                   ("synthetic/body", "synthetic/support", "synthetic/slope", "synthetic/step", "synthetic/pose"))
    path = GroundPath(("a", "b", "c"), ((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)),
                      ("ground0",) * 3, (False, False))
    reference = PhysicalReference(path, Truth.PASS, contract, "synthetic/reference")
    observation = FiveFrameSupport(path, (0, 2, 4, 6, 8), 8, Observation.SUPPORTED_PATH,
                                   (True,) * 3, (True,) * 2, (0, 8), ("synthetic/ray0", "synthetic/ray8"), None)
    return reference, observation


def fused(reference, observed):
    return fuse_ground_path(reference, observed, allow_synthetic=True)


@pytest.mark.parametrize("truth,observed,expected", [
    (Truth.PASS, Observation.SUPPORTED_PATH, Label.POSITIVE),
    (Truth.BLOCKED, Observation.SUPPORTED_PATH, Label.UNKNOWN),
    (Truth.UNKNOWN, Observation.SUPPORTED_PATH, Label.UNKNOWN),
    (Truth.PASS, Observation.OBSERVED_OBSTRUCTION, Label.UNKNOWN),
    (Truth.BLOCKED, Observation.OBSERVED_OBSTRUCTION, Label.NEGATIVE),
    (Truth.UNKNOWN, Observation.OBSERVED_OBSTRUCTION, Label.UNKNOWN),
    (Truth.PASS, Observation.UNKNOWN, Label.UNKNOWN),
    (Truth.BLOCKED, Observation.UNKNOWN, Label.UNKNOWN),
    (Truth.UNKNOWN, Observation.UNKNOWN, Label.UNKNOWN),
])
def test_full_fusion_table(truth, observed, expected):
    ref, obs = fixtures()
    obs = replace(obs, status=observed,
                  obstruction=Obstruction.COLLISION if observed is Observation.OBSERVED_OBSTRUCTION else None)
    target = fused(replace(ref, status=truth), obs)
    assert target.label is expected
    assert target.synthetic_only
    assert target.label_scope == "THIS_DISCRETE_GROUND_PATH_ONLY"


def test_unbound_and_synthetic_contract_fail_closed():
    ref, obs = fixtures()
    assert fuse_ground_path(ref, obs).label is Label.UNKNOWN
    assert fused(replace(ref, contract=None), obs).reason == "ROBOT_PHYSICAL_CONTRACT_UNBOUND"


@pytest.mark.parametrize("change", [
    {"envelope_height_m": None}, {"envelope_width_m": True}, {"max_slope_deg": 90.},
    {"max_step_m": -1.}, {"envelope_length_m": float("nan")}, {"envelope_height_m": 0.},
    {"source_bindings": ("partial",)}, {"source_bindings": ("",) * 5}, {"scope": "DEPLOYMENT_BOUND"},
])
def test_invalid_contract_rejected(change):
    ref, _ = fixtures()
    with pytest.raises(ValueError):
        replace(ref.contract, **change)


def test_observed_evidence_unchanged_cannot_flip_positive_to_negative_with_hidden_map():
    ref, obs = fixtures()
    assert fused(ref, obs).label is Label.POSITIVE
    assert fused(replace(ref, status=Truth.BLOCKED), obs).label is Label.UNKNOWN
    occluded = replace(obs, status=Observation.UNKNOWN, witness_ids=(), witness_frame_indices=())
    assert {fused(replace(ref, status=t), occluded).label for t in Truth} == {Label.UNKNOWN}


@pytest.mark.parametrize("reason", list(Obstruction))
def test_negative_needs_visible_obstruction_not_search_failure(reason):
    ref, obs = fixtures()
    blocked = replace(obs, status=Observation.OBSERVED_OBSTRUCTION, obstruction=reason)
    assert fused(replace(ref, status=Truth.BLOCKED), blocked).label is Label.NEGATIVE
    with pytest.raises(ValueError):
        replace(blocked, obstruction="SEARCH_FAILED")
    with pytest.raises(ValueError):
        replace(blocked, witness_ids=(), witness_frame_indices=())


def test_stacked_xy_does_not_create_a_ground_transition():
    ref, obs = fixtures()
    path = replace(ref.path, ground_xyz_m=((0., 0., 0.), (0., 0., 3.), (1., 0., 3.)),
                   layer_ids=("lower", "upper", "upper"))
    assert fused(replace(ref, path=path), replace(obs, path=path)).reason == "UNSUPPORTED_CROSS_LAYER_TRANSITION"
    # This is an upstream supported-ramp attestation, not inference from XY.
    ramp = replace(path, layer_transition_supported=(True, False))
    assert fused(replace(ref, path=ramp), replace(obs, path=ramp)).label is Label.POSITIVE


def test_air_and_unobserved_ground_do_not_become_supported():
    ref, obs = fixtures()
    for unsupported in [replace(obs, ground_supported=(False, True, True)),
                        replace(obs, transition_supported=(True, False))]:
        assert fused(ref, unsupported).label is Label.UNKNOWN


def test_reference_must_confirm_same_path_not_an_alternative_gt_route():
    ref, obs = fixtures()
    altered = replace(obs.path, ground_xyz_m=((0., 0., 0.), (1., 1., 0.), (2., 0., 0.)))
    with pytest.raises(ValueError, match="exact same ground path"):
        fused(ref, replace(obs, path=altered))


def test_blocked_candidate_does_not_preclude_a_different_open_branch():
    ref, obs = fixtures()
    blocked = replace(obs, status=Observation.OBSERVED_OBSTRUCTION, obstruction=Obstruction.COLLISION)
    blocked_target = fused(replace(ref, status=Truth.BLOCKED), blocked)
    alternate_path = replace(obs.path, state_ids=("a", "d", "e"),
                             ground_xyz_m=((0., 0., 0.), (0., 1., 0.), (0., 2., 0.)))
    open_target = fused(replace(ref, path=alternate_path), replace(obs, path=alternate_path))
    assert blocked_target.label is Label.NEGATIVE and open_target.label is Label.POSITIVE
    assert blocked_target.label_scope == "THIS_DISCRETE_GROUND_PATH_ONLY"


@pytest.mark.parametrize("change", [
    {"frame_indices": (0, 1, 2, 3)}, {"frame_indices": (0, 1, 1, 3, 8)},
    {"frame_indices": (0, 2, 4, 6, 9)}, {"current_frame_index": True},
    {"witness_frame_indices": (0, 9)}, {"witness_ids": ("only-one",)},
    {"ground_supported": (1, True, True)}, {"transition_supported": (True,)},
    {"status": "SUPPORTED_PATH"}, {"obstruction": Obstruction.COLLISION},
])
def test_bad_observation_contract_rejected(change):
    _, obs = fixtures()
    with pytest.raises(ValueError):
        replace(obs, **change)


@pytest.mark.parametrize("change", [
    {"state_ids": ("a", "a", "c")}, {"ground_xyz_m": ((0., 0., float("inf")),) * 3},
    {"layer_ids": ("floor",)}, {"layer_transition_supported": (1, 0)},
])
def test_bad_ground_state_contract_rejected(change):
    ref, _ = fixtures()
    with pytest.raises(ValueError):
        replace(ref.path, **change)
