import numpy as np
import pytest

from mtare_topo.representation.gse_structural_node_evidence import (
    NODE_EVIDENCE_DIM,
    PrimitivePortEvidence,
    pair_distance,
    safest_nonempty_threshold,
    structural_node_evidence_descriptor,
)


def _port(direction, axes=(4.0, 3.0), exponent=2.0, curvature=.04, support=500):
    return PrimitivePortEvidence(direction, axes, exponent, curvature, support)


def test_descriptor_is_port_permutation_and_rigid_rotation_invariant():
    ports = (
        _port((1.0, 0.0, 0.0), axes=(3.0, 4.0)),
        _port((0.0, 1.0, 0.0), axes=(5.0, 3.0), exponent=6.0),
        _port((-1.0, 0.0, .2), axes=(4.0, 4.0), curvature=.1),
    )
    expected = structural_node_evidence_descriptor(ports)
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rotated = tuple(
        PrimitivePortEvidence(tuple(rotation @ np.asarray(port.away_direction_xyz)), port.half_axes_m,
                              port.shape_exponent, port.curvature_per_m, port.support_rays)
        for port in reversed(ports)
    )
    observed = structural_node_evidence_descriptor(rotated)
    np.testing.assert_allclose(expected, observed, atol=1e-7)
    assert expected.shape == (NODE_EVIDENCE_DIM,)


def test_descriptor_changes_when_composition_changes():
    straight = structural_node_evidence_descriptor(
        (_port((1, 0, 0)), _port((-1, 0, 0)))
    )
    branch = structural_node_evidence_descriptor(
        (_port((1, 0, 0)), _port((-1, 0, 0)), _port((0, 1, 0)))
    )
    distance = pair_distance(straight[None], branch[None])
    assert distance[0] > 0.05


def test_lower_support_increases_explicit_uncertainty():
    strong = structural_node_evidence_descriptor((_port((1, 0, 0), support=10_000),))
    weak = structural_node_evidence_descriptor((_port((1, 0, 0), support=2),))
    assert weak[-1] > strong[-1]
    assert pair_distance(strong[None], weak[None])[0] == pytest.approx(0.0)
    assert pair_distance(strong[None], weak[None], include_uncertainty=True)[0] > 0.0


def test_threshold_is_tie_safe_and_never_splits_equal_distance():
    distance = np.asarray((.1, .1, .2, .3, .4))
    target = np.asarray((True, False, True, False, False))
    result = safest_nonempty_threshold(distance, target, minimum_precision=.6)
    assert result["found"]
    assert result["threshold"] == pytest.approx(.2)
    assert result["predicted_positive"] == 3
    assert result["precision"] == pytest.approx(2 / 3)


def test_threshold_refuses_when_no_high_precision_nonempty_region_exists():
    result = safest_nonempty_threshold(
        np.asarray((.1, .2, .3, .4)),
        np.asarray((False, False, True, True)),
        minimum_precision=.98,
    )
    assert not result["found"]
    assert result["threshold"] is None


def test_invalid_port_contract_is_rejected():
    with pytest.raises(ValueError, match="contract"):
        structural_node_evidence_descriptor((_port((0, 0, 0)),))
