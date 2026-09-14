from types import SimpleNamespace
import numpy as np
from mtare_topo.semantics.observed_axis_structure import structure_from_observed_primitives

POLICY = dict(max_residual_m=.01, min_crossing_sine=.1,
              endpoint_tolerance_m=1e-8, maximum_candidates=32)


def primitive(a, b, middle=None):
    # Interface-only ideal fitter output; not an observation recovery result.
    controls = [a, ((np.asarray(a)+b)/2).tolist() if middle is None else middle, b]
    return SimpleNamespace(fit=SimpleNamespace(axis_controls_m=controls, reason='synthetic_ideal'))


def test_ideal_t_three_outward_directions():
    result = structure_from_observed_primitives([
        primitive([-3, 0, 0], [3, 0, 0]), primitive([0, 0, 0], [0, 3, 0])], **POLICY)
    assert len(result.structures) == 1
    s = result.structures[0]
    np.testing.assert_allclose(s.position_m, [0, 0, 0], atol=1e-12)
    assert set(s.directions) == {(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)}
    assert not s.connectivity_verified


def test_no_extension_or_stacked_join():
    for branch in (primitive([0, 1, 0], [0, 3, 0]), primitive([0, -3, 2], [0, 3, 2])):
        result = structure_from_observed_primitives([primitive([-3, 0, 0], [3, 0, 0]), branch], **POLICY)
        assert not result.structures


def test_curved_fit_is_not_silently_straightened():
    result = structure_from_observed_primitives([primitive([-3, 0, 0], [3, 0, 0], [0, 2, 0])], **POLICY)
    assert result.rejected_axes == ((0, 'curved_fit_not_represented_by_chord'),)
    assert not result.structures
