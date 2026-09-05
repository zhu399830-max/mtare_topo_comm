from types import SimpleNamespace

import torch

from mtare_topo.representation.gse_composition import (
    composition_input_from_prediction, visible_geometry_input_from_prediction,
)


def prediction():
    axis = torch.zeros(1, 32, 3, 3)
    axis[:, :, :, 0] = torch.tensor([1., 2., 3.])
    return SimpleNamespace(axis_control_current_sensor_m=axis, existence_logits=torch.full((1, 32), 10.),
                           endpoint_evidence_logits=torch.full((1, 32, 2), -100.),
                           endpoint_half_axes_m=torch.ones(1, 32, 2, 2),
                           endpoint_shape_exponent=torch.ones(1, 32, 2), geometry_uncertainty=torch.zeros(1, 32))


def test_visible_geometry_survives_unknown_physical_endpoints():
    source = prediction()
    old = composition_input_from_prediction(source, existence_threshold=.5, endpoint_threshold=.5)
    new = visible_geometry_input_from_prediction(source, existence_threshold=.5)
    assert old.valid.sum() == 0
    assert new.valid.sum() == 64
    torch.testing.assert_close(new.positions_m, old.positions_m)
    assert new.confidence.min() > .99


def test_node_bridge_does_not_consume_physical_endpoint_classifier():
    source = prediction()
    first = visible_geometry_input_from_prediction(source, existence_threshold=.5)
    del source.endpoint_evidence_logits
    second = visible_geometry_input_from_prediction(source, existence_threshold=.5)
    for name in first.__dict__:
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_degenerate_primitive_is_unknown_not_invented_direction():
    source = prediction()
    source.axis_control_current_sensor_m.zero_()
    output = visible_geometry_input_from_prediction(source, existence_threshold=.5)
    assert not output.valid.any()
