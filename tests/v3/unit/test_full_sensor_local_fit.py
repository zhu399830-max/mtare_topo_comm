import numpy as np
import pytest
from mtare_topo.semantics.full_sensor_local_fit import full_sensor_local_fit
from mtare_topo.semantics.primitive_relation_nonlearning import PrimitiveRelationBaselineConfig
from mtare_topo.semantics.range_exit_baseline import RangeExitBaselineConfig


def run(values):
    return full_sensor_local_fit(values, np.zeros((5, 3)), np.zeros(5),
        fit_config=PrimitiveRelationBaselineConfig(), exit_config=RangeExitBaselineConfig(smoothing_columns=1))


def test_far_returns_propose_but_never_become_local_surface():
    values = np.zeros((5, 2, 16, 720), np.float32)
    values[4, 0, :, :20] = .4; values[4, 1, :, :20] = 1
    result = run(values)
    assert len(result) == 1
    assert result[0].fit.reason == 'insufficient_directional_returns'
    assert result[0].fit.selected_ray_indices == ()


def test_no_return_does_not_invent_axis():
    assert run(np.zeros((5, 2, 16, 720), np.float32)) == ()


def test_invalid_normalization_or_mask_fails():
    for channel, value in ((0, 1.1), (1, .2), (0, np.nan)):
        v = np.zeros((5, 2, 16, 720), np.float32); v[0, channel, 0, 0] = value
        with pytest.raises(ValueError): run(v)
