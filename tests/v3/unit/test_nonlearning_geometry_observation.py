from __future__ import annotations

import numpy as np

from mtare_topo.semantics.nonlearning_geometry_observation import nonlearning_geometry_observation
from tests.v3.unit.test_range_geometry_baseline import _rectangular_tunnel_scan


def test_nonlearning_geometry_observation_recognizes_straight_two_exit_corridor() -> None:
    ranges, valid = _rectangular_tunnel_scan()
    observation = nonlearning_geometry_observation(
        np.repeat(ranges[None, ...], 5, axis=0),
        np.repeat(valid[None, ...], 5, axis=0),
    )
    assert observation.event.value == "corridor"
    assert len(observation.exit_tokens) == 2
    assert observation.width_m > 4.0
    assert observation.height_m > 3.0
    assert len(observation.place_descriptor) == 24
