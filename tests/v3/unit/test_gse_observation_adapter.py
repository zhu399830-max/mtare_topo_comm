from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.semantics.gse_observation_adapter import geometric_semantic_observation_from_arrays


def test_observation_adapter_applies_calibration_and_exit_rejection() -> None:
    outputs = {
        "event_logits": np.asarray(((0.0, 3.0, 0.0, 0.0, 0.0),)),
        "local_axis": np.asarray(((1.0, 0.0, 0.0),)),
        "width_m": np.asarray((5.0,)),
        "height_m": np.asarray((4.0,)),
        "slope_deg": np.asarray((2.0,)),
        "curvature_per_m": np.asarray((0.02,)),
        "place_descriptor": np.asarray(((1.0, 0.0),)),
        "uncertainty": np.asarray((0.1,)),
        "exit_confidence": np.asarray(((0.9, 0.2),)),
        "exit_heading_unit": np.asarray((((0.0, 1.0), (1.0, 0.0)),)),
        "exit_opening_width_m": np.asarray(((3.0, 2.0),)),
        "exit_vertical_profile": np.zeros((1, 2, 4)),
        "exit_descriptor": np.asarray((((1.0, 0.0), (0.0, 1.0)),)),
    }
    observation = geometric_semantic_observation_from_arrays(
        outputs,
        0,
        event_temperature=2.0,
        exit_presence_threshold=0.5,
    )
    assert observation.event.value == "junction"
    assert len(observation.exit_tokens) == 1
    assert observation.exit_tokens[0].heading_robot_deg == pytest.approx(0.0)
    assert sum(observation.event_probabilities.values()) == pytest.approx(1.0)
