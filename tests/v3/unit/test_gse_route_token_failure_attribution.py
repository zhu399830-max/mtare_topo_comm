from __future__ import annotations

import numpy as np
import pytest

from tools.v3.execute_gse_route_token_failure_attribution_v1 import classify_failure_mechanism


def test_exclusive_teacher_conflict_when_junction_is_stable_and_visible():
    probability = np.asarray([
        [1.00, 0.96, 0.10],
        [1.00, 0.94, 0.12],
        [1.00, 0.92, 0.08],
    ])
    route_stop = np.asarray([0.2, 0.3, 0.4])
    assert classify_failure_mechanism(probability, route_stop, 12.0) == "exclusive_teacher_conflict"


def test_seed_representation_instability_requires_route_stop_consensus():
    probability = np.asarray([
        [0.10, 0.08, 0.20],
        [0.62, 0.30, 0.10],
        [0.95, 0.40, 0.05],
    ])
    route_stop = np.asarray([0.80, 0.75, 0.90])
    assert classify_failure_mechanism(probability, route_stop, None) == "seed_representation_instability"


def test_unresolved_when_neither_conflict_nor_instability_gate_holds():
    probability = np.asarray([
        [0.50, 0.20, 0.30],
        [0.65, 0.25, 0.20],
        [0.80, 0.30, 0.10],
    ])
    route_stop = np.asarray([0.40, 0.50, 0.60])
    assert classify_failure_mechanism(probability, route_stop, None) == "unresolved"


@pytest.mark.parametrize(
    ("probability", "route_stop"),
    [
        (np.zeros((2, 3)), np.zeros(3)),
        (np.zeros((3, 3)), np.zeros(2)),
        (np.full((3, 3), np.nan), np.zeros(3)),
    ],
)
def test_invalid_shape_or_values_raise_value_error(probability, route_stop):
    with pytest.raises(ValueError, match="failure attribution"):
        classify_failure_mechanism(probability, route_stop, None)
