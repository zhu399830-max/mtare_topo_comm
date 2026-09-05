import numpy as np

from execute_gse_coust_complex_cardinality_failure_attribution_v1 import (
    circular_pair_gaps,
    cyclic_assignment_diagnostic,
)


def test_circular_pair_gaps_include_wrap():
    assert np.allclose(circular_pair_gaps(np.asarray([350.0, 10.0, 100.0])), [90.0, 250.0, 20.0])


def test_cyclic_assignment_accepts_shift_and_penalizes_crossing():
    target = np.asarray([10.0, 100.0, 230.0])
    shifted = np.asarray([100.0, 230.0, 10.0])
    correct = cyclic_assignment_diagnostic(shifted, target)
    assert correct["unrestricted_is_cyclic"]
    assert correct["cyclic_mean_error_deg"] == 0.0
    crossing = np.asarray([10.0, 230.0, 100.0])
    wrong = cyclic_assignment_diagnostic(crossing, target)
    assert not wrong["unrestricted_is_cyclic"]
    assert wrong["cyclic_mean_error_deg"] > wrong["unrestricted_mean_error_deg"]
