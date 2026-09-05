import numpy as np

from evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1 import (
    _align_cyclic_order,
    _baseline_improvement,
)


def test_alignment_allows_cyclic_but_not_reflection():
    reference = np.asarray([10.0, 80.0, 210.0])
    cyclic = np.asarray([210.0, 10.0, 80.0])
    order = _align_cyclic_order(reference, cyclic)
    assert np.allclose(cyclic[order], reference)
    reflected = np.asarray([10.0, 210.0, 80.0])
    reflected_order = _align_cyclic_order(reference, reflected)
    assert not np.allclose(reflected[reflected_order], reference)


def test_registered_gain_requires_every_predeclared_component():
    def metrics(overall, three, four, recall):
        return {
            "refusal": {"raw_exact_set_fraction_2deg": overall},
            "detection": {"recall": recall},
            "cardinality": {"strata": [
                {"target_count": 1, "exact_set_fraction_2deg": 0.0},
                {"target_count": 2, "exact_set_fraction_2deg": 0.0},
                {"target_count": 3, "exact_set_fraction_2deg": three},
                {"target_count": 4, "exact_set_fraction_2deg": four},
            ]},
        }
    passed = _baseline_improvement("c07", metrics(0.40, 0.08, 0.06, 0.06))
    assert passed["passes_registered_gain"]
    failed = _baseline_improvement("c07", metrics(0.40, 0.08, 0.049, 0.06))
    assert not failed["passes_registered_gain"]
