from __future__ import annotations

import pytest

from mtare_topo.representation.gse_corrected_causal_event import (
    REQUIRED_EVENT_MACRO_F1,
    corrected_causal_event_gate,
    wilson_interval,
)


def _metrics(*, macro: float = 0.80, transitions: int = 7) -> dict:
    return {
        "event": {"macro_f1": macro},
        "structural_selection": {
            "precision": 0.99,
            "false_accept_rate": 0.01,
            "recall": 0.50,
        },
        "identity_coverage": {
            "junction": {"correct_class_identity_coverage": 0.95},
            "terminal": {"correct_class_identity_coverage": 0.95},
            "turn": {"correct_class_identity_coverage": 0.45},
            "geometry_transition": {
                "teacher_identities": 17,
                "covered_identities": transitions,
                "correct_class_identity_coverage": transitions / 17,
            },
        },
    }


def test_corrected_causal_gate_requires_seven_of_seventeen_identities() -> None:
    passed = corrected_causal_event_gate(_metrics(transitions=7))
    assert passed["passed"]
    assert passed["change_point_identity_coverage_wilson_95"]["trials"] == 17
    failed = corrected_causal_event_gate(_metrics(transitions=6))
    assert not failed["passed"]
    assert not failed["requirements"]["change_point_identity_coverage_at_least_7_of_17"]


def test_corrected_causal_gate_uses_strongest_corrected_label_baseline() -> None:
    result = corrected_causal_event_gate(_metrics(macro=REQUIRED_EVENT_MACRO_F1 - 1e-9))
    assert not result["passed"]
    assert not result["requirements"]["event_macro_f1_vs_old_directional_plus_0p05"]


def test_wilson_interval_is_finite_and_rejects_invalid_counts() -> None:
    interval = wilson_interval(7, 17)
    assert 0.0 < interval.lower_95 < interval.estimate < interval.upper_95 < 1.0
    with pytest.raises(ValueError):
        wilson_interval(18, 17)
