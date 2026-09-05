from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/v3/execute_gse_axis_anchored_event_relation_teacher_feasibility_v1.py"
SPEC = importlib.util.spec_from_file_location("axis_anchored_teacher_feasibility", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_signed_bearing_uses_sensor_forward_as_zero() -> None:
    heading = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(module.signed_bearing_deg(heading), [0.0, 90.0, -90.0, 180.0])


def test_signed_bearing_rejects_invalid_heading() -> None:
    with pytest.raises(ValueError, match="finite"):
        module.signed_bearing_deg(np.asarray([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match=r"\[\.\.\.,2\]"):
        module.signed_bearing_deg(np.asarray([1.0, 0.0, 0.0]))


def test_minimum_circular_separation_handles_wrap() -> None:
    assert module.minimum_circular_separation_deg(np.asarray([-179.0, 179.0, 0.0])) == 2.0
    assert module.minimum_circular_separation_deg(np.asarray([42.0])) == 360.0


def test_partition_is_world_disjoint_and_rejects_forbidden_worlds() -> None:
    assert module._partition("S01_C01") == "fit"
    assert module._partition("S10_C06") == "fit"
    assert module._partition("S01_C07") == "selection"
    assert module._partition("S10_C08") == "selection"
    with pytest.raises(ValueError, match="outside C01-C08"):
        module._partition("S01_C09")


def test_identity_summary_preserves_event_and_canonical_star_counts() -> None:
    rows = [
        {
            "partition": "selection",
            "event": "junction",
            "distinct_traversals": 4,
            "full_visibility_rows": 3,
            "star_variants": 1,
            "canonical_exit_count": 3,
        },
        {
            "partition": "selection",
            "event": "junction",
            "distinct_traversals": 5,
            "full_visibility_rows": 2,
            "star_variants": 1,
            "canonical_exit_count": 4,
        },
    ]
    summary = module._identity_summary(rows)
    junction = summary["selection"]["junction"]
    assert junction["identities"] == 2
    assert junction["minimum_traversals"] == 4
    assert junction["exact_canonical_star_identities"] == 2
    assert junction["canonical_exit_count"] == {"3": 1, "4": 1}
