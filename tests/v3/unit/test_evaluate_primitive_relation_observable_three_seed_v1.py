import copy

from evaluate_primitive_relation_observable_three_seed_v1 import _comparison
from evaluate_primitive_relation_three_seed_v1 import GEOMETRY_KEYS


def _records() -> tuple[dict, dict]:
    baseline_geometry = {name: 10.0 for name in GEOMETRY_KEYS}
    baseline_geometry.update({"surface_chamfer_m_mean": 10.0, "target_coverage": .2})
    method_geometry = {name: 8.0 for name in GEOMETRY_KEYS}
    method_geometry.update({"surface_chamfer_m_mean": 8.0, "target_coverage": .4})
    baseline = {
        "geometry": baseline_geometry,
        "attachment": {"f1": .1},
        "primitive_detection": {"f1": .2},
    }
    method = {
        "geometry": method_geometry,
        "attachment": {"f1": .2},
        "primitive_detection": {"f1": .4},
        "attachment_safe_selection": {
            "available": True, "precision": .99, "true_positive": 2,
        },
    }
    return method, baseline


def test_observable_comparison_requires_nonempty_safe_true_attachment() -> None:
    method, baseline = _records()
    passing = _comparison(method, baseline)
    assert passing["pass"] is True
    empty = copy.deepcopy(method)
    empty["attachment_safe_selection"]["true_positive"] = 0
    failed = _comparison(empty, baseline)
    assert failed["pass"] is False
    assert failed["checks"]["safe_attachment_precision_ge_98_nonzero"] is False


def test_observable_comparison_keeps_all_original_geometry_relation_gates() -> None:
    method, baseline = _records()
    result = _comparison(method, baseline)
    assert result["checks"] == {
        "surface_improvement_ge_10pct": True,
        "geometry_macro_improvement_ge_10pct": True,
        "attachment_f1_gain_ge_5pt": True,
        "primitive_f1_not_below_baseline": True,
        "target_coverage_not_below_baseline": True,
        "safe_attachment_precision_ge_98_nonzero": True,
    }
