from __future__ import annotations

import inspect

from evaluate_primitive_relation_three_seed_v1 import GEOMETRY_KEYS, _comparison, main


def _record(error: float, *, attachment_f1: float, primitive_f1: float, coverage: float) -> dict:
    geometry = {name:error for name in GEOMETRY_KEYS}
    geometry.update({"surface_chamfer_m_mean":error,"target_coverage":coverage})
    return {
        "geometry":geometry,
        "attachment":{"f1":attachment_f1},
        "primitive_detection":{"f1":primitive_f1},
    }


def test_comparison_requires_independent_geometry_relation_and_coverage_gain() -> None:
    baseline = _record(10.0, attachment_f1=0.20, primitive_f1=0.50, coverage=0.50)
    method = _record(8.0, attachment_f1=0.30, primitive_f1=0.60, coverage=0.60)
    result = _comparison(method, baseline)
    assert result["pass"] is True
    assert result["surface_improvement"] >= 0.10
    assert result["attachment_f1_gain"] >= 0.05


def test_comparison_rejects_easy_subset_despite_low_matched_error() -> None:
    baseline = _record(10.0, attachment_f1=0.20, primitive_f1=0.50, coverage=0.50)
    method = _record(5.0, attachment_f1=0.30, primitive_f1=0.60, coverage=0.40)
    result = _comparison(method, baseline)
    assert result["checks"]["target_coverage_not_below_baseline"] is False
    assert result["pass"] is False


def test_c08_loader_is_created_only_after_c07_stop_branch() -> None:
    source = inspect.getsource(main)
    stop = source.index("if not c07_scientific_pass")
    transfer_loader = source.index("c08_loader = PrimitiveRelationBatchLoader", stop)
    assert transfer_loader > stop
