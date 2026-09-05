from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import evaluate_primitive_relation_three_seed_v1 as base
from evaluate_primitive_relation_sparse_port_three_seed_v1 import (
    _comparison,
    _risk_adjusted_attachment,
    main,
)
from mtare_topo.evaluation.primitive_relation_metrics import THRESHOLD_GRID
from mtare_topo.representation.primitive_relation_model import MAXIMUM_SLOTS


def _record(*, safe_true_positive: int) -> tuple[dict, dict]:
    baseline_geometry = {
        name: 10.0 for name in base.GEOMETRY_KEYS
    }
    baseline_geometry.update({
        "surface_chamfer_m_mean": 10.0,
        "target_coverage": 0.50,
    })
    method_geometry = {
        name: 8.0 for name in base.GEOMETRY_KEYS
    }
    method_geometry.update({
        "surface_chamfer_m_mean": 8.0,
        "target_coverage": 0.60,
    })
    baseline = {
        "geometry": baseline_geometry,
        "attachment": {"f1": 0.20},
        "primitive_detection": {"f1": 0.50},
    }
    method = {
        "geometry": method_geometry,
        "attachment": {"f1": 0.30},
        "primitive_detection": {"f1": 0.60},
        "attachment_safe_selection": {
            "available": True,
            "precision": 1.0,
            "true_positive": safe_true_positive,
        },
    }
    return method, baseline


def test_safe_gate_rejects_vacuous_perfect_precision() -> None:
    method, baseline = _record(safe_true_positive=0)
    result = _comparison(method, baseline)
    assert result["checks"]["safe_attachment_precision_ge_98_nonzero"] is False
    assert result["pass"] is False


def test_safe_gate_accepts_nonzero_true_relation_at_required_precision() -> None:
    method, baseline = _record(safe_true_positive=1)
    result = _comparison(method, baseline)
    assert result["checks"]["safe_attachment_precision_ge_98_nonzero"] is True
    assert result["pass"] is True


def test_relation_uncertainty_reduces_safe_attachment_score() -> None:
    existence = torch.full((1, MAXIMUM_SLOTS), -10.0)
    existence[:, :2] = 10.0
    logits = torch.full((1, MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2), -10.0)
    logits[:, 0, 0, 1, 0] = 4.0
    logits[:, 1, 0, 0, 0] = 4.0
    target = torch.zeros_like(logits, dtype=torch.bool)
    target[:, 0, 0, 1, 0] = True
    target[:, 1, 0, 0, 0] = True
    low = SimpleNamespace(
        existence_logits=existence,
        endpoint_attachment_logits=logits,
        endpoint_attachment_uncertainty=torch.zeros_like(logits),
    )
    high = SimpleNamespace(
        existence_logits=existence,
        endpoint_attachment_logits=logits,
        endpoint_attachment_uncertainty=torch.full_like(logits, 0.9),
    )
    aligned = {"attachment": target}
    threshold_index = int(torch.argmin(torch.abs(
        torch.from_numpy(THRESHOLD_GRID) - 0.5
    )))
    low_counts = _risk_adjusted_attachment(
        low, aligned, existence_threshold=0.5,
    )[threshold_index]
    high_counts = _risk_adjusted_attachment(
        high, aligned, existence_threshold=0.5,
    )[threshold_index]
    assert low_counts.true_positive == 1
    assert high_counts.true_positive == 0


def test_c08_loader_is_created_only_after_complete_c07_stop_branch() -> None:
    source = inspect.getsource(main)
    stop = source.index("if not c07_scientific_pass")
    transfer_loader = source.index(
        "c08_loader = PrimitiveRelationBatchLoader", stop,
    )
    assert transfer_loader > stop

