"""Compatibility and invariance contracts for V7 spatial pooling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("v7", TOOLS / "train_aee_corrective_spatial_pooling_v7.py")
assert spec and spec.loader
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


def test_v7_has_exact_source_checkpoint_key_and_shape_compatibility() -> None:
    source = v7.StructuralSemanticNet()
    adapted = v7.SpatialPoolingStructuralSemanticNet()
    adapted.load_state_dict(source.state_dict(), strict=True)
    assert list(source.state_dict()) == list(adapted.state_dict())
    assert all(torch.equal(source.state_dict()[name], adapted.state_dict()[name]) for name in source.state_dict())


def test_v7_classification_is_invariant_to_encoder_aligned_circular_yaw() -> None:
    model = v7.SpatialPoolingStructuralSemanticNet().eval()
    student = torch.randn(2, 2, 16, 720)
    with torch.no_grad():
        first = model(student)
        shifted = model(torch.roll(student, shifts=40, dims=-1))
    assert torch.allclose(first["count_logits"], shifted["count_logits"], atol=2e-6, rtol=0.0)
    assert torch.allclose(first["role_logits"], shifted["role_logits"], atol=2e-6, rtol=0.0)
    assert torch.allclose(torch.roll(first["direction_logits"], shifts=40, dims=-1), shifted["direction_logits"], atol=2e-6, rtol=0.0)


def test_v7_preserves_direction_when_only_embedding_changes() -> None:
    model = v7.SpatialPoolingStructuralSemanticNet().eval()
    student = torch.randn(2, 2, 16, 720)
    with torch.no_grad():
        before = model(student)["direction_logits"].clone()
        for parameter in model.embedding_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        after = model(student)["direction_logits"]
    assert torch.equal(before, after)


def test_v7_nonlinear_before_pooling_differs_from_pool_before_nonlinear() -> None:
    model = v7.SpatialPoolingStructuralSemanticNet().eval()
    features = torch.randn(3, 128, 180)
    with torch.no_grad():
        spatial = model.embedding_head(features.transpose(1, 2)).mean(dim=1)
        collapsed = model.embedding_head(features.mean(dim=2))
    assert not torch.allclose(spatial, collapsed)
