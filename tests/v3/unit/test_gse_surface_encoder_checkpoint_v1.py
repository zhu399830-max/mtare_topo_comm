"""Synthetic checkpoint serialization only; no historical weights or map payload."""
import hashlib
import io

import pytest
import torch

from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256


@pytest.fixture
def checkpoint():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = ObservableSparsePortRelationNet()
    return {"schema_version": "primitive_relation_observable_checkpoint_v1",
            "seed": 0, "epoch": 2, "model_state_dict": model.state_dict(),
            "training_contract": {"teacher_attachment_validity": "dual_endpoint_observed",
                                  "frozen_geometry_temporal": True}}


def packed(checkpoint):
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


def load(checkpoint):
    payload = packed(checkpoint)
    return load_surface_encoder(payload, expected_sha256=hashlib.sha256(payload).hexdigest(), expected_epoch=2)


def test_exact_frozen_load_preserves_rng(checkpoint):
    rng = torch.get_rng_state().clone()
    bound = load(checkpoint)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not bound.backbone.training
    assert all(not p.requires_grad for p in bound.backbone.parameters())
    assert bound.state_sha256 == module_state_sha256(bound.backbone)
    assert bound.seed == 0 and bound.epoch == 2
    for key, value in bound.backbone.state_dict().items():
        assert torch.equal(value, checkpoint["model_state_dict"][key])


def test_loaded_encoder_extracts_shared_compact_features(checkpoint):
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    bound = load(checkpoint)
    adapter = FrozenDualPathEncoderAdapterV1(bound.backbone, torch.nn.Linear(1, 1))
    raw = torch.ones(1, 5, 2, 16, 720)
    raw[:, :, 0] = .1
    translation, yaw = torch.zeros(1, 5, 3), torch.zeros(1, 5)
    first = adapter.extract_compact_features(raw, translation, yaw)
    adapter.train()  # Training the new head must not put the encoder in training mode.
    second = adapter.extract_compact_features(raw, translation, yaw)
    assert first.frozen_sensor_context.shape == (1, 900, 128)
    assert torch.isfinite(first.frozen_sensor_context).all()
    assert torch.equal(first.frozen_sensor_context, second.frozen_sensor_context)
    assert not first.frozen_sensor_context.requires_grad
    assert not bound.backbone.training
    assert bound.state_sha256 == module_state_sha256(bound.backbone)


def test_hash_checked_before_deserialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("deserializer called before digest verification")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        load_surface_encoder(b"not a checkpoint", expected_sha256="0" * 64, expected_epoch=2)


@pytest.mark.parametrize("field,value", [("seed", 1), ("seed", False), ("epoch", 3),
                                         ("schema_version", "other")])
def test_metadata_drift(checkpoint, field, value):
    checkpoint[field] = value
    with pytest.raises(ValueError, match="contract drift"):
        load(checkpoint)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "nan", "shape", "contract"])
def test_state_drift_is_not_silently_repaired(checkpoint, mutation):
    state = checkpoint["model_state_dict"]
    key = next(iter(state))
    if mutation == "missing":
        del state[key]
    elif mutation == "extra":
        state["invented"] = torch.ones(1)
    elif mutation == "dtype":
        state[key] = state[key].double()
    elif mutation == "nan":
        state[key] = torch.full_like(state[key], float("nan"))
    elif mutation == "shape":
        state[key] = state[key].reshape(-1)[:1]
    else:
        checkpoint["training_contract"]["frozen_geometry_temporal"] = False
    with pytest.raises(ValueError, match="drift"):
        load(checkpoint)
