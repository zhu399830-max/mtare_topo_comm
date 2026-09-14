"""Authenticate supplied checkpoint bytes before constructing the frozen encoder.

No paths, dataset reads, checkpoint selection, or inference occur here. The
caller must obtain the expected digest/epoch from the frozen source contract,
not compute them from an arbitrary checkpoint and call that authentication.
This binds weights only; it does not qualify labels or authenticate cache rows.
"""
from dataclasses import dataclass
import hashlib
import io
import re

import torch

from .primitive_relation_observable_model import ObservableSparsePortRelationNet
from .gse_surface_training_v1 import module_state_sha256


@dataclass(frozen=True)
class BoundSurfaceEncoder:
    backbone: ObservableSparsePortRelationNet
    checkpoint_sha256: str
    state_sha256: str
    seed: int
    epoch: int


def load_surface_encoder(payload: bytes, *, expected_sha256: str,
                         expected_epoch: int) -> BoundSurfaceEncoder:
    """Load exactly the preregistered seed-0 observable backbone on CPU.

    Hash the same immutable bytes subsequently deserialized. Never fall back to
    unrestricted pickle, partial state loading, or dtype conversion on error.
    Construction preserves the caller's CPU RNG state (paired head initialization).
    """
    if (type(payload) is not bytes or not payload or len(payload) > 512 * 1024**2):
        raise ValueError("checkpoint must be immutable bytes within 512 MiB")
    if (not isinstance(expected_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None or
            type(expected_epoch) is not int or expected_epoch < 0):
        raise ValueError("frozen checkpoint digest and epoch required")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 drift before deserialization")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or
            checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1" or
            type(checkpoint.get("seed")) is not int or checkpoint["seed"] != 0 or
            type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != expected_epoch or
            not isinstance(checkpoint.get("training_contract"), dict) or
            checkpoint["training_contract"].get("teacher_attachment_validity") != "dual_endpoint_observed" or
            checkpoint["training_contract"].get("frozen_geometry_temporal") is not True):
        raise ValueError("checkpoint schema, seed, epoch or training contract drift")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("model state dictionary required")
    with torch.random.fork_rng(devices=[]):
        model = ObservableSparsePortRelationNet()
    reference = model.state_dict()
    if set(state) != set(reference):
        raise ValueError("checkpoint model keys drift")
    for key, expected in reference.items():
        actual = state[key]
        if (not torch.is_tensor(actual) or actual.layout != torch.strided or
                actual.shape != expected.shape or actual.dtype != expected.dtype or
                not bool(torch.isfinite(actual).all())):
            raise ValueError(f"checkpoint tensor contract drift: {key}")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return BoundSurfaceEncoder(model, digest, module_state_sha256(model), 0, expected_epoch)
