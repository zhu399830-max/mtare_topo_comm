"""Inference-only reuse of the paired relation model; no teacher or file reader.

The executor authenticates manifests and supplies checkpoint bytes. This module
does not select checkpoints, instantiate an optimizer, or change predictions.
"""
from dataclasses import dataclass
import hashlib
import io
import torch
from .block_relation_training_v1 import build_relation_model, forward_observation


@dataclass(frozen=True)
class StudentObservation:
    student_representations: dict


def load_final_model(payload, *, expected_sha256, initial_sha256,
                     relation_attributes, device='cpu'):
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('checkpoint hash mismatch')
    checkpoint = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (set(checkpoint) != {'model', 'optimizer', 'updates', 'initial_sha256'}
            or type(checkpoint['updates']) is not int or checkpoint['updates'] != 2000
            or checkpoint['initial_sha256'] != initial_sha256):
        raise ValueError('only the bound final 2000-update checkpoint is allowed')
    model = build_relation_model(relation_attributes=relation_attributes, seed=0, device=device)
    reference = model.state_dict()
    state = checkpoint['model']
    if set(state) != set(reference):
        raise ValueError('checkpoint model keys differ')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != reference[key].shape
                or value.dtype != reference[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('checkpoint tensor differs or is nonfinite: ' + key)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def predict_r2(model, student):
    """Accept only the existing authenticated R2 representation, never labels."""
    if set(student) != {'blocks', 'context', 'binding'}:
        raise ValueError('only observation blocks, context and provenance allowed')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('model must be frozen and in evaluation mode')
    with torch.inference_mode():
        result = forward_observation(model, StudentObservation({'r2': student}), 'r2')
    arrays = {key: getattr(result, key).detach().cpu().numpy().copy()
              for key in ('position_m', 'presence_logits', 'directions', 'branch_logits')}
    return arrays
