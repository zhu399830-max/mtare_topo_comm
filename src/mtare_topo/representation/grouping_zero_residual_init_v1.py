"""Single registered initialization correction, architecture/loss unchanged.

Only centre residual rows start at zero, so initial positions equal observed
queries instead of being projected onto the10m boundary. Confidence/branch
weights and all remaining parameters retain the same seeded initialization.
"""
import torch
from .observed_anchor_training_v1 import build_model as original_model


def build_model(**kwargs):
    model=original_model(**kwargs)
    with torch.no_grad():
        model['head'].head.anchor.weight[:3].zero_()
        model['head'].head.anchor.bias[:3].zero_()
    return model
