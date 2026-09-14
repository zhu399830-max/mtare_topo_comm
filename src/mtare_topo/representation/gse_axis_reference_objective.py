"""Opt-in conditional axis regression using stable soft-target logit loss.

The target is a continuous unsigned axis dot product, NOT a binary event or
connectivity probability. This objective elicits its conditional mean, whereas
L1 elicits a conditional median; it is deliberately not called an equivalent
implementation of the old objective. No target, weight or mask is derived here.
"""
import torch
from torch.nn import functional as F

AXIS_REFERENCE_LOGIT_V1 = 'conditional_axis_soft_target_logit_v1'


def axis_reference_logit_loss(logits, target, known, weights):
    """Mask before computation; preserve the supplied loss weights exactly."""
    if any(not isinstance(x, torch.Tensor) for x in (logits, target, known, weights)):
        raise ValueError('explicit tensors required')
    if (known.dtype != torch.bool or any(x.shape != logits.shape for x in (target,known,weights))
            or any(x.device != logits.device for x in (target,known,weights))):
        raise ValueError('same-shaped device-bound target, mask and weights required')
    if not logits.is_floating_point() or not target.is_floating_point() or not weights.is_floating_point():
        raise ValueError('floating logits, reference values and weights required')
    z=logits[known];y=target[known].to(z.dtype);w=weights[known].to(z.dtype)
    if (any(not bool(torch.isfinite(x).all()) for x in (z,y,w))
            or bool(((y<0)|(y>1)).any()) or bool((w<0).any())):
        raise ValueError('invalid known conditional reference or weight')
    if not z.numel(): return z.sum()
    return (w*F.binary_cross_entropy_with_logits(z,y,reduction='none')).sum()
