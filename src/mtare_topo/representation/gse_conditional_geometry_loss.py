"""Separate conditional-reference objective; no observable-label substitution.

Per observation: average each unordered construction-component pair, then each
available same/cross category equally. Parents are averaged by the fixed sample
schedule. Repeated patch pairs are not independent examples. Height is scaled
by the fixed 10m input radius, not a fitted normalization statistic.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ConditionalGeometryLossTargets:
    axis: torch.Tensor
    height: torch.Tensor
    known: torch.Tensor
    same: torch.Tensor
    weights: torch.Tensor
    schema: str = 'construction_conditioned_geometry_targets_v1'


def compile_conditional_loss_targets(arrays, device='cpu'):
    if str(arrays['schema_version']) != 'construction_conditioned_geometry_targets_v1' or bool(arrays['observability_certified']):
        raise ValueError('explicit conditional reference schema required')
    known = np.asarray(arrays['axis_known'], dtype=bool)
    n = np.asarray(arrays['neighbor_index']); c = np.asarray(arrays['patch_reference_component'])
    a = np.asarray(arrays['axis_abs_dot']); h = np.asarray(arrays['height_difference_m'])
    same = np.asarray(arrays['same_reference_component'], dtype=bool)
    if (known.shape != n.shape or known.shape != a.shape or known.shape != h.shape or same.shape != known.shape
            or not np.array_equal(known, arrays['height_known']) or c.shape != (len(n),)):
        raise ValueError('target indexing mismatch')
    if not np.isfinite(a[known]).all() or not np.isfinite(h[known]).all() or np.any((a[known]<0)|(a[known]>1)):
        raise ValueError('invalid known values')
    groups = {False: {}, True: {}}
    for p, k in zip(*np.nonzero(known)):
        q = int(n[p,k])
        if q < 0 or q >= len(c) or c[p] < 0 or c[q] < 0: raise ValueError('known target missing component')
        category = bool(c[p] == c[q])
        if category != same[p,k]: raise ValueError('category mismatch')
        key = tuple(sorted((int(c[p]),int(c[q]))))
        groups[category].setdefault(key, []).append((p,k))
    weights = np.zeros_like(a, dtype=np.float32)
    active = sum(bool(g) for g in groups.values())
    for group in groups.values():
        for indices in group.values():
            for p,k in indices: weights[p,k] = 1/(active*len(group)*len(indices))
    tensor = lambda x: torch.as_tensor(x, device=device)[None]
    return ConditionalGeometryLossTargets(tensor(np.nan_to_num(a).astype(np.float32)), tensor(np.nan_to_num(h).astype(np.float32)),
        tensor(known), tensor(same), tensor(weights))


def conditional_geometry_loss(prediction, targets, *, axis_objective='conditional_axis_l1_v1'):
    if type(targets) is not ConditionalGeometryLossTargets or targets.schema != 'construction_conditioned_geometry_targets_v1':
        raise ValueError('conditional loss requires dedicated target type')
    known = targets.known
    if known.shape != prediction.computation_valid.shape or bool((known & ~prediction.computation_valid).any()):
        raise ValueError('known targets outside model pairs')
    axis = prediction.axis_abs_dot[known]; height = prediction.height_difference_m[known]
    if not bool(torch.isfinite(axis).all() & torch.isfinite(height).all()): raise ValueError('nonfinite prediction')
    w = targets.weights[known]
    if axis_objective == 'conditional_axis_l1_v1':
        la = (w*(axis-targets.axis[known]).abs()).sum()
    elif axis_objective == 'conditional_axis_soft_target_logit_v1':
        from .gse_axis_reference_objective import axis_reference_logit_loss
        if prediction.axis_logits is None:
            raise ValueError('raw axis logits required; never reconstruct from rounded probabilities')
        la = axis_reference_logit_loss(prediction.axis_logits, targets.axis, known, targets.weights)
    else:
        raise ValueError('unregistered axis objective')
    lh = (w*(height-targets.height[known]).abs()/10.).sum()
    return la+lh, {'axis': la, 'height_normalized': lh}
