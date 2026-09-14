"""Loss-only consistency on already qualified, observed correspondences.

The caller gathers corresponding local structure tokens into the same row in
each view. This function neither finds matches nor receives teacher identities;
it never treats unmatched places as contrastive negatives. A known mask is an
explicit prerequisite, not something inferred from vector similarity.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MatchedStructureConsistency:
    invariance_mse: torch.Tensor
    standard_deviation_hinge: torch.Tensor
    covariance_off_diagonal: torch.Tensor
    effective_pair_count: int
    regularizer_valid: bool


def matched_structure_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
    known: torch.Tensor,
) -> MatchedStructureConsistency:
    """Return separate terms; training weights are deliberately not selected.

    Inputs are matching ``N,D`` token matrices and an explicit boolean ``N``
    qualification mask. Unknown rows are removed *before* any finite check or
    arithmetic, so unknown NaNs produce no loss or gradient. Neither a zero
    loss on an empty mask nor two rows establishes independent revisit evidence.

    Invariance is mean squared error across known rows and dimensions. The
    standard-deviation term is ``relu(1 - sample_std)`` averaged over dimensions
    and the two views. Sample statistics use correction 1. Covariance is the
    squared off-diagonal sum divided by D, averaged over views. At least two
    qualified pairs are required for either regularizer; otherwise both return
    differentiable zero and ``regularizer_valid`` is false.
    """
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise ValueError("two floating-point tensor views are required")
    if (first.ndim != 2 or second.shape != first.shape or first.shape[1] < 1
            or not first.is_floating_point() or not second.is_floating_point()):
        raise ValueError("views must have the same N,D floating-point shape with D >= 1")
    if first.dtype != second.dtype or first.device != second.device:
        raise ValueError("view dtype and device must match")
    if (not isinstance(known, torch.Tensor) or known.dtype != torch.bool
            or known.shape != (first.shape[0],) or known.device != first.device):
        raise ValueError("known must be a same-device boolean N qualification mask")

    x, y = first[known], second[known]
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("qualified token values must be finite")
    count, dimensions = x.shape
    # Empty slices retain a gradient connection without evaluating even a
    # qualified token's value (sum(large_values) * 0 could produce NaN).
    zero = x[:0].sum() + y[:0].sum()
    invariance = (x - y).square().mean() if count else zero
    if count < 2:
        return MatchedStructureConsistency(invariance, zero, zero, count, False)

    std_penalty = (F.relu(1.0 - x.std(dim=0, correction=1)).mean()
                   + F.relu(1.0 - y.std(dim=0, correction=1)).mean()) / 2.0

    off_diagonal = ~torch.eye(dimensions, dtype=torch.bool, device=first.device)
    covariance_terms = []
    for view in (x, y):
        centered = view - view.mean(dim=0)
        covariance = centered.T @ centered / (count - 1)
        covariance_terms.append(covariance[off_diagonal].square().sum() / dimensions)
    covariance_penalty = (covariance_terms[0] + covariance_terms[1]) / 2.0
    return MatchedStructureConsistency(invariance, std_penalty, covariance_penalty, count, True)
