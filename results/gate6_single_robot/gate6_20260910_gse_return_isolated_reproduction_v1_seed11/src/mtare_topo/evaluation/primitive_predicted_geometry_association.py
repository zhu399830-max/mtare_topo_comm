"""C07 diagnostics for relation scores derived from predicted primitive endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mtare_topo.evaluation.primitive_relation_metrics import (
    _attachment_observability_validity,
    _attachment_upper_mask,
)
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationPrediction


@dataclass(frozen=True)
class PredictedGeometryAssociationSlice:
    geometry_score: np.ndarray
    learned_pair_score: np.ndarray
    target: np.ndarray
    overlap_hard_negative: np.ndarray
    eligible_pairs: int
    all_observable_target_pairs: int

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.geometry_score, self.learned_pair_score,
            self.target, self.overlap_hard_negative,
        ))
        if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
            raise ValueError("predicted geometry association slices differ")
        if not np.all(np.isfinite(arrays[0])) or not np.all(np.isfinite(arrays[1])):
            raise ValueError("predicted geometry association scores must be finite")
        if self.eligible_pairs != len(arrays[0]) or self.all_observable_target_pairs < int(np.count_nonzero(arrays[2])):
            raise ValueError("predicted geometry association population differs")


def predicted_geometry_association_slice(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    primitive_active: torch.Tensor,
) -> PredictedGeometryAssociationSlice:
    """Compare endpoint-distance and learned-pair scores under one proposal mask.

    The eligibility contract exactly matches the endpoint-observable evaluator:
    hidden matched pairs are unknown, while active unmatched proposals remain
    valid negatives.  Geometry score is negative Euclidean distance, so larger
    values indicate a more likely shared connection anchor.
    """

    active = primitive_active.bool()
    if active.ndim != 2 or active.shape[1:] != (32,):
        raise ValueError("predicted geometry proposal mask must be [batch,32]")
    batch = len(active)
    endpoint = prediction.axis_control_current_sensor_m[:, :, (0, 2), :].reshape(batch, 64, 3)
    if endpoint.shape != (batch, 64, 3) or not torch.isfinite(endpoint).all():
        raise ValueError("predicted primitive endpoints must be finite [batch,64,3]")
    geometry = -torch.cdist(endpoint, endpoint, p=2)
    learned = torch.sigmoid(prediction.endpoint_attachment_logits).reshape(batch, 64, 64)
    validity = _attachment_observability_validity(aligned)
    upper = _attachment_upper_mask(active.device)[None]
    endpoint_active = active.repeat_interleave(2, dim=1)
    eligible = endpoint_active[:, :, None] & endpoint_active[:, None, :] & validity & upper
    target_all = aligned["attachment"].reshape(batch, 64, 64).bool() & validity & upper
    overlap = aligned["overlap"].bool().repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    return PredictedGeometryAssociationSlice(
        geometry[eligible].detach().cpu().numpy().astype(np.float32, copy=False),
        learned[eligible].detach().cpu().numpy().astype(np.float32, copy=False),
        target_all[eligible].detach().cpu().numpy().astype(np.bool_, copy=False),
        (overlap & ~target_all)[eligible].detach().cpu().numpy().astype(np.bool_, copy=False),
        int(eligible.sum()), int(target_all.sum()),
    )


__all__ = ["PredictedGeometryAssociationSlice", "predicted_geometry_association_slice"]
