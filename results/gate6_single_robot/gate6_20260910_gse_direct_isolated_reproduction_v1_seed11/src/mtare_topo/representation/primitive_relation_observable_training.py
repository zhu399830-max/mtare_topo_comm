"""Loss and batch conversion for observable endpoint relations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationNumpyBatch
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
    _balanced_binary_loss,
    align_primitive_relation_targets,
    match_primitives,
)
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationPrediction
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_port_relation_losses,
)
from mtare_topo.representation.primitive_relation_training import numpy_batch_to_torch


@dataclass(frozen=True)
class ObservablePrimitiveRelationTorchBatch:
    range_valid: torch.Tensor
    relative_translation_current_sensor_m: torch.Tensor
    relative_yaw_current_sensor_deg: torch.Tensor
    targets: PrimitiveRelationLossTargets


def observable_numpy_batch_to_torch(
    batch: ObservablePrimitiveRelationNumpyBatch,
    *,
    device: torch.device,
) -> ObservablePrimitiveRelationTorchBatch:
    base = numpy_batch_to_torch(batch.base, device=device)
    endpoint_observed = torch.from_numpy(batch.endpoint_observed).to(device=device, dtype=torch.float32)
    targets = replace(base.targets, endpoint_observed=endpoint_observed)
    targets.validate()
    return ObservablePrimitiveRelationTorchBatch(
        range_valid=base.range_valid,
        relative_translation_current_sensor_m=base.relative_translation_current_sensor_m,
        relative_yaw_current_sensor_deg=base.relative_yaw_current_sensor_deg,
        targets=targets,
    )


def observable_primitive_relation_losses(
    prediction: ObservableSparsePortRelationPrediction,
    targets: PrimitiveRelationLossTargets,
    range_valid: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Replace the old port loss with attachment/overlap/evidence thirds."""

    if targets.endpoint_observed is None:
        raise ValueError("observable relation loss requires endpoint evidence targets")
    families = dict(sparse_port_relation_losses(prediction, targets, range_valid))
    assignments = match_primitives(prediction, targets)
    aligned = align_primitive_relation_targets(targets, assignments)
    # Endpoint evidence is defined for every query: unsupported crop ends and
    # inactive queries are both zero.  This does not reveal hidden physical
    # connectivity; it only teaches whether local sensor evidence exists.
    evidence_loss = _balanced_binary_loss(
        prediction.endpoint_evidence_logits.reshape(-1),
        aligned["endpoint_observed"].reshape(-1),
    )
    families["port_relations"] = (2.0 * families["port_relations"] + evidence_loss) / 3.0
    families["total"] = torch.stack(tuple(
        value for name, value in families.items() if name != "total"
    )).mean()
    return families


__all__ = [
    "ObservablePrimitiveRelationTorchBatch", "observable_numpy_batch_to_torch",
    "observable_primitive_relation_losses",
]
