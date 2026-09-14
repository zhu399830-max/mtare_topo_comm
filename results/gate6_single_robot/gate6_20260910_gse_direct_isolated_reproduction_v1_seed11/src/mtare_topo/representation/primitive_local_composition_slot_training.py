"""Permutation-safe targets and losses for local composition slots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from mtare_topo.evaluation.local_composition_slot_teacher import (
    decompose_observable_attachment,
)
from mtare_topo.representation.primitive_local_composition_slot_model import (
    COMPOSITION_SLOT_COUNT,
    DUSTBIN_INDEX,
    ENDPOINT_COUNT,
    LocalCompositionSlotPrediction,
    composition_slot_relation_probability,
)
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveAssignment,
    PrimitiveRelationLossTargets,
    align_primitive_relation_targets,
)


UNKNOWN_LABEL = -2
TEACHER_DUSTBIN_LABEL = -1


@dataclass(frozen=True)
class LocalCompositionSlotTargets:
    labels: torch.Tensor
    endpoint_supervised: torch.Tensor
    disconnected_overlap: torch.Tensor
    cluster_count: torch.Tensor

    def validate(self) -> None:
        batch = self.labels.shape[0]
        expected = {
            "labels": (batch, ENDPOINT_COUNT),
            "endpoint_supervised": (batch, ENDPOINT_COUNT),
            "disconnected_overlap": (batch, ENDPOINT_COUNT, ENDPOINT_COUNT),
            "cluster_count": (batch,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"local composition-slot target shape drift: {name}")
        if self.labels.dtype != torch.long or self.cluster_count.dtype != torch.long:
            raise ValueError("local composition-slot labels/counts must be int64")
        if self.endpoint_supervised.dtype != torch.bool or self.disconnected_overlap.dtype != torch.bool:
            raise ValueError("local composition-slot masks must be boolean")
        if bool((self.cluster_count < 0).any()) or bool(
            (self.cluster_count > COMPOSITION_SLOT_COUNT).any()
        ):
            raise ValueError("local composition-slot target capacity overflow")
        if bool((self.labels[self.endpoint_supervised] < TEACHER_DUSTBIN_LABEL).any()):
            raise ValueError("supervised endpoint has an unknown target")
        if bool((self.labels[~self.endpoint_supervised] != UNKNOWN_LABEL).any()):
            raise ValueError("unsupervised endpoint must keep the unknown label")


def align_local_composition_slot_targets(
    targets: PrimitiveRelationLossTargets,
    assignments: tuple[PrimitiveAssignment, ...],
) -> LocalCompositionSlotTargets:
    """Align Teacher relations to primitive queries, then form clique labels."""

    aligned = align_primitive_relation_targets(targets, assignments)
    observed = aligned["endpoint_observed"].reshape(-1, ENDPOINT_COUNT).bool()
    attachment = aligned["attachment"].reshape(-1, ENDPOINT_COUNT, ENDPOINT_COUNT).bool()
    endpoint_overlap = aligned["overlap"].bool().repeat_interleave(2, 1).repeat_interleave(2, 2)
    labels = torch.full(
        (len(observed), ENDPOINT_COUNT),
        UNKNOWN_LABEL,
        dtype=torch.long,
        device=observed.device,
    )
    counts = torch.zeros(len(observed), dtype=torch.long, device=observed.device)
    for batch_index in range(len(observed)):
        result = decompose_observable_attachment(
            attachment[batch_index].detach().cpu().numpy(),
            observed[batch_index].detach().cpu().numpy(),
            endpoint_overlap[batch_index].detach().cpu().numpy(),
        )
        if not result.is_clique_partition:
            raise ValueError("observable attachment component is not a clique")
        if not np.array_equal(result.observed_attachment, result.reconstructed_attachment):
            raise ValueError("local composition-slot target is not lossless")
        if result.overlap_violations:
            raise ValueError("local composition-slot target merges disconnected overlap")
        if result.cluster_count > COMPOSITION_SLOT_COUNT:
            raise ValueError("local composition-slot target exceeds frozen capacity")
        value = torch.from_numpy(result.labels.astype(np.int64, copy=False)).to(observed.device)
        labels[batch_index, observed[batch_index]] = value[observed[batch_index]]
        counts[batch_index] = result.cluster_count
    result = LocalCompositionSlotTargets(
        labels=labels,
        endpoint_supervised=observed,
        disconnected_overlap=endpoint_overlap,
        cluster_count=counts,
    )
    result.validate()
    return result


def _balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.bool()
    terms = []
    if bool(target.any()):
        terms.append(F.softplus(-logits[target]).mean())
    if bool((~target).any()):
        terms.append(F.softplus(logits[~target]).mean())
    if not terms:
        raise ValueError("local composition-slot binary loss has no elements")
    return torch.stack(terms).mean()


def match_local_composition_slots(
    prediction: LocalCompositionSlotPrediction,
    targets: LocalCompositionSlotTargets,
) -> tuple[torch.Tensor, ...]:
    """Match exchangeable predicted slots to each row's Teacher clusters."""

    prediction.validate()
    targets.validate()
    if len(prediction.assignment_logits) != len(targets.labels):
        raise ValueError("local composition-slot prediction/target batch mismatch")
    log_probability = F.log_softmax(prediction.assignment_logits, dim=-1)
    mappings: list[torch.Tensor] = []
    for batch_index in range(len(targets.labels)):
        count = int(targets.cluster_count[batch_index])
        mapping = torch.full(
            (count,), -1, dtype=torch.long, device=prediction.assignment_logits.device,
        )
        if count:
            columns = []
            for cluster in range(count):
                members = targets.labels[batch_index] == cluster
                if not bool(members.any()):
                    raise ValueError("local composition-slot cluster label has no endpoints")
                membership = -log_probability[batch_index, members, :COMPOSITION_SLOT_COUNT].mean(dim=0)
                presence = F.softplus(-prediction.slot_presence_logits[batch_index])
                columns.append(membership + presence)
            cost = torch.stack(columns, dim=1)
            rows, clusters = linear_sum_assignment(cost.detach().cpu().numpy())
            if len(rows) != count:
                raise RuntimeError("local composition-slot Hungarian match is incomplete")
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=mapping.device)
            cluster_tensor = torch.as_tensor(clusters, dtype=torch.long, device=mapping.device)
            mapping[cluster_tensor] = row_tensor
            if bool((mapping < 0).any()):
                raise RuntimeError("local composition-slot cluster mapping is incomplete")
        mappings.append(mapping)
    return tuple(mappings)


def local_composition_slot_losses(
    prediction: LocalCompositionSlotPrediction,
    targets: LocalCompositionSlotTargets,
) -> dict[str, torch.Tensor]:
    """Supervise cluster assignment, dustbin, slot presence and overlap refusal."""

    mappings = match_local_composition_slots(prediction, targets)
    endpoint_class = torch.full_like(targets.labels, DUSTBIN_INDEX)
    endpoint_class[~targets.endpoint_supervised] = UNKNOWN_LABEL
    slot_present = torch.zeros_like(prediction.slot_presence_logits, dtype=torch.bool)
    for batch_index, mapping in enumerate(mappings):
        for cluster, slot in enumerate(mapping.tolist()):
            endpoint_class[batch_index, targets.labels[batch_index] == cluster] = slot
            slot_present[batch_index, slot] = True
    clustered = targets.endpoint_supervised & (targets.labels >= 0)
    dustbin = targets.endpoint_supervised & (targets.labels == TEACHER_DUSTBIN_LABEL)
    zero = prediction.assignment_logits.sum() * 0.0
    cluster_assignment = (
        F.cross_entropy(prediction.assignment_logits[clustered], endpoint_class[clustered])
        if bool(clustered.any()) else zero
    )
    dustbin_assignment = (
        F.cross_entropy(prediction.assignment_logits[dustbin], endpoint_class[dustbin])
        if bool(dustbin.any()) else zero
    )
    slot_presence = _balanced_binary_loss(prediction.slot_presence_logits, slot_present)
    relation = composition_slot_relation_probability(prediction)
    upper = torch.triu(
        torch.ones(ENDPOINT_COUNT, ENDPOINT_COUNT, dtype=torch.bool, device=relation.device),
        diagonal=1,
    )
    overlap = (
        targets.disconnected_overlap
        & targets.endpoint_supervised[:, :, None]
        & targets.endpoint_supervised[:, None, :]
        & upper[None]
    )
    overlap_rejection = (
        -torch.log1p(-relation[overlap].clamp(max=1.0 - 1e-6)).mean()
        if bool(overlap.any()) else zero
    )
    components = (
        cluster_assignment,
        dustbin_assignment,
        slot_presence,
        overlap_rejection,
    )
    total = torch.stack(components).mean()
    return {
        "cluster_assignment": cluster_assignment,
        "dustbin_assignment": dustbin_assignment,
        "slot_presence": slot_presence,
        "overlap_rejection": overlap_rejection,
        "total": total,
    }


__all__ = [
    "LocalCompositionSlotTargets",
    "TEACHER_DUSTBIN_LABEL",
    "UNKNOWN_LABEL",
    "align_local_composition_slot_targets",
    "local_composition_slot_losses",
    "match_local_composition_slots",
]
