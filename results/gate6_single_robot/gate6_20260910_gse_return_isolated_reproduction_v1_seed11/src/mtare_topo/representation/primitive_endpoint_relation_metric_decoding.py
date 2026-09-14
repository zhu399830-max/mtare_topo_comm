"""Deterministic one-threshold complete-link decoding for endpoint relations."""

from __future__ import annotations

import torch


def decode_complete_link_clusters(
    pair_score: torch.Tensor,
    endpoint_active: torch.Tensor,
    *,
    confidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("endpoint relation threshold must lie in [0,1]")
    batch, endpoints, other = pair_score.shape
    if endpoints != other or endpoint_active.shape != (batch, endpoints):
        raise ValueError("endpoint relation decode shape drift")
    if not torch.equal(pair_score, pair_score.transpose(1, 2)) or not bool(torch.isfinite(pair_score).all()):
        raise ValueError("endpoint relation decode requires finite symmetric scores")
    labels = torch.full((batch, endpoints), -1, dtype=torch.long, device=pair_score.device)
    attachment = torch.zeros_like(pair_score, dtype=torch.bool)
    for row in range(batch):
        active = torch.nonzero(endpoint_active[row], as_tuple=False).flatten().tolist()
        clusters = [{index} for index in active]
        edges = [
            (-float(pair_score[row, left, right]), left, right)
            for offset, left in enumerate(active) for right in active[offset + 1:]
            if float(pair_score[row, left, right]) >= confidence_threshold
        ]
        edges.sort()
        for _, left, right in edges:
            left_cluster = next((value for value in clusters if left in value), None)
            right_cluster = next((value for value in clusters if right in value), None)
            if left_cluster is None or right_cluster is None or left_cluster is right_cluster:
                continue
            if all(
                float(pair_score[row, first, second]) >= confidence_threshold
                for first in left_cluster for second in right_cluster
            ):
                left_cluster.update(right_cluster); clusters.remove(right_cluster)
        nontrivial = sorted((sorted(value) for value in clusters if len(value) >= 2), key=lambda value: value[0])
        for cluster_id, members in enumerate(nontrivial):
            labels[row, members] = cluster_id
            member = torch.as_tensor(members, device=pair_score.device)
            attachment[row, member[:, None], member[None, :]] = True
    diagonal = torch.eye(endpoints, dtype=torch.bool, device=pair_score.device)[None]
    attachment &= ~diagonal
    return labels, attachment


__all__ = ["decode_complete_link_clusters"]
