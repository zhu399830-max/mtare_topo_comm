"""Deployment-only helpers for frozen causal episode inference."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def align_baseline_event_logits(
    output_global_sequence_index: Sequence[int],
    output_parent_id: Sequence[str],
    output_event_logits: np.ndarray,
    deployment_global_sequence_index: Sequence[int],
    deployment_parent_id: Sequence[str],
) -> np.ndarray:
    """Align arbitrarily ordered frozen outputs to deployment identity order."""

    output_key = np.asarray(output_global_sequence_index, dtype=np.int64)
    output_parent = np.asarray(output_parent_id, dtype=str)
    logits = np.asarray(output_event_logits, dtype=np.float32)
    deployment_key = np.asarray(deployment_global_sequence_index, dtype=np.int64)
    deployment_parent = np.asarray(deployment_parent_id, dtype=str)
    count = len(deployment_key)
    if (
        count == 0
        or output_key.shape != (count,)
        or output_parent.shape != (count,)
        or logits.shape != (count, 5)
        or deployment_parent.shape != (count,)
        or len(np.unique(output_key)) != count
        or len(np.unique(deployment_key)) != count
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError("causal baseline population contract drift")
    output_order = np.argsort(output_key, kind="stable")
    deployment_order = np.argsort(deployment_key, kind="stable")
    if (
        not np.array_equal(output_key[output_order], deployment_key[deployment_order])
        or not np.array_equal(output_parent[output_order], deployment_parent[deployment_order])
    ):
        raise RuntimeError("causal baseline and deployment identities are not bijective")
    aligned = np.empty_like(logits)
    aligned[deployment_order] = logits[output_order]
    return aligned


__all__ = ["align_baseline_event_logits"]
