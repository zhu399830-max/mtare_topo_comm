"""Geometry-only set fitting for a bounded axis-readout experiment.

Hungarian assignment is detached, uses only coordinate L1 and endpoint
reversal, and has no identity, relation, existence or threshold input. Every
active teacher receives an assignment: these fitting scores are NOT detection
precision, learned segmentation, or parity with the old relation evaluator.
Tied assignment identities may change with slot ordering; the minimum fitting
cost does not acquire a tie-breaking perturbation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


COORDINATE_SCALE_M = 50.0


def _validate(pred, target, mask):
    if not all(isinstance(value, torch.Tensor) for value in (pred, target, mask)):
        raise ValueError("prediction, target and mask must be tensors")
    if pred.ndim != 4 or pred.shape[-2:] != (3, 3):
        raise ValueError("prediction must be [B,S,3,3]")
    b, s = pred.shape[:2]
    if target.ndim != 4 or target.shape[0] != b or target.shape[-2:] != (3, 3):
        raise ValueError("target must be [B,K,3,3]")
    if b < 1 or s < 1 or target.shape[1] < 1:
        raise ValueError("nonempty batch and slot dimensions are required")
    if mask.shape != target.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,K]")
    if pred.dtype not in (torch.float32, torch.float64) or target.dtype != pred.dtype:
        raise ValueError("axes must have common float32 or float64 dtype")
    if pred.device != target.device or pred.device != mask.device:
        raise ValueError("axes and mask must share a device")
    counts = mask.sum(dim=1)
    if bool(((counts < 1) | (counts > s)).any()):
        raise ValueError("each observation needs 1..S active targets")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("predictions and active targets must be finite")


def _assignments(pred, target, mask):
    """Return prediction slots, original target slots and reversal per row."""
    _validate(pred, target, mask)
    output = []
    for batch in range(len(pred)):
        active = torch.nonzero(mask[batch], as_tuple=False).flatten()
        prediction_cpu = pred[batch].detach().to(device="cpu", dtype=torch.float64).numpy()
        teacher_cpu = target[batch, active].detach().to(device="cpu", dtype=torch.float64).numpy()
        direct = np.abs(prediction_cpu[:, None] - teacher_cpu[None]).mean(axis=(-1, -2))
        reverse = np.abs(prediction_cpu[:, None] - teacher_cpu[None, :, ::-1]).mean(axis=(-1, -2))
        cost = np.minimum(direct, reverse)
        if not np.isfinite(cost).all():
            raise ValueError("geometry assignment cost is nonfinite")
        rows, columns = linear_sum_assignment(cost)
        active_cpu = active.detach().cpu().numpy()
        output.append((rows, active_cpu[columns], (reverse < direct)[rows, columns]))
    return output


def _aligned_row(pred, target, batch, assignment):
    rows, columns, reverse = assignment
    row_index = torch.as_tensor(rows, device=pred.device, dtype=torch.long)
    target_index = torch.as_tensor(columns, device=pred.device, dtype=torch.long)
    reversed_mask = torch.as_tensor(reverse, device=pred.device, dtype=torch.bool)
    selected = target[batch, target_index]
    aligned = torch.where(reversed_mask[:, None, None], selected.flip(1), selected)
    return pred[batch, row_index], aligned


def axis_set_loss(pred, target, mask):
    """Mean coordinate L1 / 50, first per observation then across the batch."""
    assignments = _assignments(pred, target, mask)
    losses = []
    for batch, assignment in enumerate(assignments):
        prediction, teacher = _aligned_row(pred, target, batch, assignment)
        losses.append(torch.abs(prediction - teacher).mean() / COORDINATE_SCALE_M)
    result = torch.stack(losses).mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("axis set loss is nonfinite")
    return result


def axis_set_metrics(pred, target, mask):
    """All-target fitting metrics, never a detector acceptance decision."""
    assignments = _assignments(pred, target, mask)
    result = []
    for batch, assignment in enumerate(assignments):
        prediction, teacher = _aligned_row(pred, target, batch, assignment)
        residual = prediction.detach().double() - teacher.detach().double()
        coordinate = residual.abs().mean(dim=(-1, -2)).cpu().tolist()
        point = torch.linalg.vector_norm(residual, dim=-1).mean(dim=-1).cpu().tolist()
        rows, columns, reverse = assignment
        count = len(rows)
        result.append({
            "n_targets": count,
            "n_predictions": pred.shape[1],
            "unmatched_predictions": pred.shape[1] - count,
            "coordinate_mae_m": float(np.mean(coordinate)),
            "point_mean_euclidean_m": float(np.mean(point)),
            "matching": [{"prediction_index": int(rows[i]), "target_index": int(columns[i]),
                          "reversed": bool(reverse[i]), "coordinate_mae_m": coordinate[i],
                          "point_mean_euclidean_m": point[i]} for i in range(count)],
        })
    return result


__all__ = ["axis_set_loss", "axis_set_metrics"]
