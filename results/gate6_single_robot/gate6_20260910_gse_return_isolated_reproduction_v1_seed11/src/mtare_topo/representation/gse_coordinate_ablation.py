"""Coordinate-only control for a matched-budget geometry readout experiment.

Mean broadcast preserves point count, validity, token provenance and features.
Only the coordinates given to the readout change. It is NOT the historical
900-token decoder: token multiplicity and the new head are deliberately held
constant. Neither this adapter nor its tests qualify a learned detector.
"""
import torch

VARIANTS = ("raw_coordinates", "mean_broadcast_coordinates")


def coordinate_inputs(student, variant):
    """Transform the six student-only tensors; never consume scoring targets.

    Returned tensors share read-only storage with the caller, except for the
    replaced coordinates. Callers must not mutate them. Invalid raw padding is
    allowed, but valid coordinates and all frozen features must be finite.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown coordinate ablation")
    if not isinstance(student, (tuple, list)) or len(student) != 6:
        raise ValueError("exactly six student tensors required")
    if any(not isinstance(t, torch.Tensor) for t in student):
        raise ValueError("student inputs must be tensors")
    points, valid, memory, xyz, indices, slots = student
    if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
        raise ValueError("nonempty points [B,N,3] required")
    b, n, _ = points.shape
    if (memory.ndim != 3 or memory.shape[0] != b or min(memory.shape[1:]) < 1
            or xyz.shape != memory.shape[:2] + (3,)
            or slots.ndim != 3 or slots.shape[0] != b or slots.shape[1] < 1
            or slots.shape[2] != memory.shape[2]):
        raise ValueError("frozen memory/slot shape mismatch")
    if (valid.shape != (b, n) or valid.dtype != torch.bool
            or indices.shape != (b, n) or indices.dtype != torch.long):
        raise ValueError("point mask/provenance shape or dtype mismatch")
    if (points.dtype not in (torch.float32, torch.float64)
            or any(t.device != points.device for t in student)
            or any(t.dtype != points.dtype for t in (memory, xyz, slots))):
        raise ValueError("common floating dtype/device required")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("no evidence for an observation")
    if bool(((indices < 0) | (indices >= memory.shape[1])).any()):
        raise ValueError("sensor token index out of range")
    if not all(bool(torch.isfinite(t).all()) for t in (
            torch.where(valid[..., None], points, 0.), memory, xyz, slots)):
        raise ValueError("nonfinite valid geometry/features")
    if variant == "raw_coordinates":
        return tuple(student)
    means = xyz.gather(1, indices[..., None].expand(-1, -1, 3))
    means = torch.where(valid[..., None], means, 0.)
    return (means, valid, memory, xyz, indices, slots)
