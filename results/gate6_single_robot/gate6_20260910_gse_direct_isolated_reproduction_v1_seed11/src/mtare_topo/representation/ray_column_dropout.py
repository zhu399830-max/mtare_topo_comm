"""Deterministic train-only ray-column dropout for the Phase-3 corrective."""

from __future__ import annotations

import torch


def apply_ray_column_dropout(
    student: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.5,
    period: int = 10,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Mask one random periodic phase for selected samples; leave input unchanged."""
    if student.ndim != 4 or tuple(student.shape[1:]) != (2, 16, 720):
        raise ValueError(f"expected [B,2,16,720], got {tuple(student.shape)}")
    if not 0.0 <= probability <= 1.0 or period <= 0 or 720 % period:
        raise ValueError("invalid probability or period")
    augmented=student.clone();selected=torch.rand(len(student),generator=generator)<probability
    phases=torch.randint(0,period,(len(student),),generator=generator);phase_counts=[0]*period
    for index in torch.nonzero(selected,as_tuple=False).flatten().tolist():
        phase=int(phases[index]);phase_counts[phase]+=1
        augmented[index,0,:,phase::period]=1.0
        augmented[index,1,:,phase::period]=0.0
    return augmented,{"samples":len(student),"augmented_samples":int(selected.sum()),"phase_counts":phase_counts}
