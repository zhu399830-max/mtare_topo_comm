"""Small signed event-center head on a frozen causal action-set context."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_DIM = 128
MAXIMUM_OFFSET_M = 12.0
LOCAL_CENTER_SCALE_M = (12.0, 5.0, 5.0)


class EventCenterOffsetHead(nn.Module):
    """Predict route-tangent signed distance from sensor to event center."""

    def __init__(self) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, causal_context: torch.Tensor) -> torch.Tensor:
        if causal_context.ndim != 2 or causal_context.shape[1] != CONTEXT_DIM:
            raise ValueError("event-center context must have shape [B,128]")
        if not bool(torch.isfinite(causal_context).all()):
            raise ValueError("event-center context is nonfinite")
        return MAXIMUM_OFFSET_M * torch.tanh(self.regressor(causal_context).squeeze(1))


class EventCenterVectorHead(nn.Module):
    """Predict forward/lateral/up event-center coordinates in the route frame."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(CONTEXT_DIM, 64), nn.SiLU())
        self.longitudinal_output = nn.Linear(64, 1)
        self.transverse_output = nn.Linear(64, 2)
        self.register_buffer("scale_m", torch.tensor(LOCAL_CENTER_SCALE_M))

    def initialize_from_scalar_state(self, scalar_state: Mapping[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.hidden[0].weight.copy_(scalar_state["regressor.0.weight"])
            self.hidden[0].bias.copy_(scalar_state["regressor.0.bias"])
            self.longitudinal_output.weight.copy_(scalar_state["regressor.2.weight"])
            self.longitudinal_output.bias.copy_(scalar_state["regressor.2.bias"])
            self.transverse_output.weight.zero_(); self.transverse_output.bias.zero_()

    def forward(self, causal_context: torch.Tensor) -> torch.Tensor:
        if causal_context.ndim != 2 or causal_context.shape[1] != CONTEXT_DIM:
            raise ValueError("event-center vector context must have shape [B,128]")
        if not bool(torch.isfinite(causal_context).all()):
            raise ValueError("event-center vector context is nonfinite")
        hidden = self.hidden(causal_context)
        raw = torch.cat((self.longitudinal_output(hidden), self.transverse_output(hidden)), dim=1)
        return self.scale_m * torch.tanh(raw)


def event_center_offset_loss(
    predicted_offset_m: torch.Tensor,
    target_offset_m: torch.Tensor,
    event_index: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Use equal junction/terminal mass so long junction episodes do not dominate."""

    if (
        predicted_offset_m.ndim != 1 or target_offset_m.shape != predicted_offset_m.shape
        or event_index.shape != predicted_offset_m.shape
        or not bool(torch.isfinite(predicted_offset_m).all())
        or not bool(torch.isfinite(target_offset_m).all())
        or bool(((event_index != 1) & (event_index != 2)).any())
        or bool((target_offset_m.abs() > MAXIMUM_OFFSET_M + 1e-6).any())
    ):
        raise ValueError("event-center offset loss contract drift")
    values = {}
    terms = []
    for event, name in ((1, "junction"), (2, "terminal")):
        mask = event_index == event
        if not bool(mask.any()):
            raise ValueError("event-center batch lacks one decision class")
        loss = F.smooth_l1_loss(predicted_offset_m[mask], target_offset_m[mask], beta=1.0)
        values[name] = loss
        terms.append(loss)
    values["total"] = torch.stack(terms).mean()
    return values


def event_center_paired_loss(
    predicted_offset_m: torch.Tensor,
    target_offset_m: torch.Tensor,
    event_index: torch.Tensor,
    sensor_xyz_m: torch.Tensor,
    route_tangent_xyz: torch.Tensor,
    pair_index: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Add cross-traversal relative-center supervision without GT at runtime."""

    direct = event_center_offset_loss(predicted_offset_m, target_offset_m, event_index)
    relative = event_center_relative_loss(
        predicted_offset_m, target_offset_m, event_index, sensor_xyz_m,
        route_tangent_xyz, pair_index,
    )
    return {
        "junction": direct["junction"], "terminal": direct["terminal"],
        "relative_center": relative, "total": direct["total"] + relative,
    }


def event_center_relative_loss(
    predicted_offset_m: torch.Tensor,
    target_offset_m: torch.Tensor,
    event_index: torch.Tensor,
    sensor_xyz_m: torch.Tensor,
    route_tangent_xyz: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """Relative 3D center error for identity-matched cross-traversal pairs."""

    if (
        sensor_xyz_m.shape != (len(predicted_offset_m), 3)
        or route_tangent_xyz.shape != sensor_xyz_m.shape
        or pair_index.ndim != 2 or pair_index.shape[1] != 2 or len(pair_index) == 0
        or pair_index.dtype != torch.long
        or bool((pair_index < 0).any()) or bool((pair_index >= len(predicted_offset_m)).any())
        or not bool(torch.isfinite(sensor_xyz_m).all())
        or not bool(torch.isfinite(route_tangent_xyz).all())
        or bool((event_index[pair_index[:, 0]] != event_index[pair_index[:, 1]]).any())
    ):
        raise ValueError("event-center paired loss contract drift")
    predicted_center = sensor_xyz_m + predicted_offset_m[:, None] * route_tangent_xyz
    target_center = sensor_xyz_m + target_offset_m[:, None] * route_tangent_xyz
    predicted_delta = predicted_center[pair_index[:, 0]] - predicted_center[pair_index[:, 1]]
    target_delta = target_center[pair_index[:, 0]] - target_center[pair_index[:, 1]]
    return F.smooth_l1_loss(predicted_delta, target_delta, beta=1.0)


__all__ = [
    "CONTEXT_DIM", "MAXIMUM_OFFSET_M", "LOCAL_CENTER_SCALE_M", "EventCenterOffsetHead", "EventCenterVectorHead",
    "event_center_offset_loss", "event_center_paired_loss", "event_center_relative_loss",
]
