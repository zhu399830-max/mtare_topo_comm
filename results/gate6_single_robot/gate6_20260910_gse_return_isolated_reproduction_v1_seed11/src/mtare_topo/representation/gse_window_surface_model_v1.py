"""Explicit corrective model version: window openings constrained to a sphere.

No new parameters or backbone changes. Old models/checkpoints remain unchanged.
Raw opening position readout now parametrizes a direction, not an unconstrained
metric point. This requires the separate window loss and scoring contract.
"""
from dataclasses import replace
from .gse_surface_relation_model_v1 import SurfaceRelationModelV1
from .gse_window_surface_domain import window_surface_positions


class WindowSurfaceModelV1(SurfaceRelationModelV1):
    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        return replace(result,opening_position_m=window_surface_positions(
            result.opening_position_m,result.observation_supported))
