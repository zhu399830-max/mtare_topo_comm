"""Experimental fixed-window opening domain, not wired into frozen models.

Openings live on the window surface; anchors remain independent metric points.
This removes a learnable radial degree of freedom, not unknown observations.
"""
import torch


def window_surface_positions(raw_direction, observation_supported, *, radius_m=10.):
    if radius_m!=10.:raise ValueError('fixed10m window contract required')
    if raw_direction.ndim!=3 or raw_direction.shape[-1]!=3 or raw_direction.dtype not in (torch.float32,torch.float64):
        raise ValueError('batched floating direction vectors required')
    if observation_supported.shape!=(raw_direction.shape[0],) or observation_supported.dtype!=torch.bool or observation_supported.device!=raw_direction.device:
        raise ValueError('input-derived observation support required')
    if not torch.isfinite(raw_direction).all():raise ValueError('nonfinite direction')
    norm=torch.linalg.vector_norm(raw_direction,dim=-1)
    minimum=32*torch.finfo(raw_direction.dtype).eps
    if ((norm<=minimum)&observation_supported[:,None]).any():
        raise ValueError('degenerate supported query: cannot hide a prediction by setting validity false')
    unit=raw_direction/norm.clamp_min(minimum)[...,None]
    return torch.where(observation_supported[:,None,None],radius_m*unit,0.)


def complete_window_query_mask(observation_supported, complete_window, query_count):
    """Completeness comes from labels, never from predicted radius/confidence."""
    if observation_supported.dtype!=torch.bool or complete_window.dtype!=torch.bool or observation_supported.shape!=complete_window.shape:
        raise ValueError('explicit support and completeness booleans required')
    if type(query_count)!=int or query_count<1:raise ValueError('positive query count')
    return (observation_supported&complete_window)[:,None].expand(-1,query_count)
