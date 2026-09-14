"""Versioned smooth anchor domain; not permission to rerun a sealed experiment."""
import torch
from .gse_block_structure_readout import BlockStructureReadout


def smooth_anchor_coordinates(raw):
    """10*u/sqrt(1+|u|²), rotation-equivariant and locally 10 metres/unit.

    No finite real raw radius has a hard radial plateau. Floating point still
    has finite precision and large-radius gradients become small; this is not
    a global conditioning guarantee. The exact boundary is a limiting value.
    Sequential hypot avoids overflow from explicitly squaring raw values.
    """
    if raw.ndim!=2 or raw.shape[1]!=3 or raw.dtype not in (torch.float32,torch.float64):
        raise ValueError('N by 3 float32/64 raw coordinates required')
    if not torch.isfinite(raw).all():raise ValueError('nonfinite raw coordinate')
    denominator=torch.ones_like(raw[:,:1])
    for dim in range(3):denominator=torch.hypot(denominator,raw[:,dim:dim+1])
    if not torch.isfinite(denominator).all():raise ValueError('coordinate norm overflow')
    return (raw/denominator)*10.


class BlockStructureReadoutV2(BlockStructureReadout):
    """Same parameters and heads; only the anchor coordinate map changes."""
    def anchor_coordinates(self,raw):
        return smooth_anchor_coordinates(raw)
