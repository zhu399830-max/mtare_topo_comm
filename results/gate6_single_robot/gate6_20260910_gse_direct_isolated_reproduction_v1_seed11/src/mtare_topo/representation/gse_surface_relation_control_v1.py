"""Matched-neighbor explicit-relation ablation of the existing prototype.

C0 retains coordinate-bearing unary features and neighborhood topology, so it
is NOT geometry-free. It removes only the nine explicit edge attributes.
No new decoder, teacher, learned parameters or historical model edits.
"""
from dataclasses import replace
import torch
from .gse_surface_relation_model_v1 import SurfaceRelationModelV1, SurfacePatchBatch


class SurfaceRelationControlV1(SurfaceRelationModelV1):
    def __init__(self, relation_attributes=True):
        if type(relation_attributes) is not bool:
            raise ValueError('relation_attributes must be bool')
        super().__init__('C')
        self.relation_attributes=relation_attributes

    def _patch(self, patch, b, dtype, device):
        if not self.relation_attributes:
            if not isinstance(patch,SurfacePatchBatch):
                raise ValueError('surface patch batch required')
            if not bool(torch.isfinite(patch.relation).all()):
                raise ValueError('nonfinite relation input, even when ablated')
            patch=replace(patch,relation=torch.zeros_like(patch.relation))
        return super()._patch(patch,b,dtype,device)
