"""Reproduce the current contract's boundary blind spot; do not change losses."""
from dataclasses import replace
import torch
from tests.v3.unit.test_gse_surface_losses_v1 import prediction,targets
from mtare_topo.representation.gse_surface_losses_v1 import surface_relation_losses


def gradient_at(radius):
    p=prediction();t=targets()
    xyz=p.opening_position_m.detach().clone();xyz[0,3]=torch.tensor([radius,0.,0.],dtype=xyz.dtype)
    p=replace(p,opening_position_m=xyz.requires_grad_())
    t=replace(t,opening_region_complete=torch.ones_like(t.opening_region_complete))
    loss=surface_relation_losses(p,t)
    assert 3 not in loss.assignments['opening'][0].tolist()
    loss.terms['opening_presence'].backward()
    return float(p.opening_presence_logits.grad[0,3]),loss.denominators['opening_presence']


def test_unmatched_query_crossing_window_loses_presence_penalty():
    inside,n=gradient_at(10.-1e-4)
    outside,m=gradient_at(10.+1e-4)
    assert inside>0 and outside==0
    assert n==m+1


def test_on_boundary_is_supervised_but_any_fixed_positive_offset_is_not():
    assert gradient_at(10.)[0]>0
    assert gradient_at(10.+1e-6)[0]==0
