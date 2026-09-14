import pytest
import torch
from mtare_topo.representation.gse_window_surface_domain import window_surface_positions,complete_window_query_mask


def test_radial_scaling_cannot_escape_query_domain():
    x=torch.tensor([[[1.,2.,3.],[0.,-1.,2.]]],dtype=torch.float64,requires_grad=True)
    supported=torch.tensor([True]);p=window_surface_positions(x,supported)
    torch.testing.assert_close(window_surface_positions(100*x,supported),p)
    torch.testing.assert_close(torch.linalg.vector_norm(p,dim=-1),torch.full((1,2),10.,dtype=x.dtype))
    p[...,0].sum().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert complete_window_query_mask(supported,torch.tensor([True]),2).all()


def test_rotation_and_query_permutation_equivariance():
    x=torch.tensor([[[1.,2.,3.],[2.,-1.,1.]]],dtype=torch.float64);s=torch.tensor([True])
    r=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=x.dtype)
    torch.testing.assert_close(window_surface_positions(x@r.T,s),window_surface_positions(x,s)@r.T)
    torch.testing.assert_close(window_surface_positions(x.flip(1),s),window_surface_positions(x,s).flip(1))


def test_unknown_not_promoted_and_degenerate_query_not_hidden():
    assert not complete_window_query_mask(torch.tensor([True]),torch.tensor([False]),64).any()
    with pytest.raises(ValueError):window_surface_positions(torch.zeros(1,64,3),torch.tensor([True]))
    assert not window_surface_positions(torch.zeros(1,64,3),torch.tensor([False])).any()
