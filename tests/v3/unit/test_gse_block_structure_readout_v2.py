import pytest
import torch
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.gse_block_structure_readout_v2 import BlockStructureReadoutV2,smooth_anchor_coordinates


def test_zero_and_rotation_and_domain():
    torch.manual_seed(0)
    x=torch.randn(100,3,dtype=torch.float64)*5
    rotation,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    y=smooth_anchor_coordinates(x)
    torch.testing.assert_close(smooth_anchor_coordinates(x@rotation),y@rotation)
    assert (torch.linalg.vector_norm(y,dim=1)<10).all()
    torch.testing.assert_close(smooth_anchor_coordinates(torch.zeros(1,3)),torch.zeros(1,3))


@pytest.mark.parametrize('radius',[0.,.2,1.,2.,10.])
def test_jacobian_retains_radial_derivative(radius):
    raw=torch.tensor([[radius,0.,0.]],dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(smooth_anchor_coordinates,raw)[0,:,0,:]
    torch.testing.assert_close(jac[0,0],torch.tensor(10/(1+radius**2)**1.5,dtype=torch.float64))
    assert jac[0,0]>0 and torch.isfinite(jac).all()
    assert torch.autograd.gradcheck(smooth_anchor_coordinates,(raw,))


def test_actual_v2_head_center_and_near_boundary_gradient():
    head=BlockStructureReadoutV2().double()
    with torch.no_grad():
        head.anchor.weight.zero_();head.anchor.bias.copy_(torch.tensor([2.,0,0,0]))
    f=dict(embedding=torch.zeros(1,128,dtype=torch.float64),centers_m=torch.zeros(1,3,dtype=torch.float64),extent_m=torch.ones(1,3,dtype=torch.float64),degenerate=torch.zeros(1,dtype=torch.bool))
    for goal,sign in [(0.,1),(9.9,-1)]:
        p=head(f,torch.zeros(1,128,dtype=torch.float64))
        loss=torch.linalg.vector_norm(p.anchor_position_m[0]-torch.tensor([goal,0.,0.]))/10
        grad=torch.autograd.grad(loss,head.anchor.bias)[0]
        assert sign*grad[0]>0 and torch.isfinite(grad).all()


def test_state_layout_and_nonanchor_outputs_unchanged():
    torch.manual_seed(4);old=BlockStructureReadout();new=BlockStructureReadoutV2()
    new.load_state_dict(old.state_dict(),strict=True)
    f=dict(embedding=torch.randn(4,128),centers_m=torch.randn(4,3),extent_m=torch.ones(4,3),degenerate=torch.zeros(4,dtype=torch.bool));c=torch.randn(4,128)
    a,b=old(f,c),new(f,c)
    for field in ('anchor_presence_logits','opening_position_m','opening_presence_logits','opening_direction'):
        assert torch.equal(getattr(a,field),getattr(b,field))
    assert a.unknown_fields==b.unknown_fields
    empty=new({k:v[:0] for k,v in f.items()},c[:0]);assert not empty.observation_supported


def test_large_finite_input_and_invalid_input():
    x=torch.tensor([[1e20,0,0]],requires_grad=True)
    y=smooth_anchor_coordinates(x);y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError):smooth_anchor_coordinates(torch.tensor([[value,0.,0.]]))


def test_legacy_coordinate_hook_preserves_old_formula_exactly():
    x=torch.randn(32,3)
    expected=10*x/torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1.)
    assert torch.equal(BlockStructureReadout().anchor_coordinates(x),expected)
