import math
import pytest
import torch
from mtare_topo.representation.branch_selection_objective_v1 import branch_selection_objective as objective


def vectors(angles):
    a=torch.tensor(angles,dtype=torch.float64)*math.pi/180
    return torch.stack((a.cos(),a.sin(),torch.zeros_like(a)),1)


def test_close_geometry_uses_confidence_and_permutation():
    d=vectors([1,1.1,90]); l=torch.tensor([-4.,4.,-4.],dtype=torch.float64)
    t=vectors([0]); r=objective(d,l,t,complete=True)
    assert r['assignment'].tolist()==[1]
    order=torch.tensor([2,0,1]); s=objective(d[order],l[order],t,complete=True)
    assert order[s['assignment']].tolist()==[1]
    assert torch.equal(r['total'],s['total'])


def test_group_gradient_balance_and_unknown():
    d=vectors([0,90,180,270]); t=vectors([0])
    for complete in (False,True):
        l=torch.zeros(4,dtype=torch.float64,requires_grad=True)
        r=objective(d,l,t,complete=complete);r['total'].backward()
        assert l.grad[0]<0
        if complete: assert torch.allclose(-l.grad[0],l.grad[1:].sum())
        else: assert torch.equal(l.grad[1:],torch.zeros(3,dtype=l.dtype))


def test_empty_unknown_disconnected_and_known_empty_negative():
    for complete in (False,True):
        l=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        r=objective(vectors([0,90]),l,vectors([]),complete=complete);r['total'].backward()
        assert r['has_supervision']==complete
        if complete: assert (l.grad>0).all()
        else: assert l.grad is None


def test_opposites_target_permutation():
    d=vectors([0,90,180]);l=torch.zeros(3,dtype=torch.float64);t=vectors([0,180])
    a=objective(d,l,t,complete=True);b=objective(d,l,t.flip(0),complete=True)
    assert a['assignment'].tolist()==[0,2] and b['assignment'].tolist()==[2,0]
    assert torch.equal(a['total'],b['total'])


def test_ambiguous_nonfinite_capacity():
    with pytest.raises(ValueError,match='ambiguous'):objective(vectors([0,0]),torch.zeros(2,dtype=torch.float64),vectors([0]),complete=True)
    with pytest.raises(ValueError,match='finite'):objective(vectors([0]),torch.tensor([float('nan')],dtype=torch.float64),vectors([0]),complete=True)
    with pytest.raises(ValueError,match='capacity'):objective(vectors([0]),torch.zeros(1,dtype=torch.float64),vectors([0,90]),complete=True)
