"""Regression evidence for the existing head, not a corrected model."""
import torch
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout


def radial_gradient(raw_x):
    torch.manual_seed(0)
    model=BlockStructureReadout().double()
    with torch.no_grad():
        model.anchor.weight.zero_()
        model.anchor.bias.copy_(torch.tensor([raw_x,0.,0.,0.],dtype=torch.float64))
    features=dict(embedding=torch.zeros(1,128,dtype=torch.float64),
        centers_m=torch.zeros(1,3,dtype=torch.float64),extent_m=torch.ones(1,3,dtype=torch.float64),
        degenerate=torch.zeros(1,dtype=torch.bool))
    prediction=model(features,torch.zeros(1,128,dtype=torch.float64))
    # The production positive position loss is Euclidean error / 10.
    loss=torch.linalg.vector_norm(prediction.anchor_position_m[0])/10
    gradient=torch.autograd.grad(loss,model.anchor.bias)[0]
    return prediction.anchor_position_m[0].detach(),gradient


def test_saturated_anchor_has_no_gradient_toward_central_target():
    xyz,gradient=radial_gradient(2.)
    torch.testing.assert_close(xyz,torch.tensor([10.,0.,0.],dtype=torch.float64))
    torch.testing.assert_close(gradient,torch.zeros(4,dtype=torch.float64),atol=1e-12,rtol=0)


def test_interior_anchor_retains_gradient_toward_central_target():
    xyz,gradient=radial_gradient(.2)
    torch.testing.assert_close(xyz,torch.tensor([2.,0.,0.],dtype=torch.float64))
    torch.testing.assert_close(gradient,torch.tensor([1.,0.,0.,0.],dtype=torch.float64))
