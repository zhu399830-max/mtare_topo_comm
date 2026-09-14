import numpy as np
import torch
from types import SimpleNamespace
from test_observed_anchor_detector_v1 import observation
from mtare_topo.representation.observed_anchor_training_v1 import build_model as original
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.grouping_center_training_v1 import forward
from mtare_topo.representation.observed_anchor_detector_v1 import residual_positions


def test_only_position_rows_changed_queries_equal_output():
    a=original();b=build_model()
    for name,p in a.state_dict().items():
        q=b.state_dict()[name]
        if name in ('head.head.anchor.weight','head.head.anchor.bias'):
            assert torch.equal(p[3:],q[3:]) and torch.count_nonzero(q[:3])==0
        else:assert torch.equal(p,q)
    blocks=observation();student=dict(blocks=blocks,context=np.zeros((len(blocks.block_ids),128),np.float32))
    o=SimpleNamespace(student_representations={'PRIMITIVE':student})
    output=forward(b,o,'PRIMITIVE')
    assert torch.equal(output.prediction.position_m,output.query_positions_m)


def test_radial_projection_dead_direction_reproduced():
    # For collinear positive query/target, projection kills radial correction
    # while outside. Zero residual starts inside and preserves this gradient.
    query=torch.tensor([[4.,0,0]],dtype=torch.float64)
    gradients=[]
    for value in (1.,0.):
        raw=torch.tensor([[value,0,0]],dtype=torch.float64,requires_grad=True)
        p,_=residual_positions(query,raw)
        (p[0,0]-3.).square().backward();gradients.append(abs(raw.grad[0,0].item()))
    assert gradients[0]<1e-10 and gradients[1]>1
