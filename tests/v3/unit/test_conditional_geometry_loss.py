import numpy as np
import torch
import pytest
from dataclasses import replace
from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets,conditional_geometry_loss
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
from test_structural_representation import inputs


def test_frame_order_without_clock():
    x,p=inputs(); c=CausalFrameOrderContext('sensor',(10,11,12,13,14),14,'cached')
    r=GeometryStructureEncoder('C')(x,p,(c,)).relations
    assert r.computation_valid.any() and not hasattr(c,'timestamp_s')
    with pytest.raises(ValueError): replace(c,observation_frame_id=13)


def test_category_weight_and_unknown_gradient():
    x,p=inputs(); c=CausalFrameOrderContext('sensor',(0,1,2,3,4),4,'cached')
    r=GeometryStructureEncoder('C')(x,p,(c,)).relations
    n=r.neighbor_index[0].numpy(); known=r.computation_valid[0].numpy()
    comp=np.array([0,1,0]); same=np.zeros_like(known)
    for i,k in zip(*np.nonzero(known)): same[i,k]=comp[i]==comp[n[i,k]]
    arrays=dict(schema_version=np.array('construction_conditioned_geometry_targets_v1'),observability_certified=False,
        axis_known=known,height_known=known,neighbor_index=n,patch_reference_component=comp,same_reference_component=same,
        axis_abs_dot=np.where(known,.5,np.nan),height_difference_m=np.where(known,2.,np.nan))
    t=compile_conditional_loss_targets(arrays)
    assert torch.allclose(t.weights.sum(),torch.tensor(1.))
    if same.any(): assert torch.allclose(t.weights[t.same].sum(),torch.tensor(.5))
    r.height_difference_m.retain_grad(); loss,_=conditional_geometry_loss(r,t);loss.backward()
    assert torch.isfinite(loss) and not r.height_difference_m.grad[~t.known].any()
    arrays['observability_certified']=True
    with pytest.raises(ValueError): compile_conditional_loss_targets(arrays)
