import copy
import numpy as np
import pytest
import torch
from test_gse_reference_exclusion_binding_v1 import fixture as source_fixture
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.anchor_branch_loss import AnchorBranchPrediction, PartialAnchorBranches
from mtare_topo.representation.conditional_anchor_branch_loss import conditional_anchor_branch_loss


def fixture():
    bundle,grid,points,binding=source_fixture()
    xyz=torch.tensor(np.stack((points[0],points[0]*.75,[0,9,0])),dtype=torch.float64,requires_grad=True)
    prediction=AnchorBranchPrediction(xyz,torch.zeros(3,dtype=torch.float64,requires_grad=True),
        torch.tensor([[[1.,0,0]]]*3,dtype=torch.float64,requires_grad=True),
        torch.zeros((3,1),dtype=torch.float64,requires_grad=True))
    target=PartialAnchorBranches(torch.zeros((1,3),dtype=torch.float64),
        (torch.empty((0,3),dtype=torch.float64),),False,(False,))
    record=dict(anchors=[dict(position_m=[0.,0.,0.])],source_frame_indices=binding['source']['frame_rows'])
    produced=dict(record=record,source_binding=binding,target_record_sha256=canonical_sha(record))
    manifest=dict(source_binding=copy.deepcopy(binding),target_record_sha256=canonical_sha(record),target_reference_indices=(0,))
    return prediction,target,dict(bundle=bundle,grid=grid,produced_targets=produced,frozen_manifest=manifest)


def test_exact_observed_negative_matched_positive_priority_unknown_and_branch_untouched():
    p,t,kwargs=fixture();result=conditional_anchor_branch_loss(p,t,**kwargs)
    assert result['anchor_assignment'].tolist()==[1]
    assert result['conditional_anchor_evidence']['used_negative_query_indices']==[0]
    assert result['counts']['anchor_presence']==2
    result['total'].backward()
    assert p.presence_logits.grad[0]>0 and p.presence_logits.grad[1]<0 and p.presence_logits.grad[2]==0
    assert p.branch_logits.grad is None and p.directions.grad is None


@pytest.mark.parametrize('damage',['positions','hash','source','indices'])
def test_frozen_positive_mismatch_rejected(damage):
    p,t,k=fixture()
    if damage=='positions':t.position_m[0,0]=1
    if damage=='hash':k['produced_targets']['target_record_sha256']='bad'
    if damage=='source':k['frozen_manifest']['source_binding']['source']['frame_rows'][-1]=8
    if damage=='indices':k['frozen_manifest']['target_reference_indices']=(0,0)
    with pytest.raises(ValueError):conditional_anchor_branch_loss(p,t,**k)
