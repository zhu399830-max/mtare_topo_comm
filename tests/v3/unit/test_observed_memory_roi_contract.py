"""Software evidence for explicit masking, not a trained-model ablation."""
import torch
from mtare_topo.representation.gse_observed_membership_v1 import ObservedMembershipV1


def test_outside_coordinate_and_context_are_masked_after_encoding():
    torch.manual_seed(0)
    model=ObservedMembershipV1('A').eval()
    xyz=torch.tensor([[[1.,0,0],[20.,0,0]]]);context=torch.zeros(1,2,128)
    valid=torch.tensor([[True,False]]);indices=torch.tensor([[0,1]])
    with torch.no_grad():
        a,mask=model._raw(xyz,context,valid,indices)
        changed=xyz.clone();changed[:,1]=100
        context[:,1]=123
        b,newmask=model._raw(changed,context,valid,indices)
        assert torch.equal(a,b) and torch.equal(mask,newmask)
        assert mask.sum()==1
        # Context attached to a retained return is still consumed. The real
        # upstream full-scan encoder can carry outside information here.
        context[:,0]=1
        c,_=model._raw(xyz,context,valid,indices)
        assert not torch.equal(a,c)


def test_no_local_return_produces_no_old_queries():
    torch.manual_seed(0)
    model=ObservedMembershipV1('A').eval()
    with torch.no_grad():
        result=model(torch.tensor([[[20.,0,0]]]),torch.zeros(1,1,128),
                     torch.tensor([[False]]),sensor_token_index=torch.tensor([[0]]))
    assert result is None
