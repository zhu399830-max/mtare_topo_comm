from dataclasses import replace,fields
import pytest
import torch
from mtare_topo.representation.gse_surface_relation_control_v1 import SurfaceRelationControlV1
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationModelV1
from tests.v3.unit.test_gse_surface_relation_model_v1 import inputs,assert_equal,threads


def pair():
    torch.manual_seed(11)
    full=SurfaceRelationControlV1(True);removed=SurfaceRelationControlV1(False)
    removed.load_state_dict(full.state_dict())
    return full,removed


def test_full_is_bitwise_legacy_and_zero_relation_matches_control():
    x,c,v,p=inputs();full,removed=pair();old=SurfaceRelationModelV1('C')
    old.load_state_dict(full.state_dict())
    assert_equal(full(x,c,v,p),old(x,c,v,p))
    assert_equal(removed(x,c,v,p),full(x,c,v,replace(p,relation=torch.zeros_like(p.relation))))
    assert list(full.state_dict())==list(removed.state_dict())
    assert sum(t.numel() for t in full.parameters())==sum(t.numel() for t in removed.parameters())


def test_relation_control_retains_neighbors_and_ignores_attributes():
    x,c,v,p=inputs();full,removed=pair()
    altered=replace(p,relation=p.relation+.5)
    assert_equal(removed(x,c,v,p),removed(x,c,v,altered))
    assert not torch.allclose(full(x,c,v,p).membership_logits,full(x,c,v,altered).membership_logits)
    no_neighbors=replace(p,neighbor_index=torch.full_like(p.neighbor_index,-1),neighbor_valid=torch.zeros_like(p.neighbor_valid))
    assert not torch.allclose(removed(x,c,v,p).membership_logits,removed(x,c,v,no_neighbors).membership_logits)


def test_gradients_use_same_message_layers_but_not_removed_columns():
    x,c,v,p=inputs();full,removed=pair()
    for model in (full,removed):
        out=model(x,c.requires_grad_(),v,p)
        loss=sum(getattr(out,f.name).square().mean() for f in fields(out) if getattr(out,f.name).dtype!=torch.bool)
        loss.backward()
        assert c.grad is None
        for layer in model.patch_layers:
            grad=layer.message[0].weight.grad
            assert torch.isfinite(grad).all() and grad[:,:-9].abs().sum()>0
            if model.relation_attributes:assert grad[:,-9:].abs().sum()>0
            else:assert torch.count_nonzero(grad[:,-9:])==0


def test_removed_attributes_do_not_hide_nonfinite_inputs():
    x,c,v,p=inputs();_,removed=pair()
    with pytest.raises(ValueError,match='nonfinite'):
        removed(x,c,v,replace(p,relation=torch.full_like(p.relation,float('nan'))))


def test_existing_loss_unknown_membership_reaches_neither_membership_head():
    from tests.v3.unit.test_gse_surface_losses_v1 import targets
    from mtare_topo.representation.gse_surface_losses_v1 import surface_relation_losses
    x,c,v,p=inputs();t=targets()
    t=replace(t,membership_valid=torch.zeros_like(t.membership_valid),
              membership=torch.full_like(t.membership,float('nan')))
    for model in pair():
        model=model.double()
        patch=replace(p,unary=p.unary.double(),relation=p.relation.double())
        result=surface_relation_losses(model(x.double(),c.double(),v,patch),t)
        assert result.denominators['membership']==0
        result.total.backward()
        assert model.member_anchor.weight.grad is None
        assert model.member_opening.weight.grad is None
        assert model.raw_adapter[0].weight.grad.abs().sum()>0
        assert model.patch_layers[0].message[0].weight.grad.abs().sum()>0
