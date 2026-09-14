import torch
import pytest
from dataclasses import replace
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
from mtare_topo.representation.gse_observable_relation_losses import ObservableRelationTargets,observable_relation_losses
from test_structural_representation import inputs,context


def setup():
    x,p=inputs();r=GeometryStructureEncoder('C')(x,p,context()).relations
    mask=r.computation_valid.clone();mask.zero_();mask[0,0,0]=True
    value=torch.full_like(r.axis_abs_dot,float('nan'));value[mask]=0.
    section=value[...,None].expand(*value.shape,2).clone()
    t=ObservableRelationTargets(value,mask,value,mask,section,mask[...,None].expand_as(section),value,mask,('synthetic-observed-support',))
    return r,t


def test_unknown_nan_masked_and_known_gradients():
    r,t=setup();r.height_difference_m.retain_grad()
    losses,counts=observable_relation_losses(r,t)
    sum(losses.values()).backward()
    assert counts['axis']==1 and counts['section']==2 and counts['correspondence_negative']==1
    assert torch.isfinite(r.height_difference_m.grad).all()
    assert not r.height_difference_m.grad[~t.height_known].any()


def test_auxiliary_identity_schema_rejected():
    r,t=setup()
    with pytest.raises(ValueError,match='not channel supervision'):
        observable_relation_losses(r,replace(t,schema_version='source_affinity'))


def test_absent_evidence_or_invalid_pair_rejected():
    r,t=setup()
    with pytest.raises(ValueError,match='without provenance'):
        observable_relation_losses(r,replace(t,evidence_refs=()))
    m=t.axis_known.clone();m[0,0,7]=True
    with pytest.raises(ValueError,match='absent computation pair'):
        observable_relation_losses(r,replace(t,axis_known=m))


def test_all_unknown_no_learning_evidence():
    r,t=setup();m=torch.zeros_like(t.axis_known)
    t=replace(t,axis_known=m,height_known=m,section_known=m[...,None].expand_as(t.section_known),correspondence_known=m,evidence_refs=())
    loss,counts=observable_relation_losses(r,t)
    assert all(v==0 for v in counts.values()) and sum(loss.values()).item()==0
    sum(loss.values()).backward()
