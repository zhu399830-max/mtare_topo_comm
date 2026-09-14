import torch
from mtare_topo.representation.block_relation_training_v1 import build_relation_model,state_sha256


def test_exact_seeded_control_initialization_and_original_trainable_policy():
    a=build_relation_model(relation_attributes=False)
    b=build_relation_model(relation_attributes=True)
    assert state_sha256(a)==state_sha256(b)
    assert all(p.requires_grad for p in a.parameters())
    assert a['head'].relation_attributes is False
    assert b['head'].relation_attributes is True
    assert set(a.state_dict())==set(b.state_dict())
    for key in a.state_dict():assert torch.equal(a.state_dict()[key],b.state_dict()[key])
