from types import SimpleNamespace
import numpy as np
import torch
import pytest
from test_observed_anchor_detector_v1 import observation
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.frozen_candidate_scoring_v1 import (
    freeze_except_existence,forward,assert_frozen_parameters,verify_full_path)


def fixture():
    b=observation()
    class Student:
        source={'test':'synthetic'}
        student_representations={'PRIMITIVE':dict(blocks=b,context=np.zeros((len(b.block_ids),128),np.float32))}
        def __getattr__(self,name):raise AssertionError('forward cannot read teacher '+name)
    return freeze_except_existence(build_model()),Student()


def test_original_head_only_and_all_queries_unchanged_after_updates():
    m,o=fixture();m.eval();start=forward(m,o,'PRIMITIVE')
    trainable=[n for n,p in m.named_parameters() if p.requires_grad]
    assert trainable==['head.head.anchor.weight','head.head.anchor.bias']
    optimizer=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.001,weight_decay=0.)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True);out=forward(m,o,'PRIMITIVE')
        torch.nn.functional.softplus(-out.prediction.presence_logits[0]).backward()
        assert m['head'].head.anchor.weight.grad[:3].abs().sum()==0
        optimizer.step();assert_frozen_parameters(m)
        assert torch.equal(out.prediction.position_m,start.prediction.position_m)
    end=verify_full_path(m,o)
    assert not torch.equal(end.prediction.presence_logits,start.prediction.presence_logits)
    assert len(m._candidate_cache)==1


def test_accidental_position_or_shared_update_is_rejected():
    m,o=fixture();forward(m,o,'PRIMITIVE')
    with torch.no_grad():m['head'].head.anchor.weight[0,0]+=1
    with pytest.raises(ValueError):assert_frozen_parameters(m)
    with pytest.raises(ValueError):forward(m,o,'PRIMITIVE')


def test_query_feature_cache_detached_and_no_teacher_access():
    m,o=fixture();forward(m,o,'PRIMITIVE')
    assert all(not c['feature'].requires_grad for c in m._candidate_cache.values())
    assert all(not p.requires_grad for n,p in m.named_parameters() if n not in ('head.head.anchor.weight','head.head.anchor.bias'))


def test_fixed_supervision_cached_once_scores_do_not_rematch_unknown_no_gradient(monkeypatch):
    import mtare_topo.representation.frozen_candidate_scoring_v1 as module
    calls=[]
    def repair(output,o):
        calls.append(1)
        return dict(assignment=torch.tensor([0]),negative_mask=torch.tensor([False,True,False]),
            position=torch.tensor(0.),positive_count=1,negative_count=1,unknown_count=1)
    monkeypatch.setattr(module,'repaired_geometry_objective',repair)
    model=SimpleNamespace(_score_targets={});o=SimpleNamespace(source={'test':'fixed'})
    for values in ([1.,-1.,2.],[-10.,10.,-9.]):
        scores=torch.tensor(values,requires_grad=True);out=SimpleNamespace(prediction=SimpleNamespace(presence_logits=scores))
        r=module.center_objective(out,o,model=model);r['total'].backward()
        assert r['assignment'].tolist()==[0] and scores.grad[0]<0 and scores.grad[1]>0 and scores.grad[2]==0
    assert len(calls)==1
