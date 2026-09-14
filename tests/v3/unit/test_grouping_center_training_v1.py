from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import torch
from test_conditional_anchor_branch_loss import fixture
from test_observed_anchor_detector_v1 import observation
from mtare_topo.representation.grouping_center_training_v1 import build_model,forward,center_objective


def test_forward_never_reads_teacher_same_queries():
    b=observation()
    class Student:
        student_representations={k:dict(blocks=b,context=np.zeros((len(b.block_ids),128),np.float32)) for k in ('SPATIAL','PRIMITIVE')}
        def __getattr__(self,key):raise AssertionError(key)
    model=build_model();a=forward(model,Student(),'SPATIAL');b=forward(model,Student(),'PRIMITIVE')
    assert torch.equal(a.query_positions_m,b.query_positions_m)
    assert torch.equal(a.prediction.position_m,b.prediction.position_m)


def test_center_only_no_branch_gradient_or_branch_target_access():
    p,t,k=fixture()
    target=replace(t,directions=(torch.tensor([[1.,0,0]],dtype=torch.float64),),branches_complete=(True,))
    obs=SimpleNamespace(loss_only=dict(**k,target=target))
    result=center_objective(SimpleNamespace(prediction=p),obs)
    result['total'].backward()
    assert p.branch_logits.grad is None and p.directions.grad is None
    assert p.presence_logits.grad[2]==0
    assert result['assignment'].tolist()==[1]
