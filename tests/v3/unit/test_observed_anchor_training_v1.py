from types import SimpleNamespace
import numpy as np
import pytest
import torch
import mtare_topo.representation.observed_anchor_training_v1 as m
from test_observed_anchor_detector_v1 import observation


def test_forward_never_reads_teacher_or_identity():
    blocks=observation()
    class OnlyStudent:
        student_representations={'r2':dict(blocks=blocks,context=np.zeros((3,128),np.float32))}
        def __getattr__(self,name):raise AssertionError('forward accessed '+name)
    model=m.build_model();output=m.forward_observation(model,OnlyStudent())
    assert len(output.query_source_indices)==5


def fixture(monkeypatch,active=True,bad=False):
    model=torch.nn.Linear(1,1,bias=False);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    monkeypatch.setattr(m,'forward_observation',lambda model,*a:SimpleNamespace(
        prediction=model.weight,query_source_indices=torch.tensor([0])))
    def objective(output,obs):
        loss=output.prediction.square().sum() if active else torch.zeros((),requires_grad=True)
        if bad:loss=loss*float('nan')
        return dict(total=loss,has_supervision=active,terms={'x':loss},counts={'x':int(active)},
                    anchor_assignment=torch.tensor([0]),branch_assignments=(torch.tensor([0]),),
                    anchor_group_counts=dict(positive=1,negative=0,unknown=0),branch_group_counts=[])
    monkeypatch.setattr(m,'objective',objective)
    return model,opt,[SimpleNamespace(split='fit',source={'i':i}) for i in range(4)]


def test_four_samples_one_optimizer_step_and_source_log(monkeypatch):
    model,opt,batch=fixture(monkeypatch);r=m.train_update(model,opt,batch)
    assert r['optimizer_step'] and int(opt.state[model.weight]['step'])==1
    assert len(r['observations'])==4 and r['observations'][0]['query_source_indices']==[0]
    assert model.weight.grad is None


@pytest.mark.parametrize('fault',['unknown','nan','calibration','gradient'])
def test_invalid_or_unknown_batch_never_updates(monkeypatch,fault):
    model,opt,batch=fixture(monkeypatch,active=fault!='unknown',bad=fault=='nan')
    before=model.weight.detach().clone()
    if fault=='calibration':batch[1].split='calibration'
    if fault=='gradient':model.weight.register_hook(lambda grad:grad*float('inf'))
    if fault=='unknown':assert not m.train_update(model,opt,batch)['optimizer_step']
    else:
        with pytest.raises((ValueError,FloatingPointError)):m.train_update(model,opt,batch)
    assert torch.equal(before,model.weight) and not opt.state
