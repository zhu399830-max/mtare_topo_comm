from types import SimpleNamespace
import pytest
import torch
import mtare_topo.representation.development_corrective_training_v1 as m


def fixture(monkeypatch, active=True, bad=False):
    model=torch.nn.Linear(1,1,bias=False)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    monkeypatch.setattr(m,'forward_observation',lambda model,*args:model.weight)
    def objective(prediction,obs):
        loss=prediction.square().sum() if active else torch.zeros((),requires_grad=True)
        if bad:loss=loss*float('nan')
        return dict(total=loss,has_supervision=active,terms={'x':loss},counts={'x':int(active)},
                    anchor_assignment=torch.tensor([0]),branch_assignments=(torch.tensor([1]),),
                    branch_group_counts=[dict(positive=1,negative=2)])
    monkeypatch.setattr(m,'objective',objective)
    return model,opt,[SimpleNamespace(split='fit',source={'i':i}) for i in range(4)]


def test_one_step_four_observations_and_assignment_log(monkeypatch):
    model,opt,batch=fixture(monkeypatch);before=model.weight.detach().clone()
    result=m.train_update(model,opt,batch,'r0')
    assert result['optimizer_step'] and len(result['observations'])==4
    assert not torch.equal(model.weight,before) and int(opt.state[model.weight]['step'])==1
    assert result['observations'][0]['branch_assignments']==[[1]]
    assert model.weight.grad is None


@pytest.mark.parametrize('fault',['unknown','nan','development'])
def test_no_update(monkeypatch,fault):
    model,opt,batch=fixture(monkeypatch,active=fault!='unknown',bad=fault=='nan')
    before=model.weight.detach().clone()
    if fault=='development':batch[1].split='development'
    if fault=='unknown':assert not m.train_update(model,opt,batch,'r1')['optimizer_step']
    else:
        with pytest.raises((ValueError,FloatingPointError)):m.train_update(model,opt,batch,'r1')
    assert torch.equal(model.weight,before) and not opt.state


def test_gradient_nonfinite_stops_before_optimizer(monkeypatch):
    model,opt,batch=fixture(monkeypatch);before=model.weight.detach().clone()
    handle=model.weight.register_hook(lambda grad:grad*float('inf'))
    with pytest.raises(FloatingPointError,match='gradient'):m.train_update(model,opt,batch,'r2')
    handle.remove()
    assert torch.equal(model.weight,before) and not opt.state and model.weight.grad is None
