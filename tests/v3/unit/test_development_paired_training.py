from types import SimpleNamespace
import pytest
import torch
import mtare_topo.representation.development_paired_training as m


def test_shared_fixed_batch_schedule():
    a=m.batch_schedule(250);assert a==m.batch_schedule(250)
    assert len(a)==2000 and all(len(b)==4 and all(0<=i<250 for i in b) for b in a)


def fixture(monkeypatch,active=True,bad=False):
    model=torch.nn.Linear(1,1,bias=False);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    monkeypatch.setattr(m,'forward_observation',lambda model,*a:model.weight)
    def loss(prediction,obs):
        total=prediction.square().sum() if active else torch.zeros((),requires_grad=True)
        if bad:total=total*float('nan')
        return dict(total=total,has_supervision=active,terms={'x':total},counts={'x':int(active)})
    monkeypatch.setattr(m,'objective',loss)
    batch=[SimpleNamespace(split='fit',source={'index':i}) for i in range(4)]
    return model,optimizer,batch


def test_one_optimizer_step_for_four_microbatches(monkeypatch):
    model,opt,batch=fixture(monkeypatch);before=m.state_sha256(model)
    result=m.train_update(model,opt,batch,'r0')
    assert result['optimizer_step'] and len(result['observations'])==4
    assert m.state_sha256(model)!=before and int(opt.state[model.weight]['step'])==1


@pytest.mark.parametrize('fault',['unknown','nan','development'])
def test_no_update_on_unknown_nonfinite_or_nonfit(monkeypatch,fault):
    model,opt,batch=fixture(monkeypatch,active=fault!='unknown',bad=fault=='nan');before=m.state_sha256(model)
    if fault=='development':batch[2].split='development'
    if fault=='unknown':assert not m.train_update(model,opt,batch,'r1')['optimizer_step']
    else:
        with pytest.raises((ValueError,FloatingPointError)):m.train_update(model,opt,batch,'r1')
    assert m.state_sha256(model)==before and not opt.state
