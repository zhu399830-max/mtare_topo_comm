"""Original existence row only; observation-derived frozen queries, loss-only labels."""
from dataclasses import replace
import torch
from .grouping_center_training_v1 import forward as full_forward
from .geometry_match_presence_v1 import center_objective as repaired_geometry_objective,restore_presence
from mtare_topo.governance_surface_selection import digest


def freeze_except_existence(model):
    for p in model.parameters():p.requires_grad_(False)
    head=model['head'].head.anchor
    head.weight.requires_grad_(True);head.bias.requires_grad_(True)
    model._score_reference={k:v.detach().clone() for k,v in model.state_dict().items()}
    model._candidate_cache={};model._score_targets={}
    return model


def assert_frozen_parameters(model):
    for name,value in model.state_dict().items():
        old=model._score_reference[name]
        if name in ('head.head.anchor.weight','head.head.anchor.bias'):
            if not torch.equal(value[:3],old[:3]):raise ValueError('position head changed')
        elif not torch.equal(value,old):raise ValueError('shared parameter or buffer changed: '+name)


def forward(model,observation,grouping):
    if grouping!='PRIMITIVE':
        with torch.no_grad():return full_forward(model,observation,grouping)
    key=digest(observation.source)
    if key not in model._candidate_cache:
        captured={}
        def capture(_,args,out):captured.update(feature=args[0].detach().clone(),raw=out[:,:3].detach().clone())
        hook=model['head'].head.anchor.register_forward_hook(capture)
        try:
            with torch.no_grad():out=full_forward(model,observation,grouping)
        finally:hook.remove()
        model._candidate_cache[key]=dict(output=out,**captured)
    c=model._candidate_cache[key]
    raw=model['head'].head.anchor(c['feature'])
    # Only row3 receives loss gradients. wd0 ensures the other three rows stay exact.
    if not torch.equal(raw[:,:3].detach(),c['raw']):raise ValueError('cached location generation changed')
    return replace(c['output'],prediction=replace(c['output'].prediction,presence_logits=raw[:,3]))


def center_objective(output,observation,*,model):
    key=digest(observation.source)
    if key not in model._score_targets:
        result=repaired_geometry_objective(output,observation)
        # Store no computation graph and never feed labels to forward.
        model._score_targets[key]={k:(v.detach().clone() if isinstance(v,torch.Tensor) else v) for k,v in result.items()}
    result=restore_presence(model._score_targets[key],output.prediction.presence_logits)
    result['total']=result['presence']
    return result


@torch.no_grad()
def verify_full_path(model,observation):
    cached=forward(model,observation,'PRIMITIVE');actual=full_forward(model,observation,'PRIMITIVE')
    for a,b in [(cached.prediction.position_m,actual.prediction.position_m),
        (cached.prediction.presence_logits,actual.prediction.presence_logits),
        (cached.query_source_indices,actual.query_source_indices),
        (cached.query_positions_m,actual.query_positions_m)]:
        if not torch.equal(a,b):raise ValueError('cached and complete original forward disagree')
    return cached
