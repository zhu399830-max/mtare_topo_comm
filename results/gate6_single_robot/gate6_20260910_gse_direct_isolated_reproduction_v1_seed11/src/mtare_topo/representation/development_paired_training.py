"""Fixed paired microbatch training mechanics; no data access or run authority."""
from dataclasses import replace
import hashlib
import numpy as np
import torch
from .gse_block_point_encoder import BlockPointEncoder
from .gse_block_structure_readout import BlockStructureReadout
from .block_anchor_branch import BlockAnchorBranch
from .conditional_anchor_branch_loss import conditional_anchor_branch_loss


def build_model(*, seed=0, device='cuda'):
    torch.manual_seed(seed)
    return torch.nn.ModuleDict(dict(point=BlockPointEncoder(),
        head=BlockAnchorBranch(BlockStructureReadout(),branch_slots=64))).to(device)


def state_sha256(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_schedule(fit_count, *, updates=2000, seed=0):
    if type(fit_count) is not int or fit_count<1 or type(updates) is not int or updates<1:
        raise ValueError('positive fixed population/update counts required')
    rng=np.random.default_rng(seed);order=[]
    while len(order)<updates*4:order.extend(rng.permutation(fit_count).tolist())
    return tuple(tuple(order[i:i+4]) for i in range(0,updates*4,4))


def forward_observation(model, observation, representation):
    if representation not in ('r0','r1','r2'):raise ValueError('unknown representation')
    student=observation.student_representations[representation]
    parameter=next(model.parameters())
    context=torch.tensor(student['context'],dtype=parameter.dtype,device=parameter.device)
    # Nothing from observation.loss_only, source, or split enters the model.
    result=model['head'](model['point'](student['blocks']),context)
    if result is None:raise ValueError('empty observation; no guessed predictions')
    return result


def objective(prediction, observation):
    loss=observation.loss_only;target=loss['target'];device=prediction.position_m.device
    target=replace(target,position_m=target.position_m.to(device=device,dtype=prediction.position_m.dtype),
                   directions=tuple(d.to(device=device,dtype=prediction.position_m.dtype) for d in target.directions))
    return conditional_anchor_branch_loss(prediction,target,bundle=loss['bundle'],grid=loss['grid'],
        produced_targets=loss['produced_targets'],frozen_manifest=loss['frozen_manifest'])


def train_update(model, optimizer, batch, representation):
    if len(batch)!=4 or any(o.split!='fit' for o in batch):
        raise ValueError('four fit-only observations required before any update')
    model.train();optimizer.zero_grad(set_to_none=True);records=[];active=0
    try:
        for observation in batch:
            prediction=forward_observation(model,observation,representation)
            result=objective(prediction,observation)
            if not torch.isfinite(result['total']):raise FloatingPointError('nonfinite loss')
            if result['has_supervision']:
                (result['total']/4.).backward();active+=1
            records.append(dict(source=observation.source,loss=float(result['total'].detach()),
                terms={k:float(v.detach()) for k,v in result['terms'].items()},counts=result['counts']))
            del prediction,result
        # Missing tasks never trigger an AdamW weight-decay-only update.
        if not active:return dict(optimizer_step=False,observations=records)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient; no optimizer step')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite updated parameters; stop run')
        return dict(optimizer_step=True,observations=records)
    finally:
        optimizer.zero_grad(set_to_none=True)
