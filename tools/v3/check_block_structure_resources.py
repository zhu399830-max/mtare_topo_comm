"""One maximum synthetic full-chain forward/backward, no optimizer/data I/O."""
import _bootstrap
import hashlib
import argparse
import json
import resource
import time
import numpy as np
import torch
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_context import pool_block_context
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.gse_block_structure_loss import LocatedStructureTargets,located_structure_loss


def state_sha(modules):
    h=hashlib.sha256()
    for model in modules:
        for key,value in model.state_dict().items():
            h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(smooth=False):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no fallback')
    torch.manual_seed(20260908);torch.cuda.manual_seed_all(20260908)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(20260908);n=57600
    blocks=bind_block_points(rng.uniform(-2,2,(n,3)).astype(np.float32),np.arange(n)//11520,np.arange(n)%4096)
    compact=rng.normal(size=(900,128)).astype(np.float32)
    context=pool_block_context(blocks,np.arange(n),compact,np.ones(n,bool))['context']
    frozen=torch.tensor(context,device='cuda',requires_grad=True)
    if smooth:
        from mtare_topo.representation.gse_block_structure_readout_v2 import BlockStructureReadoutV2
        head_class=BlockStructureReadoutV2
    else:head_class=BlockStructureReadout
    point=BlockPointEncoder().cuda();head=head_class().cuda()
    target=LocatedStructureTargets(torch.zeros((1,3),device='cuda'),torch.tensor([[10.,0,0]],device='cuda'),
        torch.tensor([[1.,0,0]],device='cuda'),torch.ones(1,dtype=torch.bool,device='cuda'),True,True)
    before=state_sha((point,head));torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    prediction=head(point(blocks,chunk_size=2048),frozen)
    loss=located_structure_loss(prediction,target);loss['total'].backward();torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    if any(p.grad is None or not torch.isfinite(p.grad).all() for model in (point,head) for p in model.parameters()):
        raise ValueError('missing/nonfinite gradient')
    if frozen.grad is not None:raise ValueError('frozen context gradient leak')
    if state_sha((point,head))!=before:raise ValueError('state mutated without optimizer')
    peak=torch.cuda.max_memory_allocated();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if peak>28*1024**3 or rss>32*1024**3:raise MemoryError('resource cap exceeded')
    print(json.dumps(dict(coordinate_version=2 if smooth else 1,gpu=torch.cuda.get_device_name(0),points=n,blocks=4096,
        anchor_queries=32,opening_queries=64,parameters=sum(p.numel() for m in (point,head) for p in m.parameters()),
        forward_backward_s=elapsed,peak_gpu_allocated_bytes=peak,peak_rss_bytes=rss,
        parameter_state_unchanged=True,frozen_context_grad=False,optimizer_steps=0,research_observations=0,
        loss_denominators=loss['counts'],scientific_gate_pass=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--smooth',action='store_true')
    main(parser.parse_args().smooth)
