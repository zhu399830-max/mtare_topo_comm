"""Single synthetic maximum-shape CUDA backward check, zero optimizer steps."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
import time
import numpy as np
import torch
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.block_relation_training_v1 import build_relation_model,state_sha256


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU substitution')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    rng=np.random.default_rng(0)
    points=rng.uniform(-4,4,(57600,3)).astype(np.float32)
    blocks=bind_block_points(points,np.repeat(np.arange(5),11520),np.arange(57600)%54)
    rows=[]
    for enabled in (False,True):
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        model=build_relation_model(relation_attributes=enabled,device='cuda')
        initial=state_sha256(model)
        features=model['point'](blocks)
        result=model['head'](features,torch.zeros(54,128,device='cuda'))
        loss=result.position_m.square().mean()+result.presence_logits.square().mean()+result.branch_logits.square().mean()+result.directions.square().mean()
        loss.backward();torch.cuda.synchronize()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite gradient')
        if torch.cuda.max_memory_reserved()>28*1024**3:raise MemoryError('GPU bound exceeded')
        rows.append(dict(relation_attributes=enabled,initial_sha256=initial,
            parameters=sum(p.numel() for p in model.parameters()),
            elapsed_s=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del model,features,result,loss;torch.cuda.empty_cache()
    if rows[0]['initial_sha256']!=rows[1]['initial_sha256']:raise ValueError('paired initialization mismatch')
    print(json.dumps(dict(status='SYNTHETIC_CUDA_RESOURCE_PASS_NOT_TRAINING',points=57600,blocks=54,
        optimizer_steps=0,device=torch.cuda.get_device_name(0),results=rows),indent=2))


if __name__=='__main__':main()
