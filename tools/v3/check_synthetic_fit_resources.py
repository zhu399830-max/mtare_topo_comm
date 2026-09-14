"""One fixed archived fixture, frozen encoder, A/B/C backward; zero updates/writes."""
import _bootstrap
from dataclasses import fields,replace
import json
from pathlib import Path
import resource
import time
import torch
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
from mtare_topo.data.gse_synthetic_fit_example import build_example
from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationModelV1
from mtare_topo.representation.gse_surface_losses_v1 import surface_relation_losses
from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256


def move(value): return replace(value,**{f.name:getattr(value,f.name).to('cuda:0') for f in fields(value)})


def main():
    root=Path(__file__).resolve().parents[2];scope=compile_scope(root)
    row=next(r for r in scope['observations'] if r['case_id']=='four_way__circle__view2')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    ck=scope['checkpoint'];bound=load_surface_encoder((root/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
    adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).to('cuda:0')
    started=time.monotonic()
    example=build_example((root/row['input_path']).read_bytes(),(root/row['source_record_path']).read_bytes(),row,
        encoder_adapter=adapter,encoder_state_sha256=bound.state_sha256,device='cuda:0')
    torch.cuda.synchronize()
    print(json.dumps(dict(stage='assembled',case_id=row['case_id'],patches=int(example.patches.valid.sum()),
        elapsed_s=time.monotonic()-started)),flush=True)
    for branch in ('A','B','C'):
        torch.manual_seed(0);model=SurfaceRelationModelV1(branch).to('cuda:0');model.train()
        before=module_state_sha256(model);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        prediction=model.forward_compact(move(example.compact),move(example.patches))
        loss=surface_relation_losses(prediction,move(example.targets));loss.total.backward();torch.cuda.synchronize()
        if not torch.isfinite(loss.total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite backward')
        if module_state_sha256(model)!=before or module_state_sha256(bound.backbone)!=bound.state_sha256:
            raise ValueError('weights changed without optimizer')
        print(json.dumps(dict(stage='backward',branch=branch,elapsed_s=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            diagnostic_loss=float(loss.total.detach()),denominators=loss.denominators,optimizer_steps=0)),flush=True)
        del model,prediction,loss
    print(json.dumps(dict(status='RESOURCE_CHECK_COMPLETE',optimizer_steps=0,training_pass=False)),flush=True)


if __name__=='__main__':main()
