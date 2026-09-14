"""Fixed sealed-feature ABC corrective forward/backward, zero optimization."""
import _bootstrap
import json,resource,time
from pathlib import Path
import torch
from check_synthetic_fit_resources import move
from mtare_topo.data.gse_synthetic_fit_cache import compile_scope,load_examples
from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
from mtare_topo.representation.gse_window_surface_model_v1 import WindowSurfaceModelV1
from mtare_topo.representation.gse_window_set_loss_v1 import window_set_loss


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    root=Path(__file__).resolve().parents[2];s=compile_scope(root);ck=s['checkpoint']
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    bound=load_surface_encoder((root/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=2)
    examples=load_examples((root/s['cached_examples']['path']).read_bytes(),s,bound.state_sha256)
    example=next(e for e in examples if e.header['observation_id']=='four_way__circle__view2')
    for branch in ('A','B','C'):
        torch.manual_seed(0);model=WindowSurfaceModelV1(branch).to('cuda:0');before=module_state_sha256(model)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        p=model.forward_compact(move(example.compact),move(example.patches));loss=window_set_loss(p,move(example.targets))
        loss.total.backward();torch.cuda.synchronize()
        if not torch.isfinite(loss.total) or any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('nonfinite gradients')
        if module_state_sha256(model)!=before:raise ValueError('unexpected weight mutation')
        print(json.dumps(dict(branch=branch,elapsed_s=time.monotonic()-start,loss=float(loss.total.detach()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            max_surface_radius_error_m=float((torch.linalg.vector_norm(p.opening_position_m,dim=-1)-10).abs().max().detach()),
            denominators=loss.denominators,optimizer_steps=0)),flush=True)
        del model,p,loss


if __name__=='__main__':main()
