"""Single fixed cache CUDA autograd check; no targets or optimizer updates."""
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from mtare_topo.data.development_compact_blocks import read_compact_points,sensor_layout_partition,bind_partition
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.block_anchor_branch import BlockAnchorBranch


def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check(root):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    run=root/'results/gate3_semantics/gate3_20260908_gse_supplement_features_v1_seed0'
    raw=(run/'artifacts/feature_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='9e9d12c056c8ad138a66710da93e58554b430af59f717a55eabfe11b6cd051bc':
        raise ValueError('manifest drift')
    source=dict(task='S01_flat_tree_small_C01__c1_mixed',source_sequence_id=598,frame_rows=[806,807,808,809,810])
    rows=[r for r in json.loads(raw)['observations'] if all(r['source'][k]==v for k,v in source.items())]
    if len(rows)!=1:raise ValueError('fixed identity missing')
    row=rows[0];payload=(run/row['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!='ee7a80456b1e60d71ba822058fd4e1e2e2ba1f7bb2d6ca39a8b679c24e0d8956':
        raise ValueError('input drift')
    compact=read_compact_points(payload,manifest_row=row,expected_source=source,
        encoder_sha256='200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb')
    patches=extract_surface_patches(compact.points_xyz_m,np.ones(len(compact.points_xyz_m),bool),compact.source_flat_ray_index//11520)
    examples={'r0':sensor_layout_partition(compact),'r1':bind_partition(compact,patches.point_patch_index)}
    results={};initial=None
    for name,example in examples.items():
        torch.manual_seed(0)
        model=torch.nn.ModuleDict(dict(point=BlockPointEncoder(),head=BlockAnchorBranch(BlockStructureReadout(),branch_slots=64))).cuda()
        before=state_hash(model)
        if initial is not None and before!=initial:raise ValueError('initial states differ')
        initial=before;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        context=torch.tensor(example['context'],device='cuda')
        output=model['head'](model['point'](example['blocks']),context)
        # This scalar tests graph connectivity, not a teacher or research loss.
        scalar=(output.position_m.square().mean()+output.presence_logits.mean()
                +output.directions[...,0].mean()+output.branch_logits.mean())
        scalar.backward();torch.cuda.synchronize()
        bad=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        if bad:raise ValueError('missing/nonfinite gradients: '+str(bad))
        if state_hash(model)!=before:raise ValueError('weights changed without optimizer')
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        if reserved>28*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>32*1024**3:
            raise MemoryError('resource contract exceeded')
        results[name]=dict(elapsed_s=time.monotonic()-start,peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,parameter_count=sum(p.numel() for p in model.parameters()),
            finite_gradients=True,weights_unchanged=True)
        del model,output,scalar,context
        torch.cuda.empty_cache()
    return dict(status='FIXED_INPUT_AUTOGRAD_CHECK_NOT_RESEARCH_TRAINING',source=source,
        device=torch.cuda.get_device_name(0),probe_branch_slots=64,formal_branch_slots_frozen=False,
        results=results,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_steps=0,teacher_reads=0,checkpoint_saved=False,initial_state_sha256=initial)


if __name__=='__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[2]),indent=2))
