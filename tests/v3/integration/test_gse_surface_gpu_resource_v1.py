"""Opt-in single synthetic GPU allocation/backward check, optimizer steps=0.

Run with GSE_RUN_GPU_RESOURCE_V1=1 and an outer timeout120s in the pinned Torch
sidecar. This is not training, a data-dependent experiment or a scientific run.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pytest
import torch
import zarr

from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationModelV1, collate_surface_patches


@pytest.mark.skipif(os.environ.get("GSE_RUN_GPU_RESOURCE_V1") != "1", reason="explicit single GPU resource check only")
def test_c57600_points4096_patches_one_backward(record_property):
    started = time.monotonic()
    assert torch.__version__ == "2.9.0+cu129" and zarr.__version__ == "2.18.7", "pinned sidecar drift"
    assert torch.cuda.is_available(), "GPU unavailable: do not substitute CPU"
    assert "5090" in torch.cuda.get_device_name(0), "requested5090 not present"
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[3]
    sources = ["src/mtare_topo/representation/" + name for name in (
        "gse_surface_patches_v1.py", "gse_surface_relation_model_v1.py",
        "gse_dual_path_encoder_adapter_v1.py", "primitive_relation_model.py")]
    before = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    cap = 28 * 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1., cap / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    # Exactly4096 occupied0.5m cells, all inside10m, repeated across57600 rays.
    grid = np.array([(x,y,z) for x in np.arange(-3.875,4.,.5)
        for y in np.arange(-3.875,4.,.5) for z in np.arange(-3.875,4.,.5)])
    assert grid.shape == (4096,3)
    xyz = grid[np.arange(57600) % 4096].astype(np.float32)
    valid = np.ones(57600,dtype=bool); frame = np.repeat(np.arange(5),16*720)
    patches = extract_surface_patches(xyz,valid,frame)
    assert len(patches.centers_m) == 4096
    patch_batch = collate_surface_patches([patches],device="cuda:0")
    assert patch_batch.neighbor_index.shape == (1,4096,8)
    t = torch.arange(5,device="cuda:0")[:,None,None]
    az = torch.arange(720,device="cuda:0")[None,None]
    index = (t*180 + az//4).expand(5,16,720).reshape(1,-1)
    compact = CompactFrozenDualPathFeatures(torch.tensor(xyz,device="cuda:0")[None],
        torch.zeros(1,900,128,device="cuda:0"),torch.tensor(valid,device="cuda:0")[None],index)
    model = SurfaceRelationModelV1("C").cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    for branch in ("A","B"):
        other = SurfaceRelationModelV1(branch)
        assert {k:tuple(v.shape) for k,v in other.state_dict().items()} == {k:tuple(v.shape) for k,v in model.state_dict().items()}
        assert sum(p.numel() for p in other.parameters()) == parameter_count
        del other
    torch.cuda.synchronize()
    compute_started = time.monotonic()
    prediction = model.forward_compact(compact,patch_batch)
    assert prediction.anchor_position_m.shape == (1,32,3) and prediction.opening_position_m.shape == (1,64,3)
    loss = sum(value.square().mean() for value in vars(prediction).values() if value.dtype != torch.bool)
    assert torch.isfinite(loss)
    loss.backward(); torch.cuda.synchronize()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    allocated, reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
    host_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert allocated <= cap and reserved <= cap and host_bytes <= 32*1024**3
    after = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    assert before == after
    elapsed = time.monotonic()-started
    assert elapsed <= 120
    report = dict(status="SYNTHETIC_GPU_RESOURCE_CHECK_ONLY",device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,zarr_version=zarr.__version__,seed=0,parameters=parameter_count,
        microbatch=1,points=57600,patches=4096,neighbors=8,optimizer_steps=0,real_data_reads=0,
        checkpoint_reads=0,finite_gradients=True,cuda_peak_allocated_bytes=allocated,
        cuda_peak_reserved_bytes=reserved,host_peak_rss_bytes=host_bytes,elapsed_s=elapsed,
        forward_backward_s=time.monotonic()-compute_started,source_sha256=before)
    record_property("synthetic_gpu_resource",json.dumps(report,sort_keys=True))
    print("GSE_SYNTHETIC_GPU_RESOURCE="+json.dumps(report,sort_keys=True))
