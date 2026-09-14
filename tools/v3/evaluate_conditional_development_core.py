"""Streaming evaluation core; deliberately no CLI, optimizer or selection.

Caller must bind inputs/checkpoints and preflight a fresh evaluation run after
the complete reference export seals. This file alone does not authorize reads.
"""
import json
import time
import numpy as np
import torch
from ai_junction_pilot import sha,write
from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,bind_structural_patches
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets
from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256


def evaluate(root,run,entries,models,*,check_limits):
    """All observations, all outputs. Targets affect only masks and scoring."""
    loaded={};hashes={};rows=[];start=time.monotonic()
    for variant in 'ABC':
        record=models[variant]
        if sha(root/record['path'])!=record['sha256']:raise ValueError('frozen final model drift')
        model=GeometryStructureEncoder(variant)
        checkpoint=torch.load(root/record['path'],map_location='cpu',weights_only=True)
        if checkpoint['step']!=2000 or checkpoint['variant']!=variant:raise ValueError('only preregistered final model')
        model.load_state_dict(checkpoint['model'],strict=True)
        model.cuda().eval();model.requires_grad_(False)
        hashes[variant]=module_state_sha256(model);loaded[variant]=model
    with (run/'logs/observations.jsonl').open('x') as log,torch.inference_mode():
        for e in entries:
            identity=e['identity'];case=identity['case']
            with np.load(root/e['feature'],allow_pickle=False) as f:
                patches=SurfacePatches(**{k:f['patch_'+k] for k in SurfacePatches.__dataclass_fields__ if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
                p=bind_structural_patches([patches],device='cuda');x=torch.tensor(f['patch_observed_features'],device='cuda')
                roi=f['surface_return_indices']
            frames=tuple(identity['frame_rows']);ctx=(CausalFrameOrderContext('sensor_current',frames,frames[-1],identity['task']),)
            # Forward arguments contain only observation-derived data. Run all
            # queries before even opening this observation's reference payload.
            predictions={variant:model(x,p,ctx).relations for variant,model in loaded.items()}
            with np.load(root/e['target'],allow_pickle=False) as y:
                if not np.array_equal(roi,y['original_roi_indices']):raise ValueError('target/observation ROI drift')
                if not np.array_equal(p.batch.neighbor_index[0].cpu().numpy(),y['neighbor_index']):raise ValueError('target/query index drift')
                t=compile_conditional_loss_targets(y,'cuda')
            for variant,r in predictions.items():
                if r.computation_valid.shape!=t.known.shape or bool((t.known&~r.computation_valid).any()):
                    raise ValueError('known target outside computed query population')
                if not torch.equal(r.computation_valid,predictions['A'].computation_valid):
                    raise ValueError('A/B/C computation masks differ')
                if not bool(torch.isfinite(r.axis_abs_dot[r.computation_valid]).all()&torch.isfinite(r.height_difference_m[r.computation_valid]).all()):
                    raise ValueError('nonfinite prediction, including unknown reference regions')
                np.savez_compressed(run/f'artifacts/{variant}_{case:03d}.npz',axis=r.axis_abs_dot.cpu().numpy(),
                    height=r.height_difference_m.cpu().numpy(),valid=r.computation_valid.cpu().numpy(),neighbors=r.neighbor_index.cpu().numpy())
                metrics={}
                for category,mask in [('all',t.known),('same',t.known&t.same),('cross',t.known&~t.same)]:
                    count=int(mask.sum());w=t.weights[mask];w=w/w.sum() if count else w
                    metrics[category]=dict(count=count,
                        axis_mae=float((w*(r.axis_abs_dot[mask]-t.axis[mask]).abs()).sum()) if count else None,
                        height_mae_m=float((w*(r.height_difference_m[mask]-t.height[mask]).abs()).sum()) if count else None,
                        constant_axis_mae=float((w*(1-t.axis[mask]).abs()).sum()) if count else None,
                        constant_height_mae_m=float((w*t.height[mask].abs()).sum()) if count else None)
                row=dict(case=case,parent=identity['parent_id'],task=identity['task'],physical_edge=identity['physical_edge_id'],
                    variant=variant,metrics=metrics,unknown_relations=int((r.computation_valid&~t.known).sum()),
                    output_relations=int(r.computation_valid.sum()))
                rows.append(row);log.write(json.dumps(row)+'\n')
            log.flush();check_limits()
            print(json.dumps(dict(completed=case+1,elapsed_s=time.monotonic()-start)),flush=True)
            del predictions,p,x,t
    for variant,model in loaded.items():
        if module_state_sha256(model)!=hashes[variant]:raise ValueError('evaluation mutated model')
    write(run/'metrics/all_observations.json',rows)
    return rows
