#!/usr/bin/env python3
"""One read-only real batch forward/loss/backward smoke; never optimizer.step."""

from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet,multitask_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);a=p.parse_args();started=time.monotonic()
    torch.use_deterministic_algorithms(True)
    dataset=CanoV2RMultitaskDataset(a.dataset_run,'train');samples=[dataset[i] for i in range(a.batch_size)];student=torch.from_numpy(np.stack([x['student'] for x in samples])).cuda();direction=torch.from_numpy(np.stack([x['direction_target'] for x in samples])).cuda();count=torch.tensor([x['count_target'] for x in samples],device='cuda');role=torch.tensor([x['role_target'] for x in samples],device='cuda')
    torch.manual_seed(0);model=StructuralSemanticNet().cuda();before={name:value.detach().clone() for name,value in model.state_dict().items()};outputs=model(student);losses=multitask_loss(outputs,direction,count,role);losses['total'].backward();unchanged=all(torch.equal(before[name],value) for name,value in model.state_dict().items());determinism_env=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    passed=unchanged and determinism_env==':4096:8' and all(torch.isfinite(v).all() for v in outputs.values())
    result={'schema_version':'phase3_real_batch_smoke_v2','status':'PASS_PHASE3_REAL_BATCH_DETERMINISTIC_NO_UPDATE_SMOKE' if passed else 'FAIL_PHASE3_REAL_BATCH_DETERMINISTIC_NO_UPDATE_SMOKE','dataset':str(a.dataset_run.resolve().relative_to(PROJECT_ROOT)),'split':'train','batch_size':a.batch_size,'frame_ids':[x['frame_id'] for x in samples],'student_shape':list(student.shape),'target_shapes':{'direction':list(direction.shape),'count':list(count.shape),'role':list(role.shape)},'output_shapes':{k:list(v.shape) for k,v in outputs.items()},'losses':{k:float(v.detach().cpu()) for k,v in losses.items()},'parameters':sum(p.numel() for p in model.parameters()),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'cublas_workspace_config':determinism_env,'optimizer_constructed':False,'optimizer_step':False,'weights_unchanged':unchanged,'training_samples_consumed':0,'models':0,'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'duration_seconds':time.monotonic()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['status'].startswith('PASS') else 2
if __name__=='__main__':raise SystemExit(main())
