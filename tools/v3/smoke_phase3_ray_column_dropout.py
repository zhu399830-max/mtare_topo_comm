#!/usr/bin/env python3
"""Eight-frame no-update smoke for the Phase-3 ray-column dropout corrective."""

from __future__ import annotations

import hashlib,json
from pathlib import Path

import numpy as np
import torch

from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet,multitask_loss
from mtare_topo.representation.ray_column_dropout import apply_ray_column_dropout

DATASET=Path("results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0")


def state_digest(model:torch.nn.Module)->str:
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main()->int:
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.use_deterministic_algorithms(True,warn_only=True);dataset=CanoV2RMultitaskDataset(DATASET,"train");samples=[dataset[index] for index in range(8)]
    student=torch.from_numpy(np.stack([item["student"] for item in samples]));student,augmentation=apply_ray_column_dropout(student,torch.Generator().manual_seed(100000),probability=.5,period=10)
    direction=torch.from_numpy(np.stack([item["direction_target"] for item in samples])).cuda();count=torch.tensor([item["count_target"] for item in samples],dtype=torch.long,device="cuda");role=torch.tensor([item["role_target"] for item in samples],dtype=torch.long,device="cuda")
    model=StructuralSemanticNet().cuda();before=state_digest(model);outputs=model(student.cuda());losses=multitask_loss(outputs,direction,count,role);losses["total"].backward();after=state_digest(model)
    gradients=[parameter.grad for parameter in model.parameters() if parameter.grad is not None];passed=before==after and gradients and all(torch.isfinite(value).all() for value in gradients) and all(torch.isfinite(value) for value in losses.values())
    result={"overall_status":"PASS_PHASE3_RAY_COLUMN_DROPOUT_NO_UPDATE_SMOKE" if passed else "FAIL_PHASE3_RAY_COLUMN_DROPOUT_NO_UPDATE_SMOKE","real_train_frames":8,"augmentation":augmentation,"finite_forward_loss_backward":passed,"weights_unchanged":before==after,"optimizer_steps":0,"checkpoints_created":0,"strict_test_frames_read":0,"mtare_frames_read":0}
    print(json.dumps(result));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
