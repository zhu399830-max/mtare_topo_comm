#!/usr/bin/env python3
"""No-update CPU/CUDA audit of the frozen v1r3 M1 stability contract."""

from __future__ import annotations

import argparse,json,time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.evaluation.phase3_stability_contract import compare_outputs,distribution_summary,fixed_ray_column_mask
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


DATASET=Path("results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0")
SOURCE_RUN=Path("results/gate2_representation/gate2_20260812_cano_phase3_multitask_structural_semantics_v1r3_seed0")
SEEDS=(0,1,2);FRAMES=2048;BATCH_SIZE=128;SHIFT_COLUMNS=12


def _write_json(path:Path,value:dict)->None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


@torch.no_grad()
def audit_seed(dataset:CanoV2RMultitaskDataset,seed:int,device:torch.device)->dict:
    checkpoint=torch.load(SOURCE_RUN/f"artifacts/models/m1_seed{seed}/best.pt",map_location=device,weights_only=False)
    if checkpoint.get("mode")!="M1" or int(checkpoint.get("seed",-1))!=seed:
        raise RuntimeError(f"checkpoint identity mismatch for seed {seed}")
    model=StructuralSemanticNet().to(device);model.load_state_dict(checkpoint["model"]);model.eval()
    arrays={key:[] for key in ("rotation_direction_absolute_logit_error","rotation_z_role_cosine","masking_z_role_cosine")}
    seen=0
    for start in range(0,FRAMES,BATCH_SIZE):
        samples=[dataset[index] for index in range(start,min(start+BATCH_SIZE,FRAMES))]
        student=torch.from_numpy(np.stack([item["student"] for item in samples])).to(device)
        base=model(student);rotated=model(torch.roll(student,SHIFT_COLUMNS,dims=-1));masked=model(fixed_ray_column_mask(student))
        batch=compare_outputs(base,rotated,masked,SHIFT_COLUMNS)
        for key,value in batch.items():arrays[key].append(value)
        seen+=len(samples)
    if seen!=FRAMES:raise RuntimeError(f"frame count mismatch: {seen}")
    values={key:np.concatenate(parts) for key,parts in arrays.items()}
    return {
        "seed":seed,"device":device.type,"checkpoint_epoch":int(checkpoint["epoch"]),"frames":seen,
        "rotation_shift_columns":SHIFT_COLUMNS,"rotation_shift_deg":6.0,
        "rotation_direction_absolute_logit_error":distribution_summary(values["rotation_direction_absolute_logit_error"]),
        "rotation_z_role_cosine":distribution_summary(values["rotation_z_role_cosine"]),
        "fixed_masking_z_role_cosine":distribution_summary(values["masking_z_role_cosine"]),
        "parameter_gradients_created":sum(parameter.grad is not None for parameter in model.parameters()),
    }


def plot_results(records:list[dict],path:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True);x=np.arange(3);width=.36
    cpu=[item for item in records if item["device"]=="cpu"];cuda=[item for item in records if item["device"]=="cuda"]
    axes[0].bar(x-width/2,[r["rotation_direction_absolute_logit_error"]["maximum"] for r in cpu],width,label="CPU")
    axes[0].bar(x+width/2,[r["rotation_direction_absolute_logit_error"]["maximum"] for r in cuda],width,label="CUDA")
    axes[0].axhline(2e-5,color="black",linestyle="--",label="frozen 2e-5")
    axes[0].set_yscale("log");axes[0].set_xticks(x,["seed 0","seed 1","seed 2"]);axes[0].set_title("6° rotation: maximum logit error");axes[0].legend()
    axes[1].bar(x-width/2,[r["fixed_masking_z_role_cosine"]["mean"] for r in cpu],width,label="CPU")
    axes[1].bar(x+width/2,[r["fixed_masking_z_role_cosine"]["mean"] for r in cuda],width,label="CUDA")
    axes[1].axhline(.85,color="black",linestyle="--",label="frozen 0.85")
    axes[1].set_xticks(x,["seed 0","seed 1","seed 2"]);axes[1].set_ylim(-.1,1.05);axes[1].set_title("Fixed 10% ray-column masking cosine");axes[1].legend()
    fig.suptitle("Gate 2 v1r3 stability contract — first 2,048 validation frames");fig.savefig(path,dpi=150);plt.close(fig)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();run=args.run_dir.resolve();started=time.monotonic()
    dataset=CanoV2RMultitaskDataset(DATASET,"validation")
    if len(dataset)!=12500:raise RuntimeError(f"validation contract mismatch: {len(dataset)}")
    if not torch.cuda.is_available():raise RuntimeError("formal audit requires CUDA")
    torch.set_num_threads(8);records=[]
    for device_name in ("cpu","cuda"):
        for seed in SEEDS:
            records.append(audit_seed(dataset,seed,torch.device(device_name)))
    cpu=[item for item in records if item["device"]=="cpu"];cuda=[item for item in records if item["device"]=="cuda"]
    cpu_rotation_pass=all(item["rotation_direction_absolute_logit_error"]["maximum"]<=2e-5 and item["rotation_z_role_cosine"]["minimum"]>=.999 for item in cpu)
    cuda_rotation_pass=all(item["rotation_direction_absolute_logit_error"]["maximum"]<=2e-5 and item["rotation_z_role_cosine"]["minimum"]>=.999 for item in cuda)
    cpu_mask_median=float(np.median([item["fixed_masking_z_role_cosine"]["mean"] for item in cpu]));cuda_mask_median=float(np.median([item["fixed_masking_z_role_cosine"]["mean"] for item in cuda]))
    mask_device_delta=abs(cpu_mask_median-cuda_mask_median);masking_pass=cpu_mask_median>=.85
    if cpu_rotation_pass and not cuda_rotation_pass:rotation_attribution="CUDA_NUMERICAL_MAX_ERROR_NOT_ARCHITECTURE_FAILURE"
    elif cpu_rotation_pass:rotation_attribution="ROTATION_CONTRACT_PASS_ON_CPU_AND_CUDA"
    else:rotation_attribution="ROTATION_ARCHITECTURE_CONTRACT_FAILURE"
    if not masking_pass and mask_device_delta<=1e-3:masking_attribution="DEVICE_INDEPENDENT_MODEL_INPUT_ROBUSTNESS_FAILURE"
    elif masking_pass:masking_attribution="MASKING_CONTRACT_PASS"
    else:masking_attribution="MASKING_RESULT_DEVICE_SENSITIVE"
    complete=len(records)==6 and all(item["frames"]==FRAMES and item["parameter_gradients_created"]==0 for item in records)
    overall="PASS_CANO_PHASE3_STABILITY_CONTRACT_AUDIT" if complete else "FAIL_CANO_PHASE3_STABILITY_CONTRACT_AUDIT"
    summary={
        "schema_version":"cano_phase3_stability_contract_audit_v1","overall_status":overall,
        "scientific_result":"CONFIRMED_METRIC_DEVICE_CONTRACT_AND_MASKING_MODEL_FAILURE" if cpu_rotation_pass and not masking_pass and mask_device_delta<=1e-3 else "STABILITY_RESULT_REQUIRES_REVIEW",
        "source_run":str(SOURCE_RUN),"dataset_run":str(DATASET),"seeds":list(SEEDS),"validation_frames_per_seed_per_device":FRAMES,
        "unique_validation_frame_indices_read":FRAMES,"training_frames_read":0,"strict_test_frames_read":0,"mtare_frames_read":0,
        "weight_updates":0,"optimizer_steps":0,"checkpoints_created":0,"graph_runs":0,"planner_changes":0,
        "cpu_rotation_contract_pass":cpu_rotation_pass,"cuda_rotation_contract_pass":cuda_rotation_pass,"rotation_attribution":rotation_attribution,
        "cpu_masking_median_mean_cosine":cpu_mask_median,"cuda_masking_median_mean_cosine":cuda_mask_median,"cpu_cuda_masking_absolute_delta":mask_device_delta,
        "fixed_masking_contract_pass":masking_pass,"masking_attribution":masking_attribution,"duration_seconds":time.monotonic()-started,
    }
    _write_json(run/"metrics/per_seed_device.json",{"records":records});_write_json(run/"metrics/summary.json",summary);plot_results(records,run/"previews/stability_contract_cpu_cuda.png")
    print(json.dumps(summary,indent=2));return 0 if complete else 2


if __name__=="__main__":raise SystemExit(main())
