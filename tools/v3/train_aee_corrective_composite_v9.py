#!/usr/bin/env python3
"""V9 composite: learned retained direction plus frozen B0 count/role."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import train_aee_corrective_encoder_retention_v8r as v8r
from mtare_topo.data.aee_corrective_training import CorrectiveAEEMultitaskDataset, CorrectiveCanoMultitaskDataset, MaskMatchedCanoValidationDataset
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


def role_from_count(count: int) -> int:
    return 2 if int(count) <= 1 else 0 if int(count) == 2 else 1


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: tuple[int, ...]) -> tuple[list[float], float]:
    values=[]
    for value in classes:
        tp=int(np.sum((truth==value)&(prediction==value))); fp=int(np.sum((truth!=value)&(prediction==value))); fn=int(np.sum((truth==value)&(prediction!=value)))
        values.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0)
    return values, float(np.mean(values))


def composite_metrics(dataset_run: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    cano=CorrectiveCanoMultitaskDataset(dataset_run,"validation")
    aee=CorrectiveAEEMultitaskDataset(dataset_run)
    sparse=MaskMatchedCanoValidationDataset(cano,aee,20260822)
    baseline=RangeExitBaseline()
    truth_count=[]; prediction_count=[]; truth_role=[]
    for index in range(5000):
        item=sparse[index]
        output=baseline.predict(item["student"][0]*50.0,item["student"][1]>0.5,ELEVATION_DEG)
        truth_count.append(int(item["count_target"])+1); prediction_count.append(len(output["headings_robot_deg"])); truth_role.append(int(item["role_target"]))
    y_count=np.asarray(truth_count); p_count=np.asarray(prediction_count); y_role=np.asarray(truth_role); p_role=np.asarray([role_from_count(x) for x in p_count])
    count_f1,count_macro=macro_f1(y_count,p_count,(1,2,3,4)); role_f1,role_macro=macro_f1(y_role,p_role,(0,1,2))
    return {"frames":5000,"count":{"f1_1_to_4":count_f1,"macro_f1_count_1_to_4":count_macro,"accuracy":float(np.mean(y_count==p_count))},"role":{"f1":role_f1,"macro_f1_present":role_macro,"accuracy":float(np.mean(y_role==p_role))},"runtime_contract":"m1d_direction_b0_count_fixed_role_v1","b0_config":baseline.config.to_dict()},p_count,p_role


def main() -> int:
    code=v8r.main()
    output=Path(sys.argv[sys.argv.index("--output-dir")+1]).resolve()
    dataset_run=Path(sys.argv[sys.argv.index("--dataset-run")+1]).resolve()
    summary_path=output/"summary.json"; summary=json.loads(summary_path.read_text(encoding="utf-8"))
    composite,count_prediction,role_prediction=composite_metrics(dataset_run)
    neural_gate=dict(summary["gate"])
    gate={key:value for key,value in neural_gate.items() if key not in {"sparse_count_1_to_4_macro_f1","sparse_role_macro_f1"}}
    gate["composite_sparse_count_1_to_4_macro_f1"]=composite["count"]["macro_f1_count_1_to_4"]>=0.70
    gate["composite_sparse_role_macro_f1"]=composite["role"]["macro_f1_present"]>=0.70
    checkpoint_path=output/"best.pt"; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    checkpoint["mode"]="M1D_AEE_CORRECTIVE_COMPOSITE_V9"
    checkpoint["runtime_contract"]={"direction":"model.direction_logits","count":"frozen_b0.branch_count","role":"terminal_if_count_le_1_interior_if_2_junction_if_ge_3","neural_count_role":"diagnostic_only"}
    torch.save(checkpoint,checkpoint_path)
    np.savez_compressed(output/"composite_sparse_outputs.npz",count_prediction=count_prediction,role_prediction=role_prediction)
    summary.update(schema_version="aee_corrective_composite_seed_summary_v9",status="COMPLETED_AEE_CORRECTIVE_COMPOSITE_SEED_V9",best_checkpoint_sha256=sha256(checkpoint_path),neural_count_role_diagnostic_gate=neural_gate,gate=gate,scientific_gate_passed=all(gate.values()),composite_sparse_validation=composite,runtime_contract=checkpoint["runtime_contract"])
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return code


if __name__=="__main__": raise SystemExit(main())
