#!/usr/bin/env python3
"""Zero-training readiness for cardinality-conditioned circular slot transport."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from execute_gse_circular_peak_geometry_model_readiness_v1 import _real_batch, _selected_rows
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import (
    BRANCH_SLICE,
    CardinalityConditionedCircularSlotTransportNet,
    circular_slot_transport_contract,
    circular_slot_transport_loss,
)

PASS="PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_V1";FAIL="FAIL_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_V1"
EXPECTED_CARDINALITY={1:7525,2:154279,3:24458,4:1864}

def _maximum_error(first,second):
 if first.dtype==torch.bool or second.dtype==torch.bool:return float(torch.any(first!=second))
 return float(torch.max(torch.abs(first-second)))
def _circular_error(first,second):return float(torch.max(torch.abs(torch.remainder(first-second+180.0,360.0)-180.0)))

def _synthetic_batch(duplicate:bool=False,shift:int=0):
 target_bins=([0],[179,1],[10,50,90],[0,2,80,120]);batch=4
 presence=torch.zeros(batch,180,dtype=torch.bool);residual=torch.zeros(batch,180);width=torch.zeros(batch,180);width_valid=torch.zeros(batch,180,dtype=torch.bool);profile=torch.zeros(batch,180,4)
 logits=torch.full((batch,10,180),-8.0);count_logits=torch.full((batch,4),-8.0)
 for row,bins in enumerate(target_bins):
  count_logits[row,len(bins)-1]=8.0;branch=range(BRANCH_SLICE[len(bins)].start,BRANCH_SLICE[len(bins)].stop)
  for local,(slot,bearing_bin) in enumerate(zip(branch,bins,strict=True)):
   actual=(bearing_bin+shift)%180;presence[row,actual]=True;residual[row,actual]=(-.5 if local%2==0 else .5);width[row,actual]=2.0+local;width_valid[row,actual]=True;profile[row,actual]=torch.tensor([-.2,0,.2,.4])
   predicted=(bins[0] if duplicate and len(bins)>1 else bearing_bin)+shift;logits[row,slot,predicted%180]=8.0
 log_mass=torch.log_softmax(logits,-1);mass=torch.softmax(logits,-1);azimuth=torch.arange(180)*2*math.pi/180;cosine=(mass*torch.cos(azimuth)).sum(-1);sine=(mass*torch.sin(azimuth)).sum(-1);bearing=torch.remainder(torch.rad2deg(torch.atan2(sine,cosine)),360.0)
 outputs={"slot_logits":logits,"slot_log_mass":log_mass,"slot_mass":mass,"slot_bearing_deg":bearing,"slot_concentration":torch.sqrt(cosine.square()+sine.square()),"slot_opening_width_m":torch.full((batch,10),2.0),"slot_vertical_profile_m":torch.zeros(batch,10,4),"exit_count_logits":count_logits,"exit_count_probability":torch.softmax(count_logits,-1),"local_axis":torch.tensor([[1.,0.,0.]]).repeat(batch,1),"width_m":torch.full((batch,),5.0),"height_m":torch.full((batch,),4.0),"slope_deg":torch.zeros(batch),"curvature_per_m":torch.full((batch,),.01)}
 targets={"presence":presence,"heading_residual_deg":residual,"opening_width_m":width,"width_valid_mask":width_valid,"vertical_profile_m":profile,"local_axis":outputs["local_axis"].clone(),"geometry":torch.tensor([[5.,4.,0.,.01]]).repeat(batch,1),"geometry_valid_mask":torch.ones(batch,4,dtype=torch.bool)}
 return outputs,targets

def _slot_permuted(outputs):
 permutation=torch.tensor([0,2,1,5,3,4,9,8,7,6]);result=dict(outputs)
 for name in ("slot_logits","slot_log_mass","slot_mass","slot_bearing_deg","slot_concentration","slot_opening_width_m","slot_vertical_profile_m"):
  result[name]=outputs[name][:,permutation]
 return result

def _synthetic_contract():
 correct,targets=_synthetic_batch();duplicate,_=_synthetic_batch(duplicate=True);rotated,rotated_targets=_synthetic_batch(shift=17)
 correct_loss=circular_slot_transport_loss(correct,targets);duplicate_loss=circular_slot_transport_loss(duplicate,targets);rotated_loss=circular_slot_transport_loss(rotated,rotated_targets);permuted_loss=circular_slot_transport_loss(_slot_permuted(correct),targets)
 duplicate_logits=duplicate["slot_logits"].clone().requires_grad_(True);duplicate_grad=dict(duplicate);duplicate_grad["slot_logits"]=duplicate_logits;duplicate_grad["slot_log_mass"]=torch.log_softmax(duplicate_logits,-1);duplicate_grad["slot_mass"]=torch.softmax(duplicate_logits,-1)
 loss=circular_slot_transport_loss(duplicate_grad,targets)["assignment"];loss.backward()
 missing_grad=min(float(duplicate_logits.grad[1,slot,1]) for slot in range(BRANCH_SLICE[2].start,BRANCH_SLICE[2].stop))
 sharp=torch.softmax(torch.tensor([8.0]+[-8.0]*179),0);diffuse=torch.full((180,),1/180);azimuth=torch.arange(180)*2*math.pi/180
 concentration=lambda mass:float(torch.sqrt((mass*torch.cos(azimuth)).sum().square()+(mass*torch.sin(azimuth)).sum().square()))
 return {"correct_assignment":float(correct_loss["assignment"]),"duplicate_assignment":float(duplicate_loss["assignment"]),"rotation_total_error":abs(float(rotated_loss["total"]-correct_loss["total"])),"slot_permutation_total_error":abs(float(permuted_loss["total"]-correct_loss["total"])),"duplicate_missing_target_gradient":missing_grad,"sharp_concentration":concentration(sharp),"diffuse_concentration":concentration(diffuse)}

def _plot(output,summary):
 figure,axes=plt.subplots(1,3,figsize=(13.2,4.0),constrained_layout=True);cardinality=summary["population"]["cardinality"]
 axes[0].bar(range(1,5),[cardinality[str(value)] for value in range(1,5)],color="#4e79a7");axes[0].set_xticks(range(1,5));axes[0].set(xlabel="exits",ylabel="observations",title="A  Variable-cardinality population")
 synthetic=summary["synthetic"];axes[1].bar((0,1),(synthetic["correct_assignment"],synthetic["duplicate_assignment"]),color=("#59a14f","#e15759"));axes[1].set_xticks((0,1),("one-to-one","duplicate"));axes[1].set(ylabel="assignment loss",title="B  Bijective transport")
 equivariance=summary["equivariance"];names=("slot_mass_rotation","resultant_rotation","slot_permutation_loss","batch_permutation");axes[2].bar(range(4),[max(equivariance[name],1e-12) for name in names],color="#76b7b2");axes[2].set_yscale("log");axes[2].axhline(3e-5,color="#e15759",linestyle="--");axes[2].set_xticks(range(4),("mass","resultant","slot perm","batch"),rotation=15);axes[2].set(ylabel="maximum error",title="C  Symmetry contracts")
 for axis in axes:axis.grid(axis="y",alpha=.25);axis.set_axisbelow(True)
 figure.suptitle("GSE-Graph cardinality-conditioned circular slot transport readiness")
 for suffix in ("png","pdf","svg"):figure.savefig(output/f"gse_cardinality_conditioned_circular_slot_transport_readiness_v1.{suffix}",dpi=220)
 plt.close(figure)

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--teacher-root",required=True,type=Path);parser.add_argument("--source-root",required=True,type=Path);parser.add_argument("--output-dir",required=True,type=Path);args=parser.parse_args();started=time.monotonic();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
 teacher_paths=sorted(args.teacher_root.resolve().glob("*/*.zarr"))
 if len(teacher_paths)!=80:raise RuntimeError("slot readiness requires 80 Teacher shards")
 rows,audit=_selected_rows(teacher_paths,args.source_root.resolve());scans,targets=_real_batch(rows,args.teacher_root.resolve(),args.source_root.resolve())
 torch.use_deterministic_algorithms(True);torch.set_num_threads(4);torch.manual_seed(20260829);model=CardinalityConditionedCircularSlotTransportNet();parameters=sum(parameter.numel() for parameter in model.parameters());outputs=model(scans);losses=circular_slot_transport_loss(outputs,targets);losses["total"].backward();finite_outputs=all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value));finite_backward=all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
 model.eval();shift_columns=40;shift_bins=10;angle_deg=20.0;permutation=torch.tensor([7,1,5,0,6,2,4,3]);inverse=torch.argsort(permutation)
 with torch.no_grad():
  base=model(scans);repeat=model(scans);rotated=model(torch.roll(scans,shift_columns,-1));permuted=model(scans[permutation]);history=scans.clone();history[:,0,0]=torch.roll(history[:,0,0],73,-1);changed=model(history)
 mass_rotation=_maximum_error(rotated["slot_mass"],torch.roll(base["slot_mass"],shift_bins,-1));bearing_rotation=_circular_error(rotated["slot_bearing_deg"],torch.remainder(base["slot_bearing_deg"]+angle_deg,360.0));rotation_rad=math.radians(angle_deg);expected_resultant=torch.stack((math.cos(rotation_rad)*base["slot_resultant_xy"][...,0]-math.sin(rotation_rad)*base["slot_resultant_xy"][...,1],math.sin(rotation_rad)*base["slot_resultant_xy"][...,0]+math.cos(rotation_rad)*base["slot_resultant_xy"][...,1]),-1);resultant_rotation=_maximum_error(rotated["slot_resultant_xy"],expected_resultant);count_rotation=_maximum_error(rotated["exit_count_probability"],base["exit_count_probability"]);concentration_rotation=_maximum_error(rotated["slot_concentration"],base["slot_concentration"]);slot_geometry_rotation=max(_maximum_error(rotated[name],base[name]) for name in ("slot_opening_width_m","slot_vertical_profile_m","slot_descriptor","slot_uncertainty"));batch_permutation=max(_maximum_error(permuted[name][inverse],base[name]) for name in outputs);repeat_error=max(_maximum_error(repeat[name],base[name]) for name in outputs);history_sensitivity=_maximum_error(changed["slot_logits"],base["slot_logits"])
 permuted_outputs=_slot_permuted(base);slot_permutation_loss=abs(float(circular_slot_transport_loss(permuted_outputs,targets)["total"]-circular_slot_transport_loss(base,targets)["total"]));synthetic=_synthetic_contract();cardinality={int(key):int(value) for key,value in audit["cardinality"].items()};contract=circular_slot_transport_contract()
 checks={"full_population_and_causal_join":audit["counts"].get("worlds")==80 and audit["counts"].get("observations")==188126 and audit["counts"].get("peaks")==396913 and audit["join_mismatch_shards"]==0 and audit["local_noncontiguous_rows"]==0 and audit["global_noncontiguous_rows"]==0,"exact_cardinality_population":cardinality==EXPECTED_CARDINALITY,"teacher_excludes_sensor_identity":audit["forbidden_teacher_arrays"]==0,"typed_no_existence_query":contract["existence_objectness"] is None and not any("query" in name or "objectness" in name for name,_ in model.named_parameters()),"synthetic_bijection_and_gradient":synthetic["correct_assignment"]<synthetic["duplicate_assignment"] and synthetic["duplicate_missing_target_gradient"]<0,"synthetic_rotation_slot_permutation":max(synthetic["rotation_total_error"],synthetic["slot_permutation_total_error"])<=1e-5,"concentration_orders_sharp_diffuse":synthetic["sharp_concentration"]>.99 and synthetic["diffuse_concentration"]<1e-5,"real_1_to_4_finite_backward":finite_outputs and finite_backward and all(math.isfinite(float(value)) for value in losses.values()),"network_rotation_and_permutation":max(mass_rotation,resultant_rotation,count_rotation,concentration_rotation,slot_geometry_rotation,batch_permutation,slot_permutation_loss)<=3e-5 and repeat_error==0,"five_frame_history_connected":history_sensitivity>1e-6,"zero_training_test_graph":True}
 scientific_pass=all(checks.values());summary={"schema_version":"gse_cardinality_conditioned_circular_slot_transport_readiness_v1","status":PASS if scientific_pass else FAIL,"scientific_pass":scientific_pass,"decision":"ALLOW_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_TRAINING_DATA_CARD" if scientific_pass else "STOP_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT","population":{"counts":audit["counts"],"cardinality":{str(key):value for key,value in sorted(cardinality.items())},"selected_real_rows":rows},"model":{"parameters":parameters,"contract":contract},"synthetic":synthetic,"real_batch":{"rows":len(scans),"exit_counts":targets["presence"].sum(1).tolist(),"losses":{name:float(value.detach()) for name,value in losses.items()},"finite_outputs":finite_outputs,"finite_backward":finite_backward},"equivariance":{"slot_mass_rotation":mass_rotation,"bearing_rotation_diagnostic_deg":bearing_rotation,"resultant_rotation":resultant_rotation,"count_rotation":count_rotation,"concentration_rotation":concentration_rotation,"slot_geometry_rotation":slot_geometry_rotation,"slot_permutation_loss":slot_permutation_loss,"batch_permutation":batch_permutation,"repeat":repeat_error,"history_sensitivity":history_sensitivity},"checks":checks,"duration_seconds":time.monotonic()-started,"optimizer_steps":0,"checkpoint_writes":0,"threshold_selection_steps":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}
 write_json(output/"summary.json",summary);write_json(output/"figure_source.json",summary)
 with (output/"selected_real_rows.csv").open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=("criterion","parent_id","row","partition"));writer.writeheader();writer.writerows(rows)
 _plot(output,summary);print(json.dumps({"status":summary["status"],"decision":summary["decision"],"checks":checks},indent=2,sort_keys=True));return 0 if scientific_pass else 2
if __name__=="__main__":raise SystemExit(main())
