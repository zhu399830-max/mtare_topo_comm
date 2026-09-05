#!/usr/bin/env python3
"""Select C07 whole-set confidence and transfer circular slot transport to C08."""
from __future__ import annotations
import argparse,csv,json,math,time
from collections import defaultdict
from itertools import combinations,permutations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from mtare_topo.evaluation.gse_circular_peak_metrics import action_macro_f1
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE
PASS="PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_SELECTION_V1";FAIL="FAIL_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_SELECTION_V1";PRECISION_FLOOR=.995;BEARING_TOLERANCE_DEG=2.0;TOLERANCES=(2.0,4.0,10.0)
def _angular_error(a,b):return abs((a-b+180.)%360.-180.)
def _optimal_pairs(predicted,target,tolerance=math.inf):
 best=[];best_error=math.inf
 for size in range(1,min(len(predicted),len(target))+1):
  for subset in combinations(range(len(predicted)),size):
   for order in permutations(range(len(target)),size):
    pairs=[(p,t,_angular_error(float(predicted[p]),float(target[t]))) for p,t in zip(subset,order,strict=True)]
    if all(item[2]<=tolerance for item in pairs):
     error=sum(item[2] for item in pairs)
     if len(pairs)>len(best) or (len(pairs)==len(best) and error<best_error):best=pairs;best_error=error
 return best
def _align_order(reference,candidate):
 if len(reference)!=len(candidate):raise ValueError("slot alignment count drift")
 return min(permutations(range(len(candidate))),key=lambda order:sum(_angular_error(float(reference[index]),float(candidate[order[index]])) for index in range(len(reference))))
def _load_split(suffix,teacher_root,prediction_roots):
 output=defaultdict(list)
 for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
  teacher=zarr.open_group(str(teacher_path),mode="r");parent=str(teacher.attrs["parent_id"]);predictions=[np.load(root/f"{parent}.npz") for root in prediction_roots];sequence=np.asarray(teacher["global_sequence_index"][:],dtype=np.int64)
  if any(not np.array_equal(sequence,item["global_sequence_index"]) for item in predictions):raise RuntimeError(f"slot join drift:{parent}")
  values={"parent_id":np.full(len(sequence),parent,dtype=f"U{len(parent)}"),"global_sequence_index":sequence,"presence":np.asarray(teacher["presence"][:],dtype=np.uint8),"heading_target":np.asarray(teacher["heading_residual_deg"][:],dtype=np.float32),"width_target":np.asarray(teacher["opening_width_m"][:],dtype=np.float32),"width_valid":np.asarray(teacher["width_valid_mask"][:],dtype=np.uint8),"profile_target":np.asarray(teacher["vertical_profile_m"][:],dtype=np.float32),"count_probability":np.mean([np.asarray(item["exit_count_probability"],dtype=np.float32) for item in predictions],axis=0),"axis":np.mean([np.asarray(item["local_axis"],dtype=np.float32) for item in predictions],axis=0),"geometry":np.mean([np.asarray(item["geometry"],dtype=np.float32) for item in predictions],axis=0)}
  for seed,item in enumerate(predictions):
   for name in ("slot_mass","slot_bearing_deg","slot_opening_width_m","slot_vertical_profile_m"):values[f"seed{seed}_{name}"]=np.asarray(item[name],dtype=np.float32)
  for name,value in values.items():output[name].append(value)
 if len(output["presence"])!=10:raise RuntimeError(f"slot {suffix} world count drift")
 return {name:np.concatenate(parts) for name,parts in output.items()}
def _ensemble(data):
 n=len(data["presence"]);count=np.argmax(data["count_probability"],axis=1).astype(np.int64)+1;bearing=np.zeros((n,4));concentration=np.zeros((n,4));width=np.zeros((n,4));profile=np.zeros((n,4,4));mass=np.zeros((n,4,180),dtype=np.float32)
 azimuth=np.arange(180)*2*np.pi/180
 for row in range(n):
  k=int(count[row]);branch=np.arange(BRANCH_SLICE[k].start,BRANCH_SLICE[k].stop);reference=data["seed0_slot_bearing_deg"][row,branch];aligned=[]
  for seed in range(3):
   candidate=data[f"seed{seed}_slot_bearing_deg"][row,branch];order=np.arange(k) if seed==0 else np.asarray(_align_order(reference,candidate));indices=branch[order];aligned.append(indices)
  for slot in range(k):
   mass[row,slot]=np.mean([data[f"seed{seed}_slot_mass"][row,aligned[seed][slot]] for seed in range(3)],axis=0);mass[row,slot]/=mass[row,slot].sum();cosine=float((mass[row,slot]*np.cos(azimuth)).sum());sine=float((mass[row,slot]*np.sin(azimuth)).sum());bearing[row,slot]=np.degrees(np.arctan2(sine,cosine))%360;concentration[row,slot]=np.hypot(cosine,sine);width[row,slot]=np.mean([data[f"seed{seed}_slot_opening_width_m"][row,aligned[seed][slot]] for seed in range(3)]);profile[row,slot]=np.mean([data[f"seed{seed}_slot_vertical_profile_m"][row,aligned[seed][slot]] for seed in range(3)],axis=0)
 score=data["count_probability"].max(axis=1)*np.asarray([concentration[row,:count[row]].min() for row in range(n)])
 return {"count":count,"bearing":bearing,"concentration":concentration,"width":width,"profile":profile,"score":score}
def _match(data,decoded,tolerance):
 row_tp=np.zeros(len(data["presence"]),dtype=np.int64);exact=np.zeros(len(row_tp),dtype=bool);matches=[]
 for row in range(len(row_tp)):
  bins=np.flatnonzero(data["presence"][row]);target=np.mod(bins*2.+data["heading_target"][row,bins],360.);predicted=decoded["bearing"][row,:decoded["count"][row]];pairs=_optimal_pairs(predicted,target,tolerance);row_tp[row]=len(pairs);exact[row]=len(pairs)==len(predicted)==len(target)
  for slot,target_index,error in pairs:matches.append((row,slot,int(bins[target_index]),error))
 return {"row_tp":row_tp,"exact":exact,"matches":matches}
def _select(score,match,predicted_count,total_targets):
 order=np.argsort(-score,kind="stable");values=score[order];tp=np.cumsum(match["row_tp"][order]);pred=np.cumsum(predicted_count[order]);ends=np.flatnonzero(np.r_[values[1:]!=values[:-1],True]);precision=tp[ends]/np.maximum(pred[ends],1);recall=tp[ends]/total_targets;safe=np.flatnonzero(precision>=PRECISION_FLOOR)
 if not len(safe):return None
 best=max(safe,key=lambda i:(float(recall[i]),float(precision[i]),float(values[ends[i]])));return {"threshold":float(values[ends[best]]),"precision":float(precision[best]),"recall":float(recall[best]),"accepted_observations":int(ends[best]+1)}
def _evaluate(data,decoded,threshold,source_root):
 target_count=data["presence"].sum(1).astype(np.int64);matches={t:_match(data,decoded,t) for t in TOLERANCES};formal=matches[2.0];accepted=decoded["score"]>=threshold;tp=int(formal["row_tp"][accepted].sum());pred=int(decoded["count"][accepted].sum());total=int(target_count.sum());deployed=np.where(accepted,decoded["count"],0)
 cardinality=[]
 for k in range(1,5):
  rows=target_count==k;record={"target_count":k,"observations":int(rows.sum()),"count_accuracy":float(np.mean(decoded["count"][rows]==k))}
  for tolerance in TOLERANCES:record[f"exact_set_fraction_{int(tolerance)}deg"]=float(matches[tolerance]["exact"][rows].mean())
  record["safe_exact_accepted_fraction_2deg"]=float((formal["exact"]&accepted)[rows].mean());cardinality.append(record)
 accepted_matches=[item for item in formal["matches"] if accepted[item[0]]];bearing_errors=[item[3] for item in accepted_matches];width_errors=[];profile_errors=[]
 for row,slot,target_bin,_ in accepted_matches:
  if data["width_valid"][row,target_bin]:width_errors.append(abs(float(decoded["width"][row,slot]-data["width_target"][row,target_bin])))
  profile_errors.extend(np.abs(decoded["profile"][row,slot]-data["profile_target"][row,target_bin]).tolist())
 axis_target=np.empty((len(data["presence"]),3),dtype=np.float32);geometry_target=np.empty((len(data["presence"]),4),dtype=np.float32);valid=np.empty((len(data["presence"]),4),dtype=bool)
 for parent in np.unique(data["parent_id"]):
  rows=np.flatnonzero(data["parent_id"]==parent);source=zarr.open_group(str(source_root/f"{parent}.zarr"),mode="r");source_sequence=np.asarray(source["global_sequence_index"][:],dtype=np.int64)
  if not np.array_equal(data["global_sequence_index"][rows],source_sequence):raise RuntimeError(f"slot source join drift:{parent}")
  axis_target[rows]=np.asarray(source["local_axis_robot"][:],dtype=np.float32);geometry_target[rows]=np.asarray(source["geometry"][:],dtype=np.float32);valid[rows]=np.asarray(source["geometry_valid_mask"][:],dtype=bool)
 pred_axis=data["axis"]/np.clip(np.linalg.norm(data["axis"],axis=1,keepdims=True),1e-12,None);true_axis=axis_target/np.clip(np.linalg.norm(axis_target,axis=1,keepdims=True),1e-12,None);axis_error=float(np.degrees(np.arccos(np.clip((pred_axis*true_axis).sum(1),-1,1))).mean());global_mae=[float(np.mean(np.abs(data["geometry"][valid[:,index],index]-geometry_target[valid[:,index],index]))) for index in range(4)]
 per_world=[]
 for parent in np.unique(data["parent_id"]):
  rows=data["parent_id"]==parent;world_tp=int(formal["row_tp"][rows&accepted].sum());world_pred=int(decoded["count"][rows&accepted].sum());world_target=int(target_count[rows].sum());per_world.append({"parent_id":str(parent),"precision":world_tp/max(world_pred,1),"recall":world_tp/max(world_target,1),"accepted_observations":int((rows&accepted).sum()),"observations":int(rows.sum())})
 return {"detection":{"precision":tp/max(pred,1),"recall":tp/max(total,1),"true_positive":tp,"false_positive":pred-tp,"false_negative":total-tp},"refusal":{"threshold":threshold,"coverage":float(accepted.mean()),"accepted_observations":int(accepted.sum()),"safe_exact_set_fraction_all":float((formal["exact"]&accepted).mean()),"raw_exact_set_fraction_2deg":float(formal["exact"].mean())},"cardinality":{"raw_accuracy":float(np.mean(decoded["count"]==target_count)),"raw_action":action_macro_f1(decoded["count"],target_count),"deployed_action":action_macro_f1(deployed,target_count),"strata":cardinality},"slot_geometry":{"bearing_mae_deg":float(np.mean(bearing_errors)) if bearing_errors else math.inf,"opening_width_mae_m":float(np.mean(width_errors)) if width_errors else math.inf,"vertical_profile_mae_m":float(np.mean(profile_errors)) if profile_errors else math.inf,"matched_exits":len(accepted_matches)},"global_geometry":{"axis_mean_error_deg":axis_error,"width_mae_m":global_mae[0],"height_mae_m":global_mae[1],"slope_mae_deg":global_mae[2],"curvature_mae_per_m":global_mae[3]},"per_world":per_world}
def _plot(output,summary):
 figure,axes=plt.subplots(1,3,figsize=(13.4,4.1),constrained_layout=True);splits=("c07","c08");x=np.arange(2);axes[0].bar(x-.18,[summary[s]["detection"]["precision"] for s in splits],.36,label="precision");axes[0].bar(x+.18,[summary[s]["detection"]["recall"] for s in splits],.36,label="recall");axes[0].axhline(.995,color="#e15759",linestyle="--");axes[0].set_xticks(x,("C07","C08"));axes[0].set_ylim(0,1.03);axes[0].set_title("A  Safe exit sets");axes[0].legend(frameon=False)
 for split,color in (("c07","#4e79a7"),("c08","#f28e2b")):axes[1].plot(range(1,5),[row["exact_set_fraction_2deg"] for row in summary[split]["cardinality"]["strata"]],marker="o",label=split.upper(),color=color);axes[1].set_xticks(range(1,5));axes[1].set_ylim(0,1);axes[1].set(xlabel="true exits",ylabel="raw exact-set fraction",title="B  Cardinality-stratified recovery");axes[1].legend(frameon=False)
 names=("bearing_mae_deg","opening_width_mae_m","vertical_profile_mae_m");axes[2].bar(np.arange(3)-.15,[summary["c07"]["slot_geometry"][name] for name in names],.3,label="C07");axes[2].bar(np.arange(3)+.15,[summary["c08"]["slot_geometry"][name] for name in names],.3,label="C08");axes[2].set_xticks(np.arange(3),("bearing deg","width m","profile m"),rotation=15);axes[2].set_title("C  Matched slot geometry");axes[2].legend(frameon=False)
 for axis in axes:axis.grid(axis="y",alpha=.25);axis.set_axisbelow(True)
 figure.suptitle("GSE-Graph circular slot-transport selection and transfer")
 for suffix in ("png","pdf","svg"):figure.savefig(output/f"gse_cardinality_conditioned_circular_slot_transport_selection_v1.{suffix}",dpi=220)
 plt.close(figure)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--teacher-root",required=True,type=Path);parser.add_argument("--source-root",required=True,type=Path);parser.add_argument("--prediction-root",required=True,action="append",type=Path);parser.add_argument("--output-dir",required=True,type=Path);args=parser.parse_args();started=time.monotonic()
 if len(args.prediction_root)!=3:raise RuntimeError("slot selection requires 3 seeds")
 output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);c07_data=_load_split("C07",args.teacher_root.resolve(),[path.resolve() for path in args.prediction_root]);c08_data=_load_split("C08",args.teacher_root.resolve(),[path.resolve() for path in args.prediction_root]);c07_decoded=_ensemble(c07_data);c08_decoded=_ensemble(c08_data);match07=_match(c07_data,c07_decoded,2.0);choice=_select(c07_decoded["score"],match07,c07_decoded["count"],int(c07_data["presence"].sum()));threshold=2.0 if choice is None else float(choice["threshold"]);c07=_evaluate(c07_data,c07_decoded,threshold,args.source_root.resolve());c08=_evaluate(c08_data,c08_decoded,threshold,args.source_root.resolve())
 def stratum(metrics,k):return next(row for row in metrics["cardinality"]["strata"] if row["target_count"]==k)
 checks={"c07_safe_precision_recall":c07["detection"]["precision"]>=.995 and c07["detection"]["recall"]>=.50,"c08_safe_precision_recall":c08["detection"]["precision"]>=.995 and c08["detection"]["recall"]>=.50,"raw_cardinality_accuracy":c07["cardinality"]["raw_accuracy"]>=.80 and c08["cardinality"]["raw_accuracy"]>=.80,"deployed_action_macro_f1":c07["cardinality"]["deployed_action"]["macro_f1"]>=.80 and c08["cardinality"]["deployed_action"]["macro_f1"]>=.80,"overall_exact_set":c07["refusal"]["raw_exact_set_fraction_2deg"]>=.50 and c08["refusal"]["raw_exact_set_fraction_2deg"]>=.50 and c07["refusal"]["safe_exact_set_fraction_all"]>=.50 and c08["refusal"]["safe_exact_set_fraction_all"]>=.50,"three_four_exit_recovery":all(stratum(metrics,3)["exact_set_fraction_2deg"]>=.40 and stratum(metrics,3)["safe_exact_accepted_fraction_2deg"]>=.30 and stratum(metrics,4)["exact_set_fraction_2deg"]>=.20 and stratum(metrics,4)["safe_exact_accepted_fraction_2deg"]>=.10 for metrics in (c07,c08)),"slot_geometry_contract":all(metrics["slot_geometry"]["bearing_mae_deg"]<=1 and metrics["slot_geometry"]["opening_width_mae_m"]<=3 and metrics["slot_geometry"]["vertical_profile_mae_m"]<=1 for metrics in (c07,c08)),"global_geometry_contract":all(metrics["global_geometry"]["axis_mean_error_deg"]<=10 and metrics["global_geometry"]["width_mae_m"]<=2 and metrics["global_geometry"]["height_mae_m"]<=2 and metrics["global_geometry"]["slope_mae_deg"]<=2 and metrics["global_geometry"]["curvature_mae_per_m"]<=.02 for metrics in (c07,c08)),"c08_not_used_for_selection":True,"zero_test_graph_planner":True};scientific_pass=choice is not None and all(checks.values());summary={"schema_version":"gse_cardinality_conditioned_circular_slot_transport_selection_v1","status":PASS if scientific_pass else FAIL,"scientific_pass":scientific_pass,"decision":"ALLOW_GSE_STRUCTURE_NODE_GENERATION_READINESS" if scientific_pass else "STOP_CIRCULAR_SLOT_TRANSPORT_BEFORE_GRAPH","selected_on":"C07 only","transferred_once_to":"C08 zero adaptation","precision_floor":PRECISION_FLOOR,"bearing_tolerance_deg":BEARING_TOLERANCE_DEG,"threshold_choice":choice,"c07":c07,"c08":c08,"checks":checks,"duration_seconds":time.monotonic()-started,"optimizer_steps":0,"c08_checkpoint_observations":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0};write_json(output/"summary.json",summary);write_json(output/"figure_source.json",summary)
 with (output/"per_world_metrics.csv").open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=("split","parent_id","precision","recall","accepted_observations","observations"));writer.writeheader();[writer.writerow({"split":split,**row}) for split,metrics in (("C07",c07),("C08",c08)) for row in metrics["per_world"]]
 _plot(output,summary);print(json.dumps({"status":summary["status"],"decision":summary["decision"],"threshold":None if choice is None else choice["threshold"],"checks":checks},indent=2,sort_keys=True));return 0 if scientific_pass else 2
if __name__=="__main__":raise SystemExit(main())
