#!/usr/bin/env python3
"""Select one C07 refusal threshold and transfer exit-set process to C08."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations, permutations
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.evaluation.gse_circular_peak_metrics import action_macro_f1
from mtare_topo.governance import write_json


PASS="PASS_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_SELECTION_V1"
FAIL="FAIL_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_SELECTION_V1"
PRECISION_FLOOR=0.995
BEARING_TOLERANCE_DEG=2.0
SUPPRESSION_RADIUS_BINS=1


def _load_split(suffix:str,teacher_root:Path,source_root:Path,prediction_roots:list[Path])->dict[str,np.ndarray]:
    output:dict[str,list[np.ndarray]]=defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher=zarr.open_group(str(teacher_path),mode="r");parent=str(teacher.attrs["parent_id"]);source=zarr.open_group(str(source_root/f"{parent}.zarr"),mode="r");predictions=[np.load(root/f"{parent}.npz") for root in prediction_roots]
        global_sequence=np.asarray(teacher["global_sequence_index"][:],dtype=np.int64)
        if not np.array_equal(global_sequence,np.asarray(source["global_sequence_index"][:],dtype=np.int64)) or any(not np.array_equal(global_sequence,item["global_sequence_index"]) for item in predictions):raise RuntimeError(f"exit-set join drift:{parent}")
        mass=np.mean([np.asarray(item["exit_mass"],dtype=np.float64) for item in predictions],axis=0);mass/=mass.sum(axis=1,keepdims=True)
        values={
            "global_sequence_index":global_sequence,"parent_id":np.full(len(global_sequence),parent,dtype=f"U{len(parent)}"),
            "presence":np.asarray(teacher["presence"][:],dtype=np.uint8),"heading_target":np.asarray(teacher["heading_residual_deg"][:],dtype=np.float32),
            "width_target":np.asarray(teacher["opening_width_m"][:],dtype=np.float32),"width_valid":np.asarray(teacher["width_valid_mask"][:],dtype=np.uint8),"profile_target":np.asarray(teacher["vertical_profile_m"][:],dtype=np.float32),
            "axis_target":np.asarray(source["local_axis_robot"][:],dtype=np.float32),"geometry_target":np.asarray(source["geometry"][:],dtype=np.float32),"geometry_valid":np.asarray(source["geometry_valid_mask"][:],dtype=np.uint8),
            "mass":mass,"count_probability":np.mean([np.asarray(item["exit_count_probability"],dtype=np.float64) for item in predictions],axis=0),
            "heading":np.mean([np.asarray(item["heading_residual_deg"],dtype=np.float32) for item in predictions],axis=0),"width":np.mean([np.asarray(item["opening_width_m"],dtype=np.float32) for item in predictions],axis=0),"profile":np.mean([np.asarray(item["vertical_profile_m"],dtype=np.float32) for item in predictions],axis=0),
            "axis":np.mean([np.asarray(item["local_axis"],dtype=np.float32) for item in predictions],axis=0),"geometry":np.mean([np.asarray(item["geometry"],dtype=np.float32) for item in predictions],axis=0),
        }
        for seed,item in enumerate(predictions):values[f"seed{seed}_mass"]=np.asarray(item["exit_mass"],dtype=np.float64);values[f"seed{seed}_count_probability"]=np.asarray(item["exit_count_probability"],dtype=np.float64)
        for name,value in values.items():output[name].append(value)
    if len(output["global_sequence_index"])!=10:raise RuntimeError(f"exit-set {suffix} world count drift")
    return {name:np.concatenate(parts) for name,parts in output.items()}


def _decode(mass:np.ndarray,count_probability:np.ndarray,heading:np.ndarray)->dict[str,np.ndarray]:
    count=np.argmax(count_probability,axis=1).astype(np.int64)+1;bins=np.full((len(mass),4),-1,dtype=np.int64);valid=np.zeros((len(mass),4),dtype=bool);selected_mass=np.zeros((len(mass),4),dtype=np.float64)
    for row,values in enumerate(mass):
        available=np.ones(180,dtype=bool)
        for slot in range(int(count[row])):
            bearing_bin=int(np.argmax(np.where(available,values,-1.0)));bins[row,slot]=bearing_bin;valid[row,slot]=True;selected_mass[row,slot]=values[bearing_bin]
            for delta in range(-SUPPRESSION_RADIUS_BINS,SUPPRESSION_RADIUS_BINS+1):available[(bearing_bin+delta)%180]=False
    safe=np.maximum(bins,0);bearing=np.mod(safe*2.0+heading[np.arange(len(mass))[:,None],safe],360.0);bearing[~valid]=0
    normalized_min=np.asarray([min(float(count[row])*selected_mass[row,:count[row]].min(),1.0) for row in range(len(mass))])
    score=count_probability.max(axis=1)*normalized_min
    return {"count":count,"bins":bins,"valid":valid,"bearing_deg":bearing,"selected_mass":selected_mass,"score":score,"count_confidence":count_probability.max(axis=1)}


def _angular_error(first:float,second:float)->float:
    return abs((first-second+180.0)%360.0-180.0)


def _best_pairs(predicted:np.ndarray,target:np.ndarray)->list[tuple[int,int,float]]:
    best:list[tuple[int,int,float]]=[];best_error=math.inf
    for size in range(1,min(len(predicted),len(target))+1):
        for pred_subset in combinations(range(len(predicted)),size):
            for target_order in permutations(range(len(target)),size):
                pairs=[(p,t,_angular_error(float(predicted[p]),float(target[t]))) for p,t in zip(pred_subset,target_order,strict=True)]
                if all(item[2]<=BEARING_TOLERANCE_DEG for item in pairs):
                    error=sum(item[2] for item in pairs)
                    if len(pairs)>len(best) or (len(pairs)==len(best) and error<best_error):best=pairs;best_error=error
    return best


def _match_rows(data:dict[str,np.ndarray],decoded:dict[str,np.ndarray])->dict[str,object]:
    row_tp=np.zeros(len(data["presence"]),dtype=np.int64);row_exact=np.zeros(len(row_tp),dtype=bool);matches=[]
    for row in range(len(row_tp)):
        truth_bins=np.flatnonzero(data["presence"][row]);target_bearing=np.mod(truth_bins*2.0+data["heading_target"][row,truth_bins],360.0);predicted=decoded["bearing_deg"][row,:decoded["count"][row]]
        pairs=_best_pairs(predicted,target_bearing);row_tp[row]=len(pairs);row_exact[row]=len(pairs)==len(predicted)==len(target_bearing)
        for slot,target_index,error in pairs:matches.append((row,slot,int(truth_bins[target_index]),error))
    return {"row_tp":row_tp,"row_exact":row_exact,"matches":matches}


def _select_threshold(score:np.ndarray,match:dict[str,object],predicted_count:np.ndarray,total_targets:int)->dict|None:
    order=np.argsort(-score,kind="stable");values=score[order];tp=np.cumsum(match["row_tp"][order]);pred=np.cumsum(predicted_count[order]);ends=np.flatnonzero(np.r_[values[1:]!=values[:-1],True]);precision=tp[ends]/np.maximum(pred[ends],1);recall=tp[ends]/total_targets;safe=np.flatnonzero(precision>=PRECISION_FLOOR)
    if not len(safe):return None
    best=max(safe,key=lambda i:(float(recall[i]),float(precision[i]),float(values[ends[i]])))
    return {"threshold":float(values[ends[best]]),"precision":float(precision[best]),"recall":float(recall[best]),"accepted_observations":int(ends[best]+1),"true_positive_exits":int(tp[ends[best]]),"predicted_exits":int(pred[ends[best]])}


def _evaluate(data:dict[str,np.ndarray],threshold:float)->dict:
    decoded=_decode(data["mass"],data["count_probability"],data["heading"]);match=_match_rows(data,decoded);accepted=decoded["score"]>=threshold;target_count=data["presence"].sum(axis=1).astype(np.int64);tp=int(match["row_tp"][accepted].sum());predicted=int(decoded["count"][accepted].sum());total=int(target_count.sum());precision=tp/max(predicted,1);recall=tp/max(total,1)
    deployed_count=np.where(accepted,decoded["count"],0);action=action_macro_f1(deployed_count,target_count);raw_action=action_macro_f1(decoded["count"],target_count)
    accepted_matches=[item for item in match["matches"] if accepted[item[0]]];heading_errors=[item[3] for item in accepted_matches];width_errors=[];profile_errors=[]
    for row,slot,target_bin,_ in accepted_matches:
        pred_bin=decoded["bins"][row,slot]
        if data["width_valid"][row,target_bin]:width_errors.append(abs(float(data["width"][row,pred_bin]-data["width_target"][row,target_bin])))
        profile_errors.extend(np.abs(data["profile"][row,pred_bin]-data["profile_target"][row,target_bin]).tolist())
    predicted_axis=data["axis"]/np.clip(np.linalg.norm(data["axis"],axis=1,keepdims=True),1e-12,None);target_axis=data["axis_target"]/np.clip(np.linalg.norm(data["axis_target"],axis=1,keepdims=True),1e-12,None);axis_error=float(np.degrees(np.arccos(np.clip(np.sum(predicted_axis*target_axis,axis=1),-1,1))).mean())
    global_mae=[]
    for index in range(4):
        valid=data["geometry_valid"][:,index].astype(bool);global_mae.append(float(np.mean(np.abs(data["geometry"][valid,index]-data["geometry_target"][valid,index]))))
    confusion=np.zeros((4,4),dtype=np.int64)
    for truth,prediction in zip(target_count,decoded["count"],strict=True):confusion[truth-1,prediction-1]+=1
    per_world=[]
    for parent in np.unique(data["parent_id"]):
        rows=data["parent_id"]==parent;world_tp=int(match["row_tp"][rows&accepted].sum());world_pred=int(decoded["count"][rows&accepted].sum());world_target=int(target_count[rows].sum());per_world.append({"parent_id":str(parent),"precision":world_tp/max(world_pred,1),"recall":world_tp/max(world_target,1),"accepted_observations":int((rows&accepted).sum()),"observations":int(rows.sum())})
    return {
        "detection":{"precision":precision,"recall":recall,"true_positive":tp,"false_positive":predicted-tp,"false_negative":total-tp,"predicted_exits":predicted},
        "refusal":{"threshold":threshold,"accepted_observations":int(accepted.sum()),"coverage":float(accepted.mean()),"exact_set_fraction_all":float((match["row_exact"]&accepted).mean()),"exact_rate_accepted":float(match["row_exact"][accepted].mean()) if accepted.any() else 0.0},
        "cardinality":{"raw_accuracy":float(np.mean(decoded["count"]==target_count)),"confusion":confusion.tolist(),"raw_action":raw_action,"deployed_action":action},
        "peak_geometry":{"bearing_mae_deg":float(np.mean(heading_errors)) if heading_errors else math.inf,"opening_width_mae_m":float(np.mean(width_errors)) if width_errors else math.inf,"vertical_profile_mae_m":float(np.mean(profile_errors)) if profile_errors else math.inf,"matched_exits":len(accepted_matches),"matched_width_exits":len(width_errors)},
        "global_geometry":{"axis_mean_error_deg":axis_error,"width_mae_m":global_mae[0],"height_mae_m":global_mae[1],"slope_mae_deg":global_mae[2],"curvature_mae_per_m":global_mae[3]},
        "score":{"minimum":float(decoded["score"].min()),"median":float(np.median(decoded["score"])),"maximum":float(decoded["score"].max())},"per_world":per_world,
    }


def _plot(output:Path,summary:dict)->None:
    figure,axes=plt.subplots(1,3,figsize=(13.2,4.0),constrained_layout=True);splits=("c07","c08");x=np.arange(2)
    axes[0].bar(x-.18,[summary[s]["detection"]["precision"] for s in splits],.36,label="precision",color="#4e79a7");axes[0].bar(x+.18,[summary[s]["detection"]["recall"] for s in splits],.36,label="recall",color="#f28e2b");axes[0].axhline(.995,color="#e15759",linestyle="--",linewidth=1);axes[0].set_xticks(x,("C07","C08"));axes[0].set_ylim(0,1.03);axes[0].set_title("A  Safe continuous exit sets");axes[0].legend(frameon=False)
    axes[1].bar(x-.18,[summary[s]["cardinality"]["raw_accuracy"] for s in splits],.36,label="count accuracy",color="#59a14f");axes[1].bar(x+.18,[summary[s]["cardinality"]["deployed_action"]["macro_f1"] for s in splits],.36,label="action macro-F1",color="#af7aa1");axes[1].set_xticks(x,("C07","C08"));axes[1].set_ylim(0,1);axes[1].set_title("B  Structure cardinality");axes[1].legend(frameon=False)
    names=("bearing_mae_deg","opening_width_mae_m","vertical_profile_mae_m");labels=("bearing (deg)","width (m)","profile (m)")
    for index,split in enumerate(splits):axes[2].bar(np.arange(3)+(index-.5)*.22,[summary[split]["peak_geometry"][name] for name in names],.22,label=split.upper())
    axes[2].set_xticks(np.arange(3),labels,rotation=15);axes[2].set_title("C  Matched exit geometry");axes[2].legend(frameon=False)
    for axis in axes:axis.grid(axis="y",alpha=.25);axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph causal circular exit-set selection and transfer")
    for suffix in ("png","pdf","svg"):figure.savefig(output/f"gse_circular_exit_set_process_selection_v1.{suffix}",dpi=220)
    plt.close(figure)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--teacher-root",required=True,type=Path);parser.add_argument("--source-root",required=True,type=Path);parser.add_argument("--prediction-root",action="append",required=True,type=Path);parser.add_argument("--output-dir",required=True,type=Path);args=parser.parse_args();started=time.monotonic()
    if len(args.prediction_root)!=3:raise RuntimeError("exit-set selection requires 3 seeds")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);c07=_load_split("C07",args.teacher_root.resolve(),args.source_root.resolve(),args.prediction_root);c08=_load_split("C08",args.teacher_root.resolve(),args.source_root.resolve(),args.prediction_root)
    decoded07=_decode(c07["mass"],c07["count_probability"],c07["heading"]);match07=_match_rows(c07,decoded07);choice=_select_threshold(decoded07["score"],match07,decoded07["count"],int(c07["presence"].sum()));threshold=2.0 if choice is None else float(choice["threshold"]);c07_metrics=_evaluate(c07,threshold);c08_metrics=_evaluate(c08,threshold)
    checks={
        "c07_safe_precision_recall":c07_metrics["detection"]["precision"]>=.995 and c07_metrics["detection"]["recall"]>=.50,
        "c08_safe_precision_recall":c08_metrics["detection"]["precision"]>=.995 and c08_metrics["detection"]["recall"]>=.50,
        "raw_cardinality_accuracy":c07_metrics["cardinality"]["raw_accuracy"]>=.80 and c08_metrics["cardinality"]["raw_accuracy"]>=.80,
        "deployed_action_macro_f1":c07_metrics["cardinality"]["deployed_action"]["macro_f1"]>=.80 and c08_metrics["cardinality"]["deployed_action"]["macro_f1"]>=.80,
        "exact_set_fraction":c07_metrics["refusal"]["exact_set_fraction_all"]>=.50 and c08_metrics["refusal"]["exact_set_fraction_all"]>=.50,
        "peak_geometry_contract":all(m["peak_geometry"]["bearing_mae_deg"]<=1.0 and m["peak_geometry"]["opening_width_mae_m"]<=3.0 and m["peak_geometry"]["vertical_profile_mae_m"]<=1.0 for m in (c07_metrics,c08_metrics)),
        "global_geometry_contract":all(m["global_geometry"]["axis_mean_error_deg"]<=10 and m["global_geometry"]["width_mae_m"]<=2 and m["global_geometry"]["height_mae_m"]<=2 and m["global_geometry"]["slope_mae_deg"]<=2 and m["global_geometry"]["curvature_mae_per_m"]<=.02 for m in (c07_metrics,c08_metrics)),
        "c08_not_used_for_selection":True,"zero_test_graph_planner":True,
    }
    scientific_pass=choice is not None and all(checks.values());summary={"schema_version":"gse_circular_exit_set_process_selection_v1","status":PASS if scientific_pass else FAIL,"scientific_pass":scientific_pass,"decision":"ALLOW_GSE_STRUCTURE_NODE_GENERATION_READINESS" if scientific_pass else "STOP_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_BEFORE_GRAPH","selected_on":"C07 only","transferred_once_to":"C08 zero adaptation","precision_floor":PRECISION_FLOOR,"bearing_tolerance_deg":BEARING_TOLERANCE_DEG,"ensemble_refusal_threshold":None if choice is None else threshold,"c07_threshold_selection":choice,"population":{"c07_observations":len(c07["presence"]),"c07_exits":int(c07["presence"].sum()),"c08_observations":len(c08["presence"]),"c08_exits":int(c08["presence"].sum())},"c07":c07_metrics,"c08":c08_metrics,"checks":checks,"duration_seconds":time.monotonic()-started,"optimizer_steps":0,"c08_checkpoint_observations":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}
    write_json(output/"summary.json",summary);write_json(output/"figure_source.json",{"schema_version":"gse_circular_exit_set_process_selection_figure_source_v1","summary":summary})
    with (output/"per_world_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=("split","parent_id","precision","recall","accepted_observations","observations"));writer.writeheader()
        for split,metrics in (("C07",c07_metrics),("C08",c08_metrics)):
            for row in metrics["per_world"]:writer.writerow({"split":split,**row})
    _plot(output,summary);print(json.dumps({"status":summary["status"],"decision":summary["decision"],"threshold":summary["ensemble_refusal_threshold"],"checks":checks},indent=2,sort_keys=True));return 0 if scientific_pass else 2
if __name__=="__main__":raise SystemExit(main())
