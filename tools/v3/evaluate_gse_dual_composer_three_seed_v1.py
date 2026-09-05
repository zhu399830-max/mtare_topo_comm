#!/usr/bin/env python3
"""Combine three trained dual-Composer seeds and apply the frozen method gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.evaluation.gse_validation_calibration import fit_event_temperature,precision_constrained_threshold
from mtare_topo.governance import write_json
from mtare_topo.semantics.gse_composer_learning import EVENT_NAMES,identity_coverage,macro_f1,structural_acceptance_metrics


PASS="PASS_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1";FAIL="FAIL_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1"
VARIANTS=("full","no_metric","no_transport","single_frame","count_bearing_only")


def _labels(event):
    lookup={name:index for index,name in enumerate(EVENT_NAMES)};return np.asarray([lookup[str(value)] for value in event],dtype=np.int64)


def _softmax(logits,temperature=1.0):
    value=np.asarray(logits,dtype=np.float64)/temperature;value-=value.max(1,keepdims=True);value=np.exp(value);return value/value.sum(1,keepdims=True)


def _select(probability,prediction,score,truth):
    correct=(prediction==truth)&(truth!=0);eligible=truth!=0
    try:point,curve=precision_constrained_threshold(score,correct,eligible,minimum_precision=.98,maximum_false_accept_rate=.01);return point.threshold,curve,None
    except RuntimeError as exc:return 2.0,[],str(exc)


def _metrics(truth,prediction,score,threshold,identities):
    return {"raw":macro_f1(truth,prediction),"selective":structural_acceptance_metrics(truth,prediction,score,threshold),"identity_coverage":identity_coverage(truth,prediction,score,threshold,identities)}


def _variant(seed_arrays,summaries,name,split):
    probabilities=[];action=[];metric=[]
    for seed in range(3):
        source="full" if name=="no_refusal" else name;temperature=float(summaries[seed]["selection"][source]["temperature"]["temperature"])
        probabilities.append(_softmax(seed_arrays[seed][split][f"{source}_logits"],temperature));action.append(seed_arrays[seed][split][f"{source}_action_commit"]);metric.append(seed_arrays[seed][split][f"{source}_metric_commit"])
    probability=np.mean(probabilities,axis=0);prediction=probability.argmax(1);confidence=probability[np.arange(len(prediction)),prediction]
    commit=np.ones(len(prediction));action_mask=(prediction==1)|(prediction==2);metric_mask=(prediction==3)|(prediction==4)
    action_commit=np.mean(action,axis=0);metric_commit=np.mean(metric,axis=0);commit[action_mask]=action_commit[action_mask];commit[metric_mask]=metric_commit[metric_mask]
    score=confidence*commit;score[prediction==0]=0.0
    return probability,prediction,score


def _baseline(seed_arrays,split,c07_labels=None):
    probabilities=[]
    for seed in range(3):
        logits=seed_arrays[seed][split]["baseline_logits"]
        if c07_labels is None:temperature=1.0
        else:temperature=float(fit_event_temperature(logits,c07_labels)["temperature"])
        probabilities.append(_softmax(logits,temperature))
    probability=np.mean(probabilities,axis=0);prediction=probability.argmax(1);entropy=-(probability*np.log(np.maximum(probability,1e-12))).sum(1)/math.log(5.0);score=probability[np.arange(len(prediction)),prediction]*(1.0-entropy);score[prediction==0]=0.0
    return probability,prediction,score


def _plot(output,summary):
    figure,axes=plt.subplots(1,3,figsize=(14,4.5));names=("baseline","full","no_metric","no_transport","single_frame","count_bearing_only")
    values=[summary["ensemble"][name]["c08"]["raw"]["macro_f1"] for name in names]
    axes[0].bar(names,values,color=["#718096","#2b6cb0","#dd6b20","#d69e2e","#805ad5","#c53030"]);axes[0].tick_params(axis="x",rotation=40);axes[0].set_ylim(0,1);axes[0].set_title("C08 corrected-Teacher macro-F1")
    events=("junction","terminal","turn","geometry_transition");full=summary["ensemble"]["full"]["c08"]["identity_coverage"];base=summary["ensemble"]["baseline"]["c08"]["identity_coverage"]
    x=np.arange(4);axes[1].bar(x-.18,[base[e]["coverage"] for e in events],.36,label="V2R5",color="#718096");axes[1].bar(x+.18,[full[e]["coverage"] for e in events],.36,label="GSE",color="#2b6cb0");axes[1].set_xticks(x,events,rotation=35);axes[1].set_ylim(0,1);axes[1].legend();axes[1].set_title("High-precision physical identity coverage")
    safety=summary["ensemble"]["full"];axes[2].bar(("C07 precision","C08 precision","C07 recall","C08 recall"),(safety["c07"]["selective"]["precision"],safety["c08"]["selective"]["precision"],safety["c07"]["selective"]["recall"],safety["c08"]["selective"]["recall"]),color=("#2f855a","#2f855a","#2b6cb0","#2b6cb0"));axes[2].axhline(.98,color="#c53030",linestyle="--",linewidth=1);axes[2].tick_params(axis="x",rotation=35);axes[2].set_ylim(0,1);axes[2].set_title("Structural commit safety")
    figure.tight_layout()
    for suffix in ("png","pdf","svg"):figure.savefig(output/f"gse_dual_composer_three_seed_training_v1.{suffix}",dpi=220)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--models-root",required=True,type=Path);parser.add_argument("--output-dir",required=True,type=Path);args=parser.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    summaries=[];arrays=[]
    for seed in range(3):
        root=args.models_root.resolve()/f"seed{seed}";summaries.append(json.loads((root/"summary.json").read_text()))
        arrays.append({split:dict(np.load(root/"predictions"/f"{split}.npz",allow_pickle=False)) for split in ("c07","c08")})
    for split in ("c07","c08"):
        reference=arrays[0][split]
        for seed in (1,2):
            for field in ("global_sequence_index","event_name","identity"):
                if not np.array_equal(reference[field],arrays[seed][split][field]):raise RuntimeError("three-seed prediction population drift")
    truth={split:_labels(arrays[0][split]["event_name"]) for split in ("c07","c08")};identities={split:arrays[0][split]["identity"] for split in ("c07","c08")}
    ensemble={};curves={}
    for name in (*VARIANTS,"no_refusal"):
        values={split:_variant(arrays,summaries,name,split) for split in ("c07","c08")};threshold,curve,error=(0.0,[],None) if name=="no_refusal" else _select(values["c07"][0],values["c07"][1],values["c07"][2],truth["c07"]);curves[name]=curve
        ensemble[name]={split:_metrics(truth[split],values[split][1],values[split][2],threshold,identities[split]) for split in ("c07","c08")};ensemble[name]["selection"]={"threshold":threshold,"error":error,"curve_points":len(curve)}
    baseline_values={};c07_labels=truth["c07"]
    for split in ("c07","c08"):
        probabilities=[]
        for seed in range(3):
            temp=float(fit_event_temperature(arrays[seed]["c07"]["baseline_logits"],c07_labels)["temperature"]);probabilities.append(_softmax(arrays[seed][split]["baseline_logits"],temp))
        probability=np.mean(probabilities,axis=0);prediction=probability.argmax(1);entropy=-(probability*np.log(np.maximum(probability,1e-12))).sum(1)/math.log(5);score=probability[np.arange(len(prediction)),prediction]*(1-entropy);score[prediction==0]=0;baseline_values[split]=(prediction,score)
    threshold,curve,error=_select(None,*baseline_values["c07"],truth["c07"]);curves["baseline"]=curve
    ensemble["baseline"]={split:_metrics(truth[split],baseline_values[split][0],baseline_values[split][1],threshold,identities[split]) for split in ("c07","c08")};ensemble["baseline"]["selection"]={"threshold":threshold,"error":error,"curve_points":len(curve)}
    per_seed={str(seed):summaries[seed]["metrics"] for seed in range(3)}
    mean_gain=float(np.mean([summaries[s]["metrics"]["full"]["c08"]["raw"]["macro_f1"]-summaries[s]["baseline_corrected_teacher"]["c08"]["macro_f1"] for s in range(3)]))
    full=ensemble["full"]["c08"];base=ensemble["baseline"]["c08"]
    checks={
        "mean_seed_macro_f1_gain_at_least_0p05":mean_gain>=.05,
        "ensemble_macro_f1_gain_at_least_0p05":full["raw"]["macro_f1"]>=base["raw"]["macro_f1"]+.05,
        "turn_frame_f1_improves":full["raw"]["per_class"]["turn"]["f1"]>base["raw"]["per_class"]["turn"]["f1"],
        "transition_frame_f1_improves":full["raw"]["per_class"]["geometry_transition"]["f1"]>base["raw"]["per_class"]["geometry_transition"]["f1"],
        "turn_identity_coverage_improves":full["identity_coverage"]["turn"]["coverage"]>base["identity_coverage"]["turn"]["coverage"],
        "transition_identity_coverage_improves":full["identity_coverage"]["geometry_transition"]["coverage"]>base["identity_coverage"]["geometry_transition"]["coverage"],
        "c07_commit_safety":ensemble["full"]["c07"]["selective"]["precision"]>=.98 and ensemble["full"]["c07"]["selective"]["false_accept_rate"]<=.01 and ensemble["full"]["c07"]["selective"]["recall"]>=.25,
        "c08_commit_safety":full["selective"]["precision"]>=.98 and full["selective"]["false_accept_rate"]<=.01 and full["selective"]["recall"]>=.25,
        "metric_ablation_degrades_all_seeds":all(summaries[s]["metrics"]["full"]["c08"]["raw"]["per_class"]["turn"]["f1"]>summaries[s]["metrics"]["no_metric"]["c08"]["raw"]["per_class"]["turn"]["f1"] and summaries[s]["metrics"]["full"]["c08"]["raw"]["per_class"]["geometry_transition"]["f1"]>summaries[s]["metrics"]["no_metric"]["c08"]["raw"]["per_class"]["geometry_transition"]["f1"] for s in range(3)),
        "transport_ablation_degrades_all_seeds":all(np.mean([summaries[s]["metrics"]["full"]["c08"]["raw"]["per_class"][e]["f1"] for e in ("junction","terminal")])>np.mean([summaries[s]["metrics"]["no_transport"]["c08"]["raw"]["per_class"][e]["f1"] for e in ("junction","terminal")]) for s in range(3)),
        "refusal_improves_safety_all_seeds":all(summaries[s]["metrics"]["full"]["c08"]["selective"]["false_accept_rate"]<summaries[s]["metrics"]["no_refusal"]["c08"]["selective"]["false_accept_rate"] or summaries[s]["metrics"]["full"]["c08"]["selective"]["precision"]>summaries[s]["metrics"]["no_refusal"]["c08"]["selective"]["precision"] for s in range(3)),
        "zero_test_graph":all(summaries[s].get("c09_worlds_read")==0 and summaries[s].get("c10_worlds_read")==0 and summaries[s].get("mtare_worlds_read")==0 and summaries[s].get("graph_replays")==0 for s in range(3)),
    }
    scientific_pass=all(checks.values());summary={"schema_version":"gse_dual_composer_three_seed_evaluation_v1","status":PASS if scientific_pass else FAIL,"scientific_pass":scientific_pass,"mean_seed_c08_macro_f1_gain":mean_gain,"checks":checks,"ensemble":ensemble,"per_seed":per_seed,"optimizer_steps":sum(s["optimizer_steps"] for s in summaries),"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0}
    for name,curve in curves.items():write_json(output/f"{name}_ensemble_threshold_curve.json",curve)
    write_json(output/"summary.json",summary);write_json(output/"figure_source.json",summary);_plot(output,summary);print(json.dumps({"status":summary["status"],"mean_gain":mean_gain,"checks":checks},indent=2));return 0 if scientific_pass else 2


if __name__=="__main__":raise SystemExit(main())
