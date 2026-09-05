#!/usr/bin/env python3
"""Train one frozen-input GSE dual-Composer seed and export C07/C08 evidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mtare_topo.data.gse_explicit_composer_cache import validate_explicit_composer_world_cache
from mtare_topo.evaluation.gse_validation_calibration import (
    fit_event_temperature,
    precision_constrained_threshold,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_typed_composers import (
    ActionSetComposerInput,
    ActionSetRelationComposer,
    MetricChangeComposer,
    MetricChangeComposerInput,
)
from mtare_topo.semantics.gse_composer_learning import (
    EVENT_NAMES,
    fractional_backprojection_distribution,
    fuse_factorized_event_logits,
    identity_coverage,
    macro_f1,
    neutralize_transport,
    single_frame_action,
    single_frame_metric,
    structural_acceptance_metrics,
)


VARIANTS=("full","no_metric","no_transport","single_frame","count_bearing_only")
EXPECTED_ROWS={"fit":142184,"c07":21548,"c08":24394}


@dataclass(frozen=True)
class Bundle:
    global_index:np.ndarray;parent:np.ndarray;event:np.ndarray;identity:np.ndarray
    action_target:np.ndarray;metric_target:np.ndarray;mask:np.ndarray
    back_steps:np.ndarray;back_valid:np.ndarray
    bearing:np.ndarray;existence:np.ndarray;opening:np.ndarray;profile:np.ndarray;token_uncertainty:np.ndarray
    count:np.ndarray;transport:np.ndarray;reveal:np.ndarray;geometry:np.ndarray;geometry_uncertainty:np.ndarray

    def __len__(self):return len(self.global_index)


def _split(parent:str)->str:
    c=int(parent.rsplit("_C",1)[1]);return "fit" if c<=6 else "c07" if c==7 else "c08"


def _concat(records:list[Bundle])->Bundle:
    return Bundle(**{name:np.concatenate([getattr(row,name) for row in records],axis=0) for name in Bundle.__dataclass_fields__})


def load_bundles(cache_root:Path,supervision_root:Path)->dict[str,Bundle]:
    grouped={name:[] for name in EXPECTED_ROWS}
    paths=sorted(cache_root.glob("*.npz"))
    if len(paths)!=80:raise RuntimeError("dual-Composer cache must contain 80 worlds")
    for cache_path in paths:
        parent=cache_path.stem;supervision_path=supervision_root/f"{parent}.npz"
        if not supervision_path.is_file():raise RuntimeError(f"missing supervision {parent}")
        with np.load(cache_path,allow_pickle=False) as source:
            arrays={name:np.asarray(source[name]) for name in source.files};validate_explicit_composer_world_cache(arrays)
        with np.load(supervision_path,allow_pickle=False) as source:
            supervision={name:np.asarray(source[name]) for name in source.files}
        index=arrays["global_sequence_index"]
        if not np.array_equal(index,supervision["global_sequence_index"]):raise RuntimeError("cache/supervision identity drift")
        history=supervision["history_row_index"].astype(np.int64);safe=np.maximum(history,0)
        geometry=arrays["geometry"][safe].astype(np.float32)
        uncertainty=np.repeat(arrays["observation_uncertainty"][safe,None].astype(np.float32),4,axis=-1)
        record=Bundle(
            global_index=index,parent=np.full(len(index),parent,dtype="<U64"),event=supervision["event_name"],identity=supervision["identity"],
            action_target=supervision["action_target"],metric_target=supervision["metric_target"],mask=supervision["valid_history_mask"],
            back_steps=supervision["backprojection_steps_ago"],back_valid=supervision["backprojection_valid"],
            bearing=arrays["token_bearing_deg"],existence=arrays["token_existence_logits"],opening=arrays["token_opening_width_m"],
            profile=arrays["token_vertical_profile_m"],token_uncertainty=arrays["token_geometry_uncertainty"],count=arrays["token_count_probability"],
            transport=arrays["transport_row_probability"],reveal=arrays["transport_reveal_probability"],geometry=geometry,geometry_uncertainty=uncertainty,
        )
        grouped[_split(parent)].append(record)
    output={name:_concat(rows) for name,rows in grouped.items()}
    if {name:len(value) for name,value in output.items()}!=EXPECTED_ROWS:raise RuntimeError("dual-Composer split row drift")
    return output


def _indices(index:np.ndarray|list[int])->np.ndarray:return np.asarray(index,dtype=np.int64)


def action_input(bundle:Bundle,index:np.ndarray|list[int],device:torch.device)->ActionSetComposerInput:
    take=_indices(index);bearing=torch.as_tensor(bundle.bearing[take],device=device,dtype=torch.float32);radian=torch.deg2rad(bearing)
    return ActionSetComposerInput(
        token_bearing_unit=torch.stack((torch.sin(radian),torch.cos(radian)),dim=-1),
        token_existence_probability=torch.sigmoid(torch.as_tensor(bundle.existence[take],device=device,dtype=torch.float32)),
        token_opening_width_m=torch.as_tensor(bundle.opening[take],device=device,dtype=torch.float32),
        token_vertical_profile_m=torch.as_tensor(bundle.profile[take],device=device,dtype=torch.float32),
        token_geometry_uncertainty=torch.as_tensor(bundle.token_uncertainty[take],device=device,dtype=torch.float32),
        token_count_probability=torch.as_tensor(bundle.count[take],device=device,dtype=torch.float32),
        transport_row_probability=torch.as_tensor(bundle.transport[take],device=device,dtype=torch.float32),
        transport_reveal_probability=torch.as_tensor(bundle.reveal[take],device=device,dtype=torch.float32),
        valid_history_mask=torch.as_tensor(bundle.mask[take],device=device,dtype=torch.bool),
    )


def metric_input(bundle:Bundle,index:np.ndarray|list[int],device:torch.device)->MetricChangeComposerInput:
    take=_indices(index)
    return MetricChangeComposerInput(
        geometry_sequence=torch.as_tensor(bundle.geometry[take],device=device,dtype=torch.float32),
        geometry_uncertainty=torch.as_tensor(bundle.geometry_uncertainty[take],device=device,dtype=torch.float32),
        valid_history_mask=torch.as_tensor(bundle.mask[take],device=device,dtype=torch.bool),
    )


def _class_weight(target:np.ndarray,classes:int,device:torch.device)->torch.Tensor:
    count=np.bincount(target.astype(np.int64),minlength=classes).astype(np.float64)
    if np.any(count==0):raise RuntimeError("training split lacks a Composer class")
    weight=len(target)/(classes*count);return torch.as_tensor(weight,device=device,dtype=torch.float32)


def _balanced_nll(logits:np.ndarray,target:np.ndarray)->float:
    shifted=logits-logits.max(1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True));loss=-logp[np.arange(len(target)),target]
    return float(np.mean([loss[target==index].mean() for index in range(logits.shape[1])]))


def _neutral_geometry(inputs:ActionSetComposerInput,stats:dict[str,np.ndarray])->ActionSetComposerInput:
    opening=torch.as_tensor(stats["opening"],device=inputs.token_opening_width_m.device,dtype=inputs.token_opening_width_m.dtype).view(1,1,1).expand_as(inputs.token_opening_width_m)
    profile=torch.as_tensor(stats["profile"],device=inputs.token_vertical_profile_m.device,dtype=inputs.token_vertical_profile_m.dtype).view(1,1,1,4).expand_as(inputs.token_vertical_profile_m)
    uncertainty=torch.as_tensor(stats["uncertainty"],device=inputs.token_geometry_uncertainty.device,dtype=inputs.token_geometry_uncertainty.dtype).view(1,1,1,5).expand_as(inputs.token_geometry_uncertainty)
    return replace(neutralize_transport(inputs),token_opening_width_m=opening,token_vertical_profile_m=profile,token_geometry_uncertainty=uncertainty)


def _geometry_stats(bundle:Bundle)->dict[str,np.ndarray]:
    weight=1.0/(1.0+np.exp(-bundle.existence.astype(np.float64)));normalizer=weight.sum()
    return {
        "opening":np.asarray((bundle.opening*weight).sum()/normalizer,dtype=np.float32),
        "profile":np.asarray((bundle.profile*weight[...,None]).sum(axis=(0,1,2))/normalizer,dtype=np.float32),
        "uncertainty":np.asarray((bundle.token_uncertainty*weight[...,None]).sum(axis=(0,1,2))/normalizer,dtype=np.float32),
    }


@torch.no_grad()
def infer(action:ActionSetRelationComposer,metric:MetricChangeComposer,bundle:Bundle,device:torch.device,batch_size:int,stats:dict[str,np.ndarray],variants=VARIANTS)->dict[str,dict[str,np.ndarray]]:
    action.eval();metric.eval();parts={name:{"logits":[],"action_commit":[],"metric_commit":[],"steps":[],"back_probability":[]} for name in variants}
    for start in range(0,len(bundle),batch_size):
        index=np.arange(start,min(start+batch_size,len(bundle)));base_action=action_input(bundle,index,device);base_metric=metric_input(bundle,index,device)
        full_action=action(base_action);full_metric=metric(base_metric)
        outputs={"full":(full_action,full_metric)}
        if "no_metric" in variants:outputs["no_metric"]=(full_action,None)
        if "no_transport" in variants:outputs["no_transport"]=(action(neutralize_transport(base_action)),full_metric)
        if "single_frame" in variants:outputs["single_frame"]=(action(single_frame_action(base_action)),metric(single_frame_metric(base_metric)))
        if "count_bearing_only" in variants:outputs["count_bearing_only"]=(action(_neutral_geometry(base_action,stats)),None)
        for name,(left,right) in outputs.items():
            if right is None:
                metric_logits=left.event_logits.new_full(left.event_logits.shape,-30.0);metric_logits[:,0]=0.0
                metric_commit=torch.ones_like(left.commit_probability);steps=torch.zeros_like(left.commit_probability);back_probability=left.event_logits.new_zeros((len(left.event_logits),5));back_probability[:,-1]=1.0
            else:metric_logits=right.event_logits;metric_commit=right.commit_probability;steps=right.expected_steps_ago;back_probability=right.backprojection_probability
            parts[name]["logits"].append(fuse_factorized_event_logits(left.event_logits,metric_logits).cpu().numpy())
            parts[name]["action_commit"].append(left.commit_probability.cpu().numpy());parts[name]["metric_commit"].append(metric_commit.cpu().numpy());parts[name]["steps"].append(steps.cpu().numpy())
            parts[name]["back_probability"].append(back_probability.cpu().numpy())
    return {name:{key:np.concatenate(value) for key,value in fields.items()} for name,fields in parts.items()}


def _event_labels(event:np.ndarray)->np.ndarray:
    lookup={name:index for index,name in enumerate(EVENT_NAMES)};return np.asarray([lookup[str(value)] for value in event],dtype=np.int64)


def _backprojection_target(steps:np.ndarray,valid:np.ndarray)->np.ndarray:
    support=np.asarray((4.0,3.0,2.0,1.0,0.0),dtype=np.float64);target=np.maximum(0.0,1.0-np.abs(steps[:,None]-support[None]));target*=valid[:,None]
    if valid.any() and not np.allclose(target[valid].sum(1),1.0,atol=1e-6,rtol=0.0):raise RuntimeError("numpy backprojection target lost mass")
    return target


def _probability(logits:np.ndarray,temperature:float)->np.ndarray:
    value=logits/temperature;value=value-value.max(1,keepdims=True);value=np.exp(value);return value/value.sum(1,keepdims=True)


def _reliability(probability:np.ndarray,prediction:np.ndarray,output:dict[str,np.ndarray])->np.ndarray:
    score=probability[np.arange(len(prediction)),prediction];commit=np.ones(len(prediction),dtype=np.float64)
    action=(prediction==1)|(prediction==2);metric=(prediction==3)|(prediction==4)
    commit[action]=output["action_commit"][action];commit[metric]=output["metric_commit"][metric];score[prediction==0]=0.0
    return score*commit


def _select(output:dict[str,np.ndarray],bundle:Bundle)->dict[str,object]:
    truth=_event_labels(bundle.event);calibration=fit_event_temperature(output["logits"],truth);probability=_probability(output["logits"],float(calibration["temperature"]));prediction=probability.argmax(1);score=_reliability(probability,prediction,output)
    correct=(prediction==truth)&(truth!=0);eligible=truth!=0
    try:selected,curve=precision_constrained_threshold(score,correct,eligible,minimum_precision=.98,maximum_false_accept_rate=.01);threshold=selected.threshold;error=None
    except RuntimeError as exc:threshold=2.0;curve=[];error=str(exc)
    return {"temperature":calibration,"threshold":threshold,"threshold_error":error,"curve":curve,"raw":macro_f1(truth,prediction),"selective":structural_acceptance_metrics(truth,prediction,score,threshold),"identity_coverage":identity_coverage(truth,prediction,score,threshold,bundle.identity)}


def _apply(output:dict[str,np.ndarray],bundle:Bundle,selection:dict[str,object])->dict[str,object]:
    truth=_event_labels(bundle.event);probability=_probability(output["logits"],float(selection["temperature"]["temperature"]));prediction=probability.argmax(1);score=_reliability(probability,prediction,output);threshold=float(selection["threshold"])
    valid=bundle.back_valid;back_mae=float(np.abs(output["steps"][valid]-bundle.back_steps[valid]).mean()) if valid.any() else None
    return {"raw":macro_f1(truth,prediction),"selective":structural_acceptance_metrics(truth,prediction,score,threshold),"identity_coverage":identity_coverage(truth,prediction,score,threshold,bundle.identity),"backprojection_mae_frames":back_mae,"prediction":prediction,"reliability":score,"probability":probability}


def _baseline(root:Path,bundle:Bundle)->np.ndarray:
    by_index={}
    suffix="_C07.npz" if str(bundle.parent[0]).endswith("_C07") else "_C08.npz"
    for path in sorted(root.glob(f"*{suffix}")):
        with np.load(path,allow_pickle=False) as source:
            for row,index in enumerate(source["global_sequence_index"].tolist()):by_index[int(index)]=np.asarray(source["event_logits"][row],dtype=np.float32)
    if set(bundle.global_index.tolist())-set(by_index):raise RuntimeError("baseline corrected-Teacher alignment missing rows")
    return np.stack([by_index[int(index)] for index in bundle.global_index])


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--cache-root",type=Path,required=True);parser.add_argument("--supervision-root",type=Path,required=True);parser.add_argument("--baseline-root",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--seed",type=int,required=True,choices=(0,1,2));parser.add_argument("--epochs",type=int,default=10);parser.add_argument("--refusal-epochs",type=int,default=2);parser.add_argument("--batch-size",type=int,default=1024);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();started=time.monotonic();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False);(output/"predictions").mkdir()
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.use_deterministic_algorithms(True);device=torch.device(args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu")
    bundles=load_bundles(args.cache_root.resolve(),args.supervision_root.resolve());fit=bundles["fit"];c07=bundles["c07"];c08=bundles["c08"];stats=_geometry_stats(fit)
    action=ActionSetRelationComposer().to(device);metric=MetricChangeComposer().to(device)
    for parameter in list(action.refusal_head.parameters())+list(metric.refusal_head.parameters()):parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in list(action.parameters())+list(metric.parameters()) if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    action_weight=_class_weight(fit.action_target,3,device);metric_weight=_class_weight(fit.metric_target,3,device);history=[];best=None
    generator=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        action.train();metric.train();order=generator.permutation(len(fit));totals=[]
        for start in range(0,len(fit),args.batch_size):
            index=order[start:start+args.batch_size];ai=action_input(fit,index,device);mi=metric_input(fit,index,device);ao=action(ai);mo=metric(mi)
            at=torch.as_tensor(fit.action_target[index],device=device,dtype=torch.long);mt=torch.as_tensor(fit.metric_target[index],device=device,dtype=torch.long)
            steps=torch.as_tensor(fit.back_steps[index],device=device,dtype=torch.float32);valid=torch.as_tensor(fit.back_valid[index],device=device,dtype=torch.bool);soft=fractional_backprojection_distribution(steps,valid)
            event=F.cross_entropy(ao.event_logits,at,weight=action_weight)+F.cross_entropy(mo.event_logits,mt,weight=metric_weight)
            per_row=-(soft*torch.log_softmax(mo.backprojection_logits,dim=-1)).sum(-1);back=per_row[valid].mean() if bool(valid.any()) else per_row.sum()*0.0;loss=event+back
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in list(action.parameters())+list(metric.parameters()) if p.requires_grad],5.0);optimizer.step();totals.append(float(loss.detach()))
        validation=infer(action,metric,c07,device,args.batch_size,stats,variants=("full",))["full"];truth=_event_labels(c07.event);score=_balanced_nll(validation["logits"],truth)
        valid=c07.back_valid
        if valid.any():
            target=_backprojection_target(c07.back_steps,c07.back_valid);score+=float(-(target[valid]*np.log(np.maximum(validation["back_probability"][valid],1e-9))).sum(1).mean())
        record={"epoch":epoch,"train_loss":float(np.mean(totals)),"selection_score":score,"c07_macro_f1":macro_f1(truth,validation["logits"].argmax(1))["macro_f1"]};history.append(record)
        if best is None or score<best[0]-1e-12:best=(score,epoch,{k:v.detach().cpu().clone() for k,v in action.state_dict().items()},{k:v.detach().cpu().clone() for k,v in metric.state_dict().items()})
        print(json.dumps(record),flush=True)
    assert best is not None;action.load_state_dict(best[2]);metric.load_state_dict(best[3])
    for parameter in list(action.parameters())+list(metric.parameters()):parameter.requires_grad_(False)
    for parameter in list(action.refusal_head.parameters())+list(metric.refusal_head.parameters()):parameter.requires_grad_(True)
    refusal_optimizer=torch.optim.AdamW(list(action.refusal_head.parameters())+list(metric.refusal_head.parameters()),lr=3e-4,weight_decay=1e-4)
    error_count=[0,0]
    with torch.no_grad():
        for start in range(0,len(fit),args.batch_size):
            index=np.arange(start,min(start+args.batch_size,len(fit)));ao=action(action_input(fit,index,device));mo=metric(metric_input(fit,index,device));error_count[0]+=int((ao.event_logits.argmax(1).cpu().numpy()!=fit.action_target[index]).sum());error_count[1]+=int((mo.event_logits.argmax(1).cpu().numpy()!=fit.metric_target[index]).sum())
    refusal_positive_weight=[(len(fit)-value)/value if value else 1.0 for value in error_count]
    for refusal_epoch in range(args.refusal_epochs):
        order=generator.permutation(len(fit));losses=[];action.train();metric.train()
        for start in range(0,len(fit),args.batch_size):
            index=order[start:start+args.batch_size];ao=action(action_input(fit,index,device));mo=metric(metric_input(fit,index,device));at=torch.as_tensor(fit.action_target[index],device=device);mt=torch.as_tensor(fit.metric_target[index],device=device)
            ar=(ao.event_logits.detach().argmax(1)!=at).float();mr=(mo.event_logits.detach().argmax(1)!=mt).float()
            al=F.binary_cross_entropy_with_logits(ao.refusal_logit,ar,pos_weight=ao.refusal_logit.new_tensor(refusal_positive_weight[0]));ml=F.binary_cross_entropy_with_logits(mo.refusal_logit,mr,pos_weight=mo.refusal_logit.new_tensor(refusal_positive_weight[1]));loss=al+ml
            refusal_optimizer.zero_grad(set_to_none=True);loss.backward();refusal_optimizer.step();losses.append(float(loss.detach()))
        print(json.dumps({"refusal_epoch":refusal_epoch,"loss":float(np.mean(losses))}),flush=True)
    predictions={split:infer(action,metric,bundle,device,args.batch_size,stats) for split,bundle in (("c07",c07),("c08",c08))}
    selections={name:_select(predictions["c07"][name],c07) for name in VARIANTS};selections["no_refusal"]={**selections["full"],"threshold":0.0,"threshold_error":None}
    (output/"calibration").mkdir()
    compact_selection={}
    for name,value in selections.items():
        curve=value.get("curve",[]);write_json(output/"calibration"/f"{name}_threshold_curve.json",curve)
        compact_selection[name]={key:item for key,item in value.items() if key!="curve"};compact_selection[name]["curve_points"]=len(curve)
    metrics={}
    for name in (*VARIANTS,"no_refusal"):
        source="full" if name=="no_refusal" else name;metrics[name]={split:_apply(predictions[split][source],bundle,selections[name]) for split,bundle in (("c07",c07),("c08",c08))}
        for split in ("c07","c08"):
            for key in ("prediction","reliability","probability"):metrics[name][split].pop(key)
    baseline={};baseline_logits={}
    for split,bundle in (("c07",c07),("c08",c08)):
        logits=_baseline(args.baseline_root.resolve(),bundle);baseline_logits[split]=logits;baseline[split]=macro_f1(_event_labels(bundle.event),logits.argmax(1))
    checkpoint={"seed":args.seed,"selected_epoch":best[1],"action_state_dict":action.state_dict(),"metric_state_dict":metric.state_dict(),"selection":compact_selection,"geometry_neutral_stats":stats}
    torch.save(checkpoint,output/"best.pt");write_json(output/"history.json",history)
    for split,bundle in (("c07",c07),("c08",c08)):
        fields={"global_sequence_index":bundle.global_index,"event_name":bundle.event,"identity":bundle.identity}
        fields["baseline_logits"]=baseline_logits[split].astype(np.float32)
        for name in VARIANTS:
            for key,value in predictions[split][name].items():fields[f"{name}_{key}"]=value.astype(np.float32)
        np.savez_compressed(output/"predictions"/f"{split}.npz",**fields)
    summary={"schema_version":"gse_dual_composer_seed_training_v1","seed":args.seed,"device":str(device),"epochs":args.epochs,"refusal_epochs":args.refusal_epochs,"batch_size":args.batch_size,"optimizer_steps":args.epochs*math.ceil(len(fit)/args.batch_size)+args.refusal_epochs*math.ceil(len(fit)/args.batch_size),"selected_epoch":best[1],"selection_score":best[0],"refusal_error_count":error_count,"refusal_positive_weight":refusal_positive_weight,"metrics":metrics,"baseline_corrected_teacher":baseline,"selection":compact_selection,"rows":EXPECTED_ROWS,"duration_seconds":time.monotonic()-started,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0}
    write_json(output/"summary.json",summary);print(json.dumps({"seed":args.seed,"selected_epoch":best[1],"c08_macro_f1":metrics["full"]["c08"]["raw"]["macro_f1"],"baseline":baseline["c08"]["macro_f1"],"duration_seconds":summary["duration_seconds"]},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
