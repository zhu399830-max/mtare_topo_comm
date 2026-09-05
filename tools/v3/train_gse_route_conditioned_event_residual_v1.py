#!/usr/bin/env python3
"""Train one frozen-base route-conditioned event residual seed."""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_endpoint_identity_residual import identity_balanced_endpoint_mil_loss
from mtare_topo.representation.gse_route_conditioned_event_residual import RouteConditionedEventResidual,route_flow_features
from train_gse_endpoint_identity_residual_v1 import _endpoint_metrics,_extract_base,_partition_contract,_read_jsonl
from mtare_topo.evaluation.gse_causal_episode_metrics import evaluate_decision_mass_triggers

EXPECTED=188_126
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def _probability(structural,conditional,structural_residual,conditional_residual):
 logit=np.asarray(structural,dtype=np.float64)+np.asarray(structural_residual,dtype=np.float64);mass=1/(1+np.exp(-np.clip(logit,-80,80)));shift=np.asarray(conditional,dtype=np.float64)+np.asarray(conditional_residual,dtype=np.float64);shift-=shift.max(1,keepdims=True);classes=np.exp(shift);classes/=classes.sum(1,keepdims=True);action=np.concatenate(((1-mass)[:,None],mass[:,None]*classes),1);probability=np.zeros((len(action),5),dtype=np.float64);probability[:,:3]=action;uncertainty=-(probability*np.log(np.clip(probability,1e-8,1))).sum(1)/np.log(5)
 if not np.allclose(probability.sum(1),1,rtol=0,atol=1e-5):raise RuntimeError("route event probability drift")
 return probability.astype(np.float32),uncertainty.astype(np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cache-dir",required=True,type=Path);ap.add_argument("--endpoint-audit",required=True,type=Path);ap.add_argument("--base-checkpoint",required=True,type=Path);ap.add_argument("--output-dir",required=True,type=Path);ap.add_argument("--seed",required=True,type=int,choices=(0,1,2));ap.add_argument("--steps",type=int,default=200);ap.add_argument("--evaluation-interval",type=int,default=10);ap.add_argument("--learning-rate",type=float,default=3e-4);ap.add_argument("--weight-decay",type=float,default=1e-4);a=ap.parse_args();started=time.monotonic()
 if a.output_dir.exists():raise RuntimeError("route event seed output exists; overwrite forbidden")
 a.output_dir.mkdir(parents=True);import torch
 if not torch.cuda.is_available():raise RuntimeError("route event training requires CUDA")
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;device=torch.device("cuda");cache=a.cache_dir.resolve();manifest=json.loads((cache/"manifest.json").read_text())
 if manifest.get("causal_observations")!=EXPECTED:raise RuntimeError("route event cache drift")
 partition=np.load(cache/"partition_code.npy").astype(np.uint8);target=np.load(cache/"decision_target.npy").astype(np.int64);episode=np.load(cache/"decision_episode_id.npy").astype(np.int64);identity=np.load(cache/"identity.npy").astype(str);traversal=np.load(cache/"traversal_id.npy").astype(str);sequence=np.load(cache/"sequence_index.npy").astype(np.int64);global_index=np.load(cache/"global_sequence_index.npy").astype(np.int64);references=np.load(cache/"history_references.npy").astype(np.int64);mask=np.load(cache/"history_mask.npy").astype(bool);raw=np.load(cache/"raw_tokens.npy",mmap_mode="r");event=np.asarray(["corridor" if x==0 else "junction" if x==1 else "terminal" for x in target])
 records=_read_jsonl(a.endpoint_audit.resolve());fit_records=[x for x in records if x["partition"]=="fit"];selection_records=[x for x in records if x["partition"]=="selection"]
 if len(fit_records)!=72 or len(selection_records)!=26:raise RuntimeError("route event endpoint drift")
 flow=route_flow_features(raw,references,mask);fit_rows=np.flatnonzero(partition==0);mean=flow[fit_rows].mean(0,dtype=np.float64).astype(np.float32);scale=flow[fit_rows].std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1;flow=(flow-mean)/scale
 base=_extract_base(cache,a.base_checkpoint.resolve(),a.seed,device);fit=_partition_contract(code=0,partition=partition,target=target,episode=episode,identity=identity,endpoint_names=[x["identity"] for x in fit_records],device=device);selection=_partition_contract(code=1,partition=partition,target=target,episode=episode,identity=identity,endpoint_names=[x["identity"] for x in selection_records],device=device)
 if int(torch.sum(fit["endpoint_identity"]>=0))!=286 or int(torch.sum(selection["endpoint_identity"]>=0))!=110:raise RuntimeError("route event episode exposure drift")
 selection_rows=selection["rows"];fit_flow=torch.from_numpy(flow[fit_rows]).to(device);fit_structural=torch.from_numpy(base["structural"][fit_rows]).to(device);fit_conditional=torch.from_numpy(base["conditional"][fit_rows]).to(device);model=RouteConditionedEventResidual().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);history=[]
 def evaluate(step):
  model.eval();struct=[];conditional=[]
  with torch.inference_mode():
   for start in range(0,EXPECTED,8192):
    value=model(torch.from_numpy(flow[start:start+8192]).to(device));struct.append(value["structural_residual"].cpu().numpy());conditional.append(value["conditional_residual"].cpu().numpy())
  probability,uncertainty=_probability(base["structural"],base["conditional"],np.concatenate(struct),np.concatenate(conditional));decision=evaluate_decision_mass_triggers(probability[selection_rows],selection["target_np"],selection["episode_np"],traversal[selection_rows],sequence[selection_rows],uncertainty[selection_rows],decision_threshold=.97);endpoint=_endpoint_metrics(probability,uncertainty,selection_rows,traversal,sequence,identity,event,selection_records);record={"step":step,"decision":decision,"endpoint":endpoint};history.append(record);return record,probability,uncertainty
 baseline,_,_=evaluate(0);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=0;best_key=None
 for step in range(1,a.steps+1):
  model.train();optimizer.zero_grad(set_to_none=True);residual=model(fit_flow);loss=identity_balanced_endpoint_mil_loss(structural_logit=fit_structural+residual["structural_residual"],conditional_decision_logits=fit_conditional+residual["conditional_residual"],decision_target=fit["target"],episode_id=fit["episode"],endpoint_identity_by_episode=fit["endpoint_identity"]);loss["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
  if step%a.evaluation_interval==0 or step==a.steps:
   record,_,_=evaluate(step);decision=record["decision"];endpoint=record["endpoint"];false=decision["predicted_decision_triggers"]-decision["correctly_classified_unique_decision_episodes"];base_false=baseline["decision"]["predicted_decision_triggers"]-baseline["decision"]["correctly_classified_unique_decision_episodes"];safe=false<=base_false and decision["correctly_classified_unique_decision_episodes"]>=baseline["decision"]["correctly_classified_unique_decision_episodes"] and endpoint["high_correct"]>=baseline["endpoint"]["high_correct"] and endpoint["all_correct"]>=baseline["endpoint"]["all_correct"];key=(endpoint["low_correct"],endpoint["all_correct"],decision["correctly_classified_unique_decision_episodes"],-false,-float(loss["total"].detach().cpu()),-step);record["safe_candidate"]=safe;record["fit_loss"]={k:float(v.detach().cpu()) for k,v in loss.items()}
   if safe and (best_key is None or key>best_key):best_key=key;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   print(json.dumps(record,sort_keys=True),flush=True)
 model.load_state_dict(best_state);_,probability,uncertainty=evaluate(best_step);selected=next(x for x in history if x["step"]==best_step);torch.save({"schema_version":"gse_route_conditioned_event_residual_checkpoint_v1","seed":a.seed,"step":best_step,"model":model.state_dict(),"feature_mean":mean,"feature_scale":scale,"base_checkpoint_sha256":_sha(a.base_checkpoint.resolve()),"selection_baseline":baseline,"selection_selected":selected},a.output_dir/"best.pt");np.savez_compressed(a.output_dir/"all_outputs.npz",global_sequence_index=global_index,probability=probability,uncertainty=uncertainty)
 summary={"schema_version":"gse_route_conditioned_event_residual_seed_v1","seed":a.seed,"steps":a.steps,"best_step":best_step,"trainable_parameters":sum(p.numel() for p in model.parameters()),"base_model_optimizer_steps":0,"optimizer_steps":a.steps,"fit_observations":len(fit_rows),"selection_observations":len(selection_rows),"fit_endpoint_identities":72,"selection_endpoint_identities":26,"selection_baseline":baseline,"selection_selected":selected,"duration_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0};(a.output_dir/"history.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n");(a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
