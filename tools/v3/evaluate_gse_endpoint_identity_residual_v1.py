#!/usr/bin/env python3
"""Evaluate the three-seed identity residual ensemble at fixed threshold 0.97."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from mtare_topo.evaluation.gse_causal_episode_metrics import evaluate_decision_mass_triggers,extract_decision_mass_triggers

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def _read(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def _metrics(probability,uncertainty,rows,target,episode,traversal,sequence,identity,event,endpoints):
 decision=evaluate_decision_mass_triggers(probability[rows],target[rows],episode[rows],traversal[rows],sequence[rows],uncertainty[rows],decision_threshold=.97)
 correct=set()
 for t in extract_decision_mass_triggers(probability[rows],traversal[rows],sequence[rows],uncertainty[rows],decision_threshold=.97):
  row=int(rows[int(t.row)]); predicted={1:"junction",2:"terminal"}[int(t.predicted_event_index)]
  if event[row]==predicted and identity[row]:correct.add(str(identity[row]))
 low={x["identity"] for x in endpoints if x["support_bin"]=="1-3"};high={x["identity"] for x in endpoints if int(x["teacher_rows"])>=11};names={x["identity"] for x in endpoints}
 return {"decision":decision,"endpoint":{"low_correct":len(correct&low),"low_total":len(low),"high_correct":len(correct&high),"high_total":len(high),"all_correct":len(correct&names),"all_total":len(names)}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cache-dir",required=True,type=Path);ap.add_argument("--endpoint-audit",required=True,type=Path);ap.add_argument("--baseline-ensemble",required=True,type=Path);ap.add_argument("--seed0",required=True,type=Path);ap.add_argument("--seed1",required=True,type=Path);ap.add_argument("--seed2",required=True,type=Path);ap.add_argument("--output-dir",required=True,type=Path);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);cache=a.cache_dir.resolve()
 part=np.load(cache/"partition_code.npy");target=np.load(cache/"decision_target.npy").astype(np.int64);episode=np.load(cache/"decision_episode_id.npy").astype(np.int64);traversal=np.load(cache/"traversal_id.npy").astype(str);sequence=np.load(cache/"sequence_index.npy").astype(np.int64);identity=np.load(cache/"identity.npy").astype(str);global_index=np.load(cache/"global_sequence_index.npy").astype(np.int64);event=np.asarray(["corridor" if x==0 else "junction" if x==1 else "terminal" for x in target]);rows=np.flatnonzero(part==1);endpoints=[x for x in _read(a.endpoint_audit) if x["partition"]=="selection"]
 values=[]
 for path in (a.seed0,a.seed1,a.seed2):
  with np.load(path,allow_pickle=False) as z:
   if not np.array_equal(z["global_sequence_index"],global_index):raise RuntimeError("residual ensemble seed identity drift")
   values.append(z["probability"].astype(np.float64))
 probability=np.mean(np.stack(values),axis=0);uncertainty=-(probability*np.log(np.clip(probability,1e-8,1))).sum(1)/np.log(3.0)
 with np.load(a.baseline_ensemble.resolve(),allow_pickle=False) as z:
  if not np.array_equal(z["global_sequence_index"],global_index):raise RuntimeError("residual baseline identity drift")
  base_probability=z["probability"].astype(np.float64);base_uncertainty=z["uncertainty"].astype(np.float64)
 baseline=_metrics(base_probability,base_uncertainty,rows,target,episode,traversal,sequence,identity,event,endpoints);corrected=_metrics(probability,uncertainty,rows,target,episode,traversal,sequence,identity,event,endpoints)
 def false_count(x):return x["decision"]["predicted_decision_triggers"]-x["decision"]["correctly_classified_unique_decision_episodes"]
 gates={"low_support_recovers_at_least_one":corrected["endpoint"]["low_correct"]>=baseline["endpoint"]["low_correct"]+1,"high_support_no_regression":corrected["endpoint"]["high_correct"]>=baseline["endpoint"]["high_correct"],"all_endpoint_no_regression":corrected["endpoint"]["all_correct"]>=baseline["endpoint"]["all_correct"],"correct_episode_no_regression":corrected["decision"]["correctly_classified_unique_decision_episodes"]>=baseline["decision"]["correctly_classified_unique_decision_episodes"],"false_trigger_no_increase":false_count(corrected)<=false_count(baseline)};gates["all_passed"]=all(gates.values())
 np.savez_compressed(out/"corrected_action_ensemble.npz",global_sequence_index=global_index,probability=probability.astype(np.float32),uncertainty=uncertainty.astype(np.float32))
 summary={"schema_version":"gse_endpoint_identity_residual_ensemble_v1","status":"PASS_GSE_ENDPOINT_IDENTITY_RESIDUAL_ENSEMBLE_V1" if gates["all_passed"] else "FAIL_GSE_ENDPOINT_IDENTITY_RESIDUAL_ENSEMBLE_V1","baseline":baseline,"corrected":corrected,"gates":gates,"decision_threshold":.97,"threshold_selection_steps":0,"optimizer_steps":0,"model_updates":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"ensemble_sha256":_sha(out/"corrected_action_ensemble.npz")};(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 with (out/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(("metric","baseline","corrected"));w.writerow(("low_endpoint_correct",baseline["endpoint"]["low_correct"],corrected["endpoint"]["low_correct"]));w.writerow(("high_endpoint_correct",baseline["endpoint"]["high_correct"],corrected["endpoint"]["high_correct"]));w.writerow(("correct_episodes",baseline["decision"]["correctly_classified_unique_decision_episodes"],corrected["decision"]["correctly_classified_unique_decision_episodes"]));w.writerow(("false_triggers",false_count(baseline),false_count(corrected)))
 import matplotlib;matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(9.8,3.8),constrained_layout=True);x=np.arange(3);width=.36
 labels=("Low support","High support","All endpoints");before=(baseline["endpoint"]["low_correct"],baseline["endpoint"]["high_correct"],baseline["endpoint"]["all_correct"]);after=(corrected["endpoint"]["low_correct"],corrected["endpoint"]["high_correct"],corrected["endpoint"]["all_correct"])
 axes[0].bar(x-width/2,before,width,label="Frozen base",color="#9AA0A6");axes[0].bar(x+width/2,after,width,label="Identity residual",color="#2878B5");axes[0].set_xticks(x,labels);axes[0].set_ylabel("Correct endpoint identities");axes[0].set_title("A  Relation-endpoint proposals");axes[0].legend(frameon=False)
 labels2=("Correct episodes","False triggers");before2=(baseline["decision"]["correctly_classified_unique_decision_episodes"],false_count(baseline));after2=(corrected["decision"]["correctly_classified_unique_decision_episodes"],false_count(corrected));x2=np.arange(2);axes[1].bar(x2-width/2,before2,width,color="#9AA0A6");axes[1].bar(x2+width/2,after2,width,color="#2878B5");axes[1].set_xticks(x2,labels2);axes[1].set_ylabel("Count");axes[1].set_title("B  Global trigger safety and coverage")
 fig.suptitle("Identity-balanced structural residual at fixed deployment threshold 0.97")
 for suffix in ("png","pdf","svg"):fig.savefig(out/f"gse_endpoint_identity_residual.{suffix}",dpi=240 if suffix=="png" else None)
 plt.close(fig);summary["figure_sha256"]=_sha(out/"gse_endpoint_identity_residual.png");(out/"figure_source.json").write_text(json.dumps({"schema_version":"gse_endpoint_identity_residual_figure_source_v1","baseline":baseline,"corrected":corrected},indent=2,sort_keys=True)+"\n",encoding="utf-8");(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps(summary,sort_keys=True));return 0 if gates["all_passed"] else 2
if __name__=="__main__":raise SystemExit(main())
