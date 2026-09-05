#!/usr/bin/env python3
"""Audit a fixed physical route-conditioned exit-flow score without training."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np

def _read(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def route_flow(raw):
 values=np.asarray(raw,dtype=np.float64);confidence=np.clip(values[...,0],0,1);sine=values[...,1];cosine=values[...,2]
 forward=np.sum(confidence*np.clip(cosine,0,None),axis=2);backward=np.sum(confidence*np.clip(-cosine,0,None),axis=2);lateral=np.sum(confidence*np.abs(sine),axis=2);denominator=forward+backward+lateral
 score=(backward-forward)/np.maximum(denominator,1e-8)
 return {"forward":forward.mean(1),"backward":backward.mean(1),"lateral":lateral.mean(1),"route_stop_score":score.mean(1),"seed_score":score}
def auc(labels,scores):
 y=np.asarray(labels,dtype=np.int64);s=np.asarray(scores,dtype=np.float64);positive=s[y==1];negative=s[y==0]
 if not len(positive) or not len(negative):raise ValueError("route-flow AUC needs both classes")
 return float(sum(np.sum(value>negative)+.5*np.sum(value==negative) for value in positive)/(len(positive)*len(negative)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cache-dir",required=True,type=Path);ap.add_argument("--endpoint-audit",required=True,type=Path);ap.add_argument("--output-dir",required=True,type=Path);a=ap.parse_args();out=a.output_dir.resolve();started=time.monotonic()
 if out.exists():raise RuntimeError("route-flow audit output exists; overwrite forbidden")
 out.mkdir(parents=True);cache=a.cache_dir.resolve();raw=np.load(cache/"raw_tokens.npy",mmap_mode="r");identity=np.load(cache/"identity.npy").astype(str);partition=np.load(cache/"partition_code.npy");records=_read(a.endpoint_audit.resolve())
 if raw.shape!=(188126,3,6,40) or len(records)!=98:raise RuntimeError("route-flow population drift")
 flow={key:np.empty(188126,dtype=np.float64) for key in ("forward","backward","lateral","route_stop_score")};seed_score=np.empty((188126,3),dtype=np.float64)
 for start in range(0,len(raw),8192):
  value=route_flow(raw[start:start+8192]);stop=start+len(value["forward"])
  for key in flow:flow[key][start:stop]=value[key]
  seed_score[start:stop]=value["seed_score"]
 endpoint=[];split_summary={}
 for record in records:
  code=0 if record["partition"]=="fit" else 1;rows=np.flatnonzero((identity==record["identity"])&(partition==code));best=int(rows[np.argmax(flow["route_stop_score"][rows])])
  endpoint.append({**record,"maximum_route_stop_score":float(flow["route_stop_score"][best]),"best_row":best,"forward_mass":float(flow["forward"][best]),"backward_mass":float(flow["backward"][best]),"lateral_mass":float(flow["lateral"][best]),"seed_route_stop_score":seed_score[best].tolist()})
 for split in ("fit","selection"):
  rows=[x for x in endpoint if x["partition"]==split];labels=np.asarray([x["event"]=="terminal" for x in rows]);scores=np.asarray([x["maximum_route_stop_score"] for x in rows]);terminal=scores[labels];junction=scores[~labels]
  low=[x for x in rows if x["support_bin"]=="1-3"]
  split_summary[split]={"endpoint_identities":len(rows),"terminal_identities":int(labels.sum()),"junction_identities":int((~labels).sum()),"terminal_vs_junction_auc":auc(labels,scores),"terminal_mean":float(terminal.mean()),"junction_mean":float(junction.mean()),"terminal_median":float(np.median(terminal)),"junction_median":float(np.median(junction)),"low_support_scores":{x["identity"]:x["maximum_route_stop_score"] for x in low}}
 fit_median=split_summary["fit"]["junction_median"];selection_low_terminal=[x for x in endpoint if x["partition"]=="selection" and x["support_bin"]=="1-3" and x["event"]=="terminal"]
 recoverable=sum(x["maximum_route_stop_score"]>fit_median for x in selection_low_terminal)
 gates={"fit_auc_at_least_0p80":split_summary["fit"]["terminal_vs_junction_auc"]>=.8,"selection_auc_at_least_0p80":split_summary["selection"]["terminal_vs_junction_auc"]>=.8,"terminal_mean_above_junction_both_splits":all(split_summary[x]["terminal_mean"]>split_summary[x]["junction_mean"] for x in ("fit","selection")),"at_least_one_low_selection_terminal_above_fit_junction_median":recoverable>=1};gates["all_passed"]=all(gates.values())
 summary={"schema_version":"gse_route_conditioned_exit_flow_audit_v1","status":"PASS_GSE_ROUTE_CONDITIONED_EXIT_FLOW_AUDIT_V1" if gates["all_passed"] else "FAIL_GSE_ROUTE_CONDITIONED_EXIT_FLOW_AUDIT_V1","formula":"mean_seed((backward_confidence_mass-forward_confidence_mass)/(forward+backward+lateral)) then maximum over endpoint Teacher rows","split":split_summary,"low_selection_terminal_above_fit_junction_median":recoverable,"gates":gates,"observations":188126,"relation_endpoints":98,"optimizer_steps":0,"model_inference_frames":0,"model_updates":0,"threshold_selection_steps":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"duration_seconds":time.monotonic()-started}
 (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");
 with (out/"endpoint_flow.jsonl").open("w") as f:
  for x in endpoint:f.write(json.dumps(x,sort_keys=True)+"\n")
 with (out/"endpoint_flow.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(("partition","identity","event","support_bin","route_stop_score","forward","backward","lateral"));[w.writerow((x["partition"],x["identity"],x["event"],x["support_bin"],x["maximum_route_stop_score"],x["forward_mass"],x["backward_mass"],x["lateral_mass"])) for x in endpoint]
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10.2,4),constrained_layout=True)
 for index,split in enumerate(("fit","selection")):
  j=[x["maximum_route_stop_score"] for x in endpoint if x["partition"]==split and x["event"]=="junction"];t=[x["maximum_route_stop_score"] for x in endpoint if x["partition"]==split and x["event"]=="terminal"]
  axes[0].scatter(np.full(len(j),index-.12),j,color="#D95F02",alpha=.7,label="Junction" if index==0 else None);axes[0].scatter(np.full(len(t),index+.12),t,color="#2CA02C",alpha=.7,label="Terminal" if index==0 else None)
 axes[0].set_xticks((0,1),("C01-C06","C07-C08"));axes[0].set_ylabel("Route-stop score");axes[0].set_title("A  Identity-level route-conditioned separation");axes[0].legend(frameon=False)
 labels=("C01-C06","C07-C08");values=(split_summary["fit"]["terminal_vs_junction_auc"],split_summary["selection"]["terminal_vs_junction_auc"]);axes[1].bar(labels,values,color=("#7F7F7F","#9467BD"));axes[1].axhline(.8,color="black",linestyle="--",linewidth=1);axes[1].set_ylim(.5,1);axes[1].set_ylabel("Terminal vs junction ROC-AUC");axes[1].set_title("B  Fixed physical score, no training")
 fig.suptitle("Route-conditioned exit flow resolves route-agnostic structural aliasing")
 for suffix in ("png","pdf","svg"):fig.savefig(out/f"gse_route_conditioned_exit_flow.{suffix}",dpi=240 if suffix=="png" else None)
 plt.close(fig);(out/"figure_source.json").write_text(json.dumps({"schema_version":"gse_route_conditioned_exit_flow_figure_source_v1","summary":summary,"endpoint":endpoint},indent=2,sort_keys=True)+"\n");print(json.dumps(summary,sort_keys=True));return 0 if gates["all_passed"] else 2
if __name__=="__main__":raise SystemExit(main())
