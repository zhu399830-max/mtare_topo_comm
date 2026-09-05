#!/usr/bin/env python3
"""Train, evaluate and graph-qualify one immutable three-seed endpoint residual."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
from run_gse_endpoint_geometry_capacity_v1 import _verify_axis_subset

PASS="PASS_GSE_ENDPOINT_IDENTITY_RESIDUAL_TRAINING_V1";FAIL="FAIL_GSE_ENDPOINT_IDENTITY_RESIDUAL_TRAINING_V1";PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def _verify(spec):
 out={}
 for rel,expected in spec["frozen_inputs"].items():
  actual=_sha(PROJECT_ROOT/rel)
  if actual!=expected:raise RuntimeError(f"endpoint residual input drift: {rel}")
  out[rel]=actual
 return out
def _run(cmd,log,env,timeout,allowed=(0,)):
 with log.open("w",encoding="utf-8") as f:c=subprocess.run(cmd,cwd=PROJECT_ROOT,env=env,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=False)
 if c.returncode not in allowed:raise RuntimeError(f"subprocess failed code {c.returncode}: {cmd[1]}")
 return int(c.returncode)
def _seal(run):
 seal=run/"artifacts/evidence_sha256.txt";files=sorted(p for p in run.rglob("*") if p.is_file() and p!=seal)
 with seal.open("w",encoding="utf-8") as f:
  for p in files:f.write(f"{_sha(p)}  {p.relative_to(PROJECT_ROOT)}\n")
 return len(files)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--spec",required=True,type=Path);ap.add_argument("--run-dir",required=True,type=Path);a=ap.parse_args();run=a.run_dir.resolve();spec=load_json(a.spec.resolve())
 expected_run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=expected_run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("endpoint residual run may execute only once")
 start=time.monotonic();overall=FAIL;error=None;ensemble={};graph={};processes=[];before=after={}
 try:
  if spec.get("gate")!=3 or spec.get("operation")!="training" or spec.get("seed")!=0:raise RuntimeError("endpoint residual scope drift")
  card=load_json(PROJECT_ROOT/spec["data_card"])
  if not str(card.get("status","")).startswith("APPROVED_FOR_ONE_IMMUTABLE_GSE_ENDPOINT_IDENTITY_RESIDUAL_TRAINING_"):raise RuntimeError("endpoint residual Data Card drift")
  for rec in spec["frozen_tools"].values():
   if _sha(PROJECT_ROOT/rec["path"])!=rec["sha256"]:raise RuntimeError(f"endpoint residual tool drift: {rec['path']}")
  before=_verify(spec);dataset=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0";axis_entries=_verify_axis_subset(dataset);write_json(run/"config/source_integrity_before.json",{"frozen_inputs":before,"axis_subset_seal_entries":axis_entries})
  versions=json.loads(subprocess.check_output([str(PYTHON),"-c","import json,matplotlib,numpy,scipy,sys,torch,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"],text=True));expected={"python":"3.13.5","numpy":"2.1.3","scipy":"1.15.3","matplotlib":"3.10.0","torch":"2.9.0+cu129","cuda":"12.9","zarr":"2.18.7","gpu":"NVIDIA GeForce RTX 5090 D"}
  if versions!=expected:raise RuntimeError(f"endpoint residual environment drift: {versions}")
  write_json(run/"config/environment.json",{"executable":str(PYTHON),"versions":versions,"deterministic_algorithms":True});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":expected_run_id,"state":"RUNNING"})
  env=os.environ.copy();env.update({"PYTHONPATH":str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3"),"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"})
  action=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0";cache=action/"scratch/action_set_cache";endpoint=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0/artifacts/audit/endpoint_support_audit.jsonl";models=run/"artifacts/models";models.mkdir(parents=True)
  commands=[]
  for seed in range(3):
   cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/train_gse_endpoint_identity_residual_v1.py"),"--cache-dir",str(cache),"--endpoint-audit",str(endpoint),"--base-checkpoint",str(action/f"artifacts/models/seed{seed}/best.pt"),"--output-dir",str(models/f"seed{seed}"),"--seed",str(seed),"--steps","200","--evaluation-interval","10","--learning-rate","0.0003","--weight-decay","0.0001"]
   commands.append(cmd);code=_run(cmd,run/f"logs/0{seed}_seed{seed}_training.log",env,1800);processes.append({"stage":"training","seed":seed,"returncode":code})
   result=load_json(models/f"seed{seed}/summary.json")
   if result.get("trainable_parameters")!=129 or result.get("optimizer_steps")!=200 or result.get("base_model_optimizer_steps")!=0 or any(result.get(k)!=0 for k in ("c09_worlds_read","c10_worlds_read","mtare_worlds_read")):raise RuntimeError(f"endpoint residual seed{seed} contract drift")
  evaluation=run/"artifacts/evaluation";cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/evaluate_gse_endpoint_identity_residual_v1.py"),"--cache-dir",str(cache),"--endpoint-audit",str(endpoint),"--baseline-ensemble",str(PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts/replay/action_ensemble.npz"),"--seed0",str(models/"seed0/all_outputs.npz"),"--seed1",str(models/"seed1/all_outputs.npz"),"--seed2",str(models/"seed2/all_outputs.npz"),"--output-dir",str(evaluation)];commands.append(cmd);code=_run(cmd,run/"logs/03_ensemble_evaluation.log",env,300,(0,2));processes.append({"stage":"ensemble_evaluation","returncode":code});ensemble=load_json(evaluation/"summary.json")
  teacher=PROJECT_ROOT/"results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts";replay=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0/artifacts";graph_out=run/"artifacts/graph_qualification";cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_endpoint_geometry_capacity_v1.py"),"--teacher",str(teacher/"teacher_observations.jsonl"),"--traversals",str(teacher/"traversal_manifest.jsonl"),"--frame-manifest",str(dataset/"artifacts/frame_manifest.jsonl"),"--dataset-root",str(dataset/"artifacts/dataset"),"--pair-cache",str(PROJECT_ROOT/"results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"),"--action-ensemble",str(evaluation/"corrected_action_ensemble.npz"),"--spatial-projection",str(replay/"projection/spatial_center_ensemble_all_rows.npz"),"--association-pairs",str(replay/"replay/association_pairs.npz"),"--objective-teacher",str(PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz"),"--baseline-summary",str(PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_post_commit_endpoint_inventory_v1_seed0/artifacts/inventory/summary.json"),"--output-dir",str(graph_out)];commands.append(cmd);code=_run(cmd,run/"logs/04_graph_qualification.log",env,600,(0,2));processes.append({"stage":"graph_qualification","returncode":code});graph=load_json(graph_out/"summary.json")
  selection=graph.get("scores",{}).get("selection",{});fit=graph.get("scores",{}).get("fit",{});graph_gates={"fit_node_precision":fit.get("node_precision",0)>=.98,"fit_edge_precision":fit.get("edge_precision",0)>=.98,"selection_node_precision":selection.get("node_precision",0)>=.98,"selection_edge_precision":selection.get("edge_precision",0)>=.98,"selection_node_recall_no_regression":selection.get("node_recall",0)>=.7043795620437956,"selection_edge_recall_no_regression":selection.get("edge_recall",0)>=4/13,"selection_false_loop":selection.get("false_loop_merge_fraction",1)<=.01};graph_gates["all_passed"]=all(graph_gates.values())
  overall=PASS if ensemble.get("gates",{}).get("all_passed") and graph_gates["all_passed"] else FAIL
  (run/"config/command.txt").write_text("\n".join(" ".join(cmd) for cmd in commands)+"\n",encoding="utf-8")
  after=_verify(spec)
  if before!=after or _verify_axis_subset(dataset)!=axis_entries:raise RuntimeError("endpoint residual sources changed")
  required=[evaluation/"corrected_action_ensemble.npz",evaluation/"gse_endpoint_identity_residual.png",graph_out/"summary.json",graph_out/"gse_endpoint_geometry_capacity.png"]
  if any(not p.is_file() for p in required):raise RuntimeError("endpoint residual evidence incomplete")
 except Exception as exc:error=f"{type(exc).__name__}: {exc}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
 graph_gates=locals().get("graph_gates",{});summary={"schema_version":"gse_endpoint_identity_residual_training_outer_v1","overall_status":overall,"scientific_pass":overall==PASS,"error":error,"ensemble":ensemble,"graph":graph,"graph_gates":graph_gates,"processes":processes,"duration_seconds":time.monotonic()-start,"source_unchanged":bool(before and before==after),"optimizer_steps":600 if len([p for p in processes if p["stage"]=="training" and p["returncode"]==0])==3 else None,"trainable_parameters_per_seed":129,"base_model_optimizer_steps":0,"threshold_selection_steps":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0};write_json(run/"metrics/summary.json",summary);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":expected_run_id,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error});entries=_seal(run);print(json.dumps({"overall_status":overall,"error":error,"seal_entries":entries},indent=2));return 0 if overall==PASS and error is None else 2
if __name__=="__main__":raise SystemExit(main())
