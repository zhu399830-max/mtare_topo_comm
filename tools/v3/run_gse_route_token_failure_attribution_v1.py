#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
PASS="PASS_GSE_ROUTE_TOKEN_FAILURE_ATTRIBUTION_V1";FAIL="FAIL_GSE_ROUTE_TOKEN_FAILURE_ATTRIBUTION_V1";PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def seal(run):
 target=run/"artifacts/evidence_sha256.txt";files=sorted(p for p in run.rglob("*") if p.is_file() and p!=target)
 with target.open("w") as f:
  for p in files:f.write(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n")
 return len(files)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--spec",required=True,type=Path);ap.add_argument("--run-dir",required=True,type=Path);a=ap.parse_args();spec=load_json(a.spec.resolve());run=a.run_dir.resolve();run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("route token failure attribution may execute only once")
 start=time.monotonic();overall=FAIL;error=None;result={};before={};after={}
 try:
  card=load_json(PROJECT_ROOT/spec["data_card"])
  if card.get("status")!="APPROVED_FOR_ONE_IMMUTABLE_GSE_ROUTE_TOKEN_FAILURE_ATTRIBUTION_V1":raise RuntimeError("route-token-attribution card drift")
  for rec in spec["frozen_tools"].values():
   if sha(PROJECT_ROOT/rec["path"])!=rec["sha256"]:raise RuntimeError(f"route-token-attribution tool drift {rec['path']}")
  for rel,expected in spec["frozen_inputs"].items():
   actual=sha(PROJECT_ROOT/rel)
   if actual!=expected:raise RuntimeError(f"route-token-attribution input drift {rel}")
   before[rel]=actual
  write_json(run/"config/source_integrity_before.json",before);write_json(run/"config/environment.json",{"executable":str(PYTHON),"gpu_used":False});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
  cache=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0/scratch/action_set_cache";endpoint=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0/artifacts/audit/endpoint_support_audit.jsonl";flow=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_route_conditioned_exit_flow_audit_v1_seed0/artifacts/audit/endpoint_flow.jsonl";observe=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_low_support_endpoint_observability_audit_v1_seed0/artifacts/audit/low_selection_rows.json";route_run=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_route_conditioned_event_residual_training_v1_seed0";graphs=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0/artifacts/candidates";cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_route_token_failure_attribution_v1.py"),"--cache-dir",str(cache),"--endpoint-audit",str(endpoint),"--endpoint-flow",str(flow),"--low-observability",str(observe),"--route-run",str(route_run),"--graph-root",str(graphs),"--output-dir",str(run/"artifacts/audit")];(run/"config/command.txt").write_text(" ".join(cmd)+"\n")
  env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
  with (run/"logs/audit.log").open("w") as f:code=subprocess.run(cmd,cwd=PROJECT_ROOT,env=env,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=False).returncode
  if code:raise RuntimeError(f"route-token-attribution executor failed {code}")
  result=load_json(run/"artifacts/audit/summary.json")
  if result.get("status")!=PASS or not result.get("gates",{}).get("all_passed") or result.get("population",{}).get("relation_endpoints")!=98 or result.get("recommendation")!="spatial_multi_event_teacher_before_new_head" or any(result.get(k)!=0 for k in ("optimizer_steps","model_inference_frames","model_updates","threshold_selection_steps","c09_worlds_read","c10_worlds_read","mtare_worlds_read")):raise RuntimeError("route-token-attribution result drift")
  after={rel:sha(PROJECT_ROOT/rel) for rel in before}
  if before!=after:raise RuntimeError("route-token-attribution sources changed")
  required=[run/"artifacts/audit/low_endpoint_attribution.json",run/"artifacts/audit/endpoint_adjacency.jsonl",run/"artifacts/audit/gse_route_token_failure_attribution.png"]
  if any(not path.is_file() for path in required):raise RuntimeError("route-token-attribution evidence incomplete")
  overall=PASS
 except Exception as exc:error=f"{type(exc).__name__}: {exc}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc())
 write_json(run/"metrics/summary.json",{"schema_version":"gse_route_token_failure_attribution_outer_v1","overall_status":overall,"error":error,"result":result,"duration_seconds":time.monotonic()-start,"source_unchanged":bool(before and before==after),"optimizer_steps":0,"model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if overall==PASS else "FAILED","overall_status":overall,"error":error});entries=seal(run);print(json.dumps({"overall_status":overall,"error":error,"seal_entries":entries},indent=2));return 0 if overall==PASS else 2
if __name__=="__main__":raise SystemExit(main())
