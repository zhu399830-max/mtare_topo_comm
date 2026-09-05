#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
PASS="PASS_GSE_SPATIAL_EVENT_SET_READINESS_V1";FAIL="FAIL_GSE_SPATIAL_EVENT_SET_READINESS_V1";PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
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
 p=argparse.ArgumentParser();p.add_argument("--spec",required=True,type=Path);p.add_argument("--run-dir",required=True,type=Path);a=p.parse_args();spec=load_json(a.spec.resolve());run=a.run_dir.resolve();run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("readiness executes once")
 start=time.monotonic();overall=FAIL;error=None;result={};before={};after={}
 try:
  for rec in spec["frozen_tools"].values():
   if sha(PROJECT_ROOT/rec["path"])!=rec["sha256"]:raise RuntimeError(f"tool drift {rec['path']}")
  for rel,expected in spec["frozen_inputs"].items():
   before[rel]=sha(PROJECT_ROOT/rel)
   if before[rel]!=expected:raise RuntimeError(f"input drift {rel}")
  write_json(run/"config/source_integrity_before.json",before);write_json(run/"config/environment.json",{"executable":str(PYTHON),"torch":__import__("torch").__version__,"gpu_used":False});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
  teacher=PROJECT_ROOT/"results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0/artifacts/export/teacher";source=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/dataset";cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_spatial_event_set_readiness_v1.py"),"--teacher-root",str(teacher),"--source-root",str(source),"--output-dir",str(run/"artifacts/audit")];(run/"config/command.txt").write_text(" ".join(cmd)+"\n");env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
  with (run/"logs/audit.log").open("w") as f:code=subprocess.run(cmd,cwd=PROJECT_ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,text=True,timeout=300,check=False).returncode
  if code not in (0,2):raise RuntimeError(f"executor system failure {code}")
  result=load_json(run/"artifacts/audit/summary.json")
  if result.get("status") not in (PASS,FAIL) or result.get("population",{}).get("observations")!=188126 or any(result.get(k)!=0 for k in ("optimizer_steps","trained_model_inference_frames","c09_worlds_read","c10_worlds_read","mtare_worlds_read")):raise RuntimeError("readiness result drift")
  after={rel:sha(PROJECT_ROOT/rel) for rel in before}
  if before!=after:raise RuntimeError("sources changed")
  overall=result["status"]
 except Exception as exc:error=f"{type(exc).__name__}: {exc}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc())
 write_json(run/"metrics/summary.json",{"schema_version":"gse_spatial_event_set_readiness_outer_v1","overall_status":overall,"error":error,"result":result,"duration_seconds":time.monotonic()-start,"source_unchanged":bool(before and before==after)});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if overall==PASS and error is None else "FAILED","overall_status":overall,"error":error});n=seal(run);print(json.dumps({"overall_status":overall,"error":error,"seal_entries":n},indent=2));return 0 if overall==PASS and error is None else 2
if __name__=="__main__":raise SystemExit(main())
