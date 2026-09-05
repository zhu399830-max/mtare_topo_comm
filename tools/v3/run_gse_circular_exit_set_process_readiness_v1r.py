#!/usr/bin/env python3
"""Execute and seal metric-only V1R exit-set readiness."""

from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,validate_data_card,write_json

PASS="PASS_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1R";FAIL="FAIL_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1R"
INNER_PASS="PASS_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1";CARD_STATUS="APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1R"
PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0";TEACHER=PROJECT_ROOT/"results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
def sha256(path):
 d=hashlib.sha256()
 with path.open("rb") as s:
  for b in iter(lambda:s.read(4*1024*1024),b""):d.update(b)
 return d.hexdigest()
def seal(run_dir):
 target=run_dir/"artifacts/evidence_sha256.txt";files=sorted(p for p in run_dir.rglob("*") if p.is_file() and p!=target)
 with target.open("w",encoding="utf-8") as s:
  for p in files:s.write(f"{sha256(p)}  {p.relative_to(PROJECT_ROOT)}\n")
 return len(files)
def main():
 p=argparse.ArgumentParser();p.add_argument("--spec",required=True,type=Path);p.add_argument("--run-dir",required=True,type=Path);a=p.parse_args();spec=load_json(a.spec.resolve());run=a.run_dir.resolve();rid=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=rid or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("V1R executes once")
 started=time.monotonic();overall=FAIL;error=None;result={};before={};after={};code=None
 try:
  card=load_json(PROJECT_ROOT/spec["data_card"]);v=validate_data_card(card)
  if not v.passed or card.get("status")!=CARD_STATUS:raise RuntimeError(f"V1R card invalid:{v.errors}")
  for r in spec["frozen_tools"].values():
   if sha256(PROJECT_ROOT/r["path"])!=r["sha256"]:raise RuntimeError(f"tool drift:{r['path']}")
  for rel,expected in spec["frozen_inputs"].items():
   before[rel]=sha256(PROJECT_ROOT/rel)
   if before[rel]!=expected:raise RuntimeError(f"input drift:{rel}")
  env=json.loads(subprocess.check_output([str(PYTHON),"-c","import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__},sort_keys=True))"],text=True));expected={"python":"3.13.5","numpy":"2.1.3","torch":"2.9.0+cu129","zarr":"2.18.7"}
  if env!=expected:raise RuntimeError(f"environment drift:{env}")
  write_json(run/"config/environment.json",{"executable":str(PYTHON),"versions":env,"device":"cpu","deterministic_algorithms":True});write_json(run/"config/source_integrity_before.json",before)
  command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_circular_exit_set_process_readiness_v1.py"),"--teacher-root",str(TEACHER/"artifacts/export/teacher"),"--source-root",str(DATASET/"artifacts/dataset/train"),"--output-dir",str(run/"artifacts/readiness"),"--synthetic-rotation-atol","0.000001"]
  write_json(run/"config/commands.json",[command]);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":rid,"state":"RUNNING"});process=os.environ.copy();process.update({"PYTHONPATH":str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3"),"OMP_NUM_THREADS":"4"})
  with (run/"logs/00_readiness.log").open("w",encoding="utf-8") as s:code=subprocess.run(command,cwd=PROJECT_ROOT,env=process,stdout=s,stderr=subprocess.STDOUT,text=True,timeout=900,check=False).returncode
  if code not in (0,2):raise RuntimeError(f"executor system failure:{code}")
  result=load_json(run/"artifacts/readiness/summary.json");passed=result.get("status")==INNER_PASS and result.get("scientific_pass") is True
  if (code==0)!=passed:raise RuntimeError("status/return mismatch")
  if result.get("synthetic",{}).get("rotation_atol")!=1e-6 or result.get("synthetic",{}).get("rotation_error",1)>1e-6:raise RuntimeError("metric repair drift")
  counts=result.get("population",{}).get("counts",{})
  if counts.get("observations")!=188126 or result.get("model",{}).get("parameters")!=769784:raise RuntimeError("population drift")
  after={rel:sha256(PROJECT_ROOT/rel) for rel in before}
  if before!=after:raise RuntimeError("source changed")
  overall=PASS if passed else FAIL
 except Exception as exc:error=f"{type(exc).__name__}: {exc}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
 write_json(run/"metrics/summary.json",{"schema_version":"gse_circular_exit_set_process_readiness_outer_v1r","overall_status":overall,"scientific_pass":overall==PASS,"error":error,"executor_returncode":code,"result":result,"duration_seconds":time.monotonic()-started,"source_unchanged":bool(before and before==after),"metric_only_repair":{"v1_rotation_error":5.960464477539063e-8,"v1_exact_comparison":0.0,"v1r_atol":1e-6},"optimizer_steps":0,"checkpoint_writes":0,"threshold_selection_steps":0,"graph_replays":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":rid,"state":"COMPLETED" if overall==PASS and error is None else "FAILED","overall_status":overall,"error":error});entries=seal(run);print(json.dumps({"overall_status":overall,"error":error,"decision":result.get("decision"),"seal_entries":entries},indent=2));return 0 if overall==PASS and error is None else 2
if __name__=="__main__":raise SystemExit(main())
