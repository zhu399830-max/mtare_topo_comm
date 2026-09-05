#!/usr/bin/env python3
"""Execute and seal circular slot-transport readiness."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,validate_data_card,write_json
PASS="PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_V1";FAIL="FAIL_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_V1";PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python");DATASET=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0";TEACHER=PROJECT_ROOT/"results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
def sha256(path):
 d=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b""):d.update(block)
 return d.hexdigest()
def seal(run):
 target=run/"artifacts/evidence_sha256.txt";files=sorted(path for path in run.rglob("*") if path.is_file() and path!=target)
 with target.open("w") as stream:
  for path in files:stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
 return len(files)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve();run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("slot readiness executes once")
 started=time.monotonic();overall=FAIL;error=None;result={};before={};after={};code=None
 try:
  card=load_json(PROJECT_ROOT/spec["data_card"]);validation=validate_data_card(card)
  expected_card_status=spec.get("data_card_status","APPROVED_FOR_ONE_IMMUTABLE_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_READINESS_V1")
  if not validation.passed or card.get("status")!=expected_card_status:raise RuntimeError(f"Data Card invalid: {validation.errors}")
  for record in spec["frozen_tools"].values():
   if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]:raise RuntimeError(f"tool drift: {record['path']}")
  for relative,expected in spec["frozen_inputs"].items():
   before[relative]=sha256(PROJECT_ROOT/relative)
   if before[relative]!=expected:raise RuntimeError(f"input drift: {relative}")
  environment=json.loads(subprocess.check_output([str(PYTHON),"-c","import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__},sort_keys=True))"],text=True))
  if environment!={"python":"3.13.5","numpy":"2.1.3","torch":"2.9.0+cu129","zarr":"2.18.7"}:raise RuntimeError(f"environment drift: {environment}")
  write_json(run/"config/environment.json",{"executable":str(PYTHON),"versions":environment,"device":"cpu"});write_json(run/"config/source_integrity_before.json",before);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
  command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_cardinality_conditioned_circular_slot_transport_readiness_v1.py"),"--teacher-root",str(TEACHER/"artifacts/export/teacher"),"--source-root",str(DATASET/"artifacts/dataset/train"),"--output-dir",str(run/"artifacts/readiness")];write_json(run/"config/commands.json",[command]);environment_variables=os.environ.copy();environment_variables["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
  with (run/"logs/00_readiness.log").open("w") as stream:code=subprocess.run(command,cwd=PROJECT_ROOT,env=environment_variables,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=1200).returncode
  if code not in (0,2):raise RuntimeError(f"executor system failure: {code}")
  result=load_json(run/"artifacts/readiness/summary.json");passed=result.get("status")==PASS and result.get("scientific_pass") is True
  if (code==0)!=passed:raise RuntimeError("status/return mismatch")
  if any(result.get(name)!=0 for name in ("optimizer_steps","checkpoint_writes","threshold_selection_steps","c09_worlds_read","c10_worlds_read","mtare_worlds_read","graph_replays","planner_calls")):raise RuntimeError("forbidden operation")
  required=[run/"artifacts/readiness/summary.json",run/"artifacts/readiness/figure_source.json",run/"artifacts/readiness/selected_real_rows.csv",*[run/f"artifacts/readiness/gse_cardinality_conditioned_circular_slot_transport_readiness_v1.{suffix}" for suffix in ("png","pdf","svg")]]
  if any(not path.is_file() or path.stat().st_size==0 for path in required):raise RuntimeError("evidence incomplete")
  after={relative:sha256(PROJECT_ROOT/relative) for relative in before}
  if before!=after:raise RuntimeError("source changed")
  overall=PASS if passed else FAIL
 except Exception as exception:error=f"{type(exception).__name__}: {exception}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc())
 write_json(run/"metrics/summary.json",{"schema_version":"gse_cardinality_conditioned_circular_slot_transport_readiness_outer_v1","overall_status":overall,"scientific_pass":overall==PASS,"error":error,"executor_returncode":code,"result":result,"duration_seconds":time.monotonic()-started,"source_unchanged":bool(before and before==after),"optimizer_steps":0,"checkpoint_writes":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if overall==PASS and error is None else "FAILED","overall_status":overall,"error":error});entries=seal(run);print(json.dumps({"overall_status":overall,"error":error,"decision":result.get("decision"),"seal_entries":entries},indent=2));return 0 if overall==PASS and error is None else 2
if __name__=="__main__":raise SystemExit(main())
