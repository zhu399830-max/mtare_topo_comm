#!/usr/bin/env python3
"""Execute and seal corrected dual-Composer supervision materialization."""

from __future__ import annotations

import argparse,json,os,subprocess,time,traceback
from pathlib import Path

import run_gse_explicit_composer_cache_export_v1 as helpers
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,validate_data_card,write_json

PASS="PASS_GSE_COMPOSER_SUPERVISION_V1";FAIL="FAIL_GSE_COMPOSER_SUPERVISION_V1"
PYTHON=helpers.PYTHON
CACHE=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1r_seed0"
TEACHER=PROJECT_ROOT/"results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
TIMING=PROJECT_ROOT/"results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"


def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--spec",required=True,type=Path);p.add_argument("--run-dir",required=True,type=Path);a=p.parse_args();spec=load_json(a.spec.resolve());run=a.run_dir.resolve();run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("Composer supervision executes exactly once")
 started=time.monotonic();overall=FAIL;error=None;before={};after={};subprocesses=[]
 try:
  card=load_json(PROJECT_ROOT/spec["data_card"]);report=validate_data_card(card)
  expected_card_status="APPROVED_FOR_ONE_IMMUTABLE_GSE_COMPOSER_SUPERVISION_V1R" if spec["slug"].endswith("_v1r") else "APPROVED_FOR_ONE_IMMUTABLE_GSE_COMPOSER_SUPERVISION_V1"
  if not report.passed or card.get("status")!=expected_card_status:raise RuntimeError(f"Composer supervision card invalid: {report.errors}")
  for record in spec["frozen_tools"].values():
   if helpers.sha256(PROJECT_ROOT/record["path"])!=record["sha256"]:raise RuntimeError(f"tool drift: {record['path']}")
  for relative,expected in spec["frozen_inputs"].items():
   before[relative]=helpers.sha256(PROJECT_ROOT/relative)
   if before[relative]!=expected:raise RuntimeError(f"input drift: {relative}")
  sources={"cache":helpers.verify_run_seal(CACHE,"PASS_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R"),"teacher":helpers.verify_run_seal(TEACHER,"PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"),"timing":helpers.verify_run_seal(TIMING,"PASS_GSE_CAUSAL_EVENT_SUPERVISION_AUDIT_V1")}
  write_json(run/"config/source_evidence.json",sources);write_json(run/"config/source_integrity_before.json",before);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
  env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
  with (run/"logs/00_unit_tests.log").open("w",encoding="utf-8") as stream:u=subprocess.run([str(PYTHON),"-m","pytest","-q","tests/v3/unit/test_gse_composer_supervision.py"],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
  subprocesses.append({"stage":"unit_tests","returncode":u.returncode})
  if u.returncode:raise RuntimeError("Composer supervision unit tests failed")
  command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/materialize_gse_composer_supervision_v1.py"),"--cache-root",str(CACHE/"artifacts/cache/seed0"),"--teacher",str(TEACHER/"artifacts/teacher_observations.jsonl"),"--transition-timing",str(TIMING/"artifacts/transition_timing.csv"),"--output-dir",str(run/"artifacts/supervision")]
  (run/"config/command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
  with (run/"logs/01_supervision.log").open("w",encoding="utf-8") as stream:c=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
  subprocesses.append({"stage":"supervision","returncode":c.returncode});result=load_json(run/"artifacts/supervision/summary.json") if (run/"artifacts/supervision/summary.json").is_file() else {}
  if c.returncode not in (0,2) or result.get("status") not in (PASS,FAIL):raise RuntimeError("Composer supervision produced no scientific result")
  overall=result["status"]
  for relative in spec["frozen_inputs"]:after[relative]=helpers.sha256(PROJECT_ROOT/relative)
  if after!=before:raise RuntimeError("Composer supervision source changed")
  write_json(run/"config/source_integrity_after.json",after);write_json(run/"metrics/summary.json",{"schema_version":"gse_composer_supervision_runner_v1","overall_status":overall,"scientific_pass":overall==PASS,"result":result,"subprocesses":subprocesses,"optimizer_steps":0,"model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"duration_seconds":time.monotonic()-started})
 except Exception as e:
  error=f"{type(e).__name__}: {e}";overall=FAIL;(run/"logs/runner_error.log").write_text(traceback.format_exc(),encoding="utf-8");write_json(run/"metrics/summary.json",{"schema_version":"gse_composer_supervision_runner_v1","overall_status":overall,"scientific_pass":False,"error":error,"subprocesses":subprocesses,"optimizer_steps":0,"model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"duration_seconds":time.monotonic()-started})
 write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error,"duration_seconds":time.monotonic()-started});evidence=helpers.seal(run);print(json.dumps({"run_id":run_id,"overall_status":overall,"error":error,"evidence_files":evidence},indent=2));return 0 if error is None and overall==PASS else 2


if __name__=="__main__":raise SystemExit(main())
