#!/usr/bin/env python3
"""Execute and seal the three-seed frozen-input dual-Composer training."""

from __future__ import annotations

import argparse,json,os,subprocess,time,traceback
from pathlib import Path

import run_gse_explicit_composer_cache_export_v1 as helpers
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,validate_data_card,write_json


PASS="PASS_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1";FAIL="FAIL_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1";CARD_STATUS="APPROVED_FOR_ONE_IMMUTABLE_GSE_DUAL_COMPOSER_THREE_SEED_TRAINING_V1"
PYTHON=helpers.PYTHON
CACHE=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1r_seed0"
SUPERVISION=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_composer_supervision_v1r_seed0"
BASELINE=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r5_seed0"


def main()->int:
 parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve();run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
 if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("dual-Composer training executes exactly once")
 started=time.monotonic();overall=FAIL;error=None;before={};after={};subprocesses=[]
 try:
  card=load_json(PROJECT_ROOT/spec["data_card"]);report=validate_data_card(card)
  if not report.passed or card.get("status")!=CARD_STATUS:raise RuntimeError(f"dual-Composer training card invalid: {report.errors}")
  for record in spec["frozen_tools"].values():
   if helpers.sha256(PROJECT_ROOT/record["path"])!=record["sha256"]:raise RuntimeError(f"tool drift: {record['path']}")
  for relative,expected in spec["frozen_inputs"].items():
   before[relative]=helpers.sha256(PROJECT_ROOT/relative)
   if before[relative]!=expected:raise RuntimeError(f"input drift: {relative}")
  sources={"cache":helpers.verify_run_seal(CACHE,"PASS_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R"),"supervision":helpers.verify_run_seal(SUPERVISION,"PASS_GSE_COMPOSER_SUPERVISION_V1"),"baseline":helpers.verify_run_seal(BASELINE,"FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2")}
  write_json(run/"config/source_evidence.json",sources);write_json(run/"config/source_integrity_before.json",before);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
  env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3");env["PYTHONHASHSEED"]="0";env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
  tests=["tests/v3/unit/test_gse_composer_learning.py","tests/v3/unit/test_gse_typed_composers.py","tests/v3/unit/test_gse_composer_supervision.py","tests/v3/unit/test_train_gse_dual_composer_v1.py"]
  with (run/"logs/00_unit_tests.log").open("w",encoding="utf-8") as stream:result=subprocess.run([str(PYTHON),"-m","pytest","-q",*tests],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
  subprocesses.append({"stage":"unit_tests","returncode":result.returncode})
  if result.returncode:raise RuntimeError("dual-Composer unit tests failed")
  models=run/"artifacts/models";models.mkdir(parents=True)
  for seed in range(3):
   command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/train_gse_dual_composer_v1.py"),"--cache-root",str(CACHE/f"artifacts/cache/seed{seed}"),"--supervision-root",str(SUPERVISION/"artifacts/supervision/worlds"),"--baseline-root",str(BASELINE/f"artifacts/models/seed{seed}/development_predictions"),"--output-dir",str(models/f"seed{seed}"),"--seed",str(seed),"--epochs","10","--refusal-epochs","2","--batch-size","1024","--device","cuda"]
   (run/f"config/seed{seed}_command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
   with (run/f"logs/0{seed+1}_seed{seed}_training.log").open("w",encoding="utf-8") as stream:result=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=7200,check=False)
   subprocesses.append({"stage":f"seed{seed}_training","returncode":result.returncode})
   if result.returncode:raise RuntimeError(f"dual-Composer seed{seed} training failed")
  evaluation=run/"metrics/evaluation";command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/evaluate_gse_dual_composer_three_seed_v1.py"),"--models-root",str(models),"--output-dir",str(evaluation)]
  (run/"config/evaluation_command.txt").write_text(" ".join(command)+"\n",encoding="utf-8")
  with (run/"logs/04_evaluation.log").open("w",encoding="utf-8") as stream:result=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=1200,check=False)
  subprocesses.append({"stage":"evaluation","returncode":result.returncode});evaluation_summary=load_json(evaluation/"summary.json") if (evaluation/"summary.json").is_file() else {}
  if result.returncode not in (0,2) or evaluation_summary.get("status") not in (PASS,FAIL):raise RuntimeError("dual-Composer evaluation produced no scientific result")
  overall=evaluation_summary["status"]
  for relative in spec["frozen_inputs"]:after[relative]=helpers.sha256(PROJECT_ROOT/relative)
  if after!=before:raise RuntimeError("dual-Composer training source changed")
  write_json(run/"config/source_integrity_after.json",after);write_json(run/"metrics/summary.json",{"schema_version":"gse_dual_composer_three_seed_training_runner_v1","overall_status":overall,"scientific_pass":overall==PASS,"evaluation":evaluation_summary,"subprocesses":subprocesses,"optimizer_steps":evaluation_summary.get("optimizer_steps",0),"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"duration_seconds":time.monotonic()-started})
 except Exception as exc:
  error=f"{type(exc).__name__}: {exc}";overall=FAIL;(run/"logs/runner_error.log").write_text(traceback.format_exc(),encoding="utf-8");write_json(run/"metrics/summary.json",{"schema_version":"gse_dual_composer_three_seed_training_runner_v1","overall_status":overall,"scientific_pass":False,"error":error,"subprocesses":subprocesses,"optimizer_steps":0,"c09_worlds_read":0,"c10_worlds_read":0,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"duration_seconds":time.monotonic()-started})
 write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error,"duration_seconds":time.monotonic()-started});evidence=helpers.seal(run);print(json.dumps({"run_id":run_id,"overall_status":overall,"error":error,"evidence_files":evidence},indent=2));return 0 if error is None and overall==PASS else 2


if __name__=="__main__":raise SystemExit(main())
