#!/usr/bin/env python3
"""Verify, execute once and seal the approved selector-V2 feasibility audit."""

from __future__ import annotations

import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest

RUN_ID="gate1_20260812_cano_phase2_selector_v2_feasibility_audit_seed0";PYTHON=Path("/tmp/mtare_cano_e1_zarr2187/bin/python");EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_phase2_selector_v2_feasibility_audit.py";M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0";SOURCE=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0";TIME_LIMIT=600;DISK_LIMIT=100*1024**2

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def verify_seal(path,expected_count):
 lines=Path(path).read_text().splitlines();bad=[]
 for line in lines:
  expected,raw=line.split("  ",1);p=PROJECT_ROOT/raw
  if not p.is_file() or sha(p)!=expected:bad.append(raw)
 return {"entries":len(lines),"expected_entries":expected_count,"mismatch_count":len(bad),"mismatches":bad}

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve()
 if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("one-time run state mismatch")
 if spec.get("gate")!=1 or spec.get("operation")!="audit" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED":raise RuntimeError("approved selector audit scope mismatch")
 proposal=load_json(PROJECT_ROOT/spec["config_path"])
 if proposal.get("status")!="APPROVED_FOR_ONE_EXECUTION":raise RuntimeError("proposal mismatch")
 for name,item in spec["frozen_tools"].items():
  if sha(PROJECT_ROOT/item["path"])!=item["sha256"]:raise RuntimeError(f"frozen tool mismatch: {name}")
 for raw,expected in spec["frozen_source_files"].items():
  if sha(PROJECT_ROOT/raw)!=expected:raise RuntimeError(f"frozen source mismatch: {raw}")
 m1r=verify_seal(M1R/"artifacts/evidence_sha256.txt",1001);source=verify_seal(SOURCE/"artifacts/evidence_sha256.txt",12790)
 if m1r["mismatch_count"] or source["mismatch_count"] or m1r["entries"]!=1001 or source["entries"]!=12790:raise RuntimeError("source seal failed")
 version=subprocess.run([str(PYTHON),"-c","import numpy,scipy,matplotlib;print(numpy.__version__,scipy.__version__,matplotlib.__version__)"],text=True,capture_output=True,check=True).stdout.strip()
 if version!="1.26.4 1.12.0 3.11.1":raise RuntimeError(f"environment mismatch: {version}")
 write_json(run/"config/environment_identity.json",{"python":str(PYTHON),"versions":version,"m1r_seal":m1r,"source_dataset_seal":source});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Zero-ray selector feasibility only; no data export or training."})
 env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3");argv=[str(PYTHON),str(EXECUTOR),"--run-dir",str(run)];started=time.monotonic();code=124;output=""
 try:
  done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=TIME_LIMIT,check=False);code=done.returncode;output=done.stdout
 except subprocess.TimeoutExpired as exc:output=(exc.stdout.decode() if isinstance(exc.stdout,bytes) else exc.stdout or "")+"\nTIMEOUT\n"
 duration=time.monotonic()-started;(run/"logs/01_selector_v2_feasibility.log").write_text(output+f"\nduration_seconds={duration:.6f}\nexit_code={code}\n");print(output,end="")
 summary=load_json(run/"metrics/summary.json") if (run/"metrics/summary.json").is_file() else {};figures=list((run/"previews/train_spatial_coverage").glob("*.png"));size=sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
 passed=bool(code==0 and duration<=TIME_LIMIT and summary.get("overall_status")=="PASS_CANO_PHASE2_SELECTOR_V2_FEASIBILITY_AUDIT" and summary.get("selected_clusters")==22500 and summary.get("worlds_with_missing_tunnels")==0 and summary.get("maximum_candidate_to_selected_same_tunnel_arc_distance_m",99)<=10 and summary.get("deterministic_replay_failures")==0 and len(figures)==10 and size<=DISK_LIMIT)
 overall="PASS_CANO_PHASE2_SELECTOR_V2_FEASIBILITY_AUDIT" if passed else "FAIL_CANO_PHASE2_SELECTOR_V2_FEASIBILITY_AUDIT";write_json(run/"metrics/runner_summary.json",{"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"train_spatial_figures":len(figures),"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT,"m1r_seal":m1r,"source_dataset_seal":source,"data_exported":False,"training_samples_consumed":0,"models":0});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":"Zero-ray selector feasibility only; no data export or training."});sealed=_seal_manifest(run);print(json.dumps({"overall_status":overall,"sealed_files":sealed},indent=2));return 0 if passed else 2

if __name__=="__main__":raise SystemExit(main())
