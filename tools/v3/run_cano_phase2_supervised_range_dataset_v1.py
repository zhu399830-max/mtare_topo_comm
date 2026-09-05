#!/usr/bin/env python3
"""Verify, execute once, and seal the approved Gate-1 dataset export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest

RUN_ID="gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0"
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_phase2_supervised_range_dataset_v1.py"
PYTHON=Path("/tmp/mtare_cano_e1_zarr2187/bin/python")
M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
TIME_LIMIT_SECONDS=10800
DISK_LIMIT_BYTES=12*1024**3


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def verify_m1r_seal() -> dict:
    lines=(M1R/"artifacts/evidence_sha256.txt").read_text().splitlines();bad=[]
    for line in lines:
        expected,raw=line.split("  ",1);path=PROJECT_ROOT/raw
        if not path.is_file() or sha256(path)!=expected:bad.append(raw)
    return {"entries":len(lines),"mismatch_count":len(bad),"mismatches":bad}


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args()
    spec=load_json(args.spec.resolve());run_dir=args.run_dir.resolve()
    if run_dir.name!=RUN_ID or load_json(run_dir/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=1 or spec.get("operation")!="data_export" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED":raise RuntimeError("approved Gate-1 scope mismatch")
    card=load_json(PROJECT_ROOT/spec["data_card"]);proposal=load_json(PROJECT_ROOT/spec["config_path"])
    if card.get("status")!="APPROVED_FOR_ONE_EXECUTION" or card["approval"].get("authorized_operations")!=["data_export","teacher_generation"]:raise RuntimeError("approved data card mismatch")
    if proposal.get("status")!="APPROVED_FOR_ONE_EXECUTION":raise RuntimeError("approved proposal mismatch")
    for name,record in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]:raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected:raise RuntimeError(f"frozen source mismatch: {raw}")
    seal=verify_m1r_seal()
    if seal["mismatch_count"]:raise RuntimeError(f"M1R seal mismatch: {seal}")
    if not PYTHON.is_file():raise RuntimeError("approved E1+Zarr interpreter missing")
    version=subprocess.run([str(PYTHON),"-c","import numpy,open3d,zarr,numcodecs;print(numpy.__version__,open3d.__version__,zarr.__version__,numcodecs.__version__)"],text=True,capture_output=True,check=True).stdout.strip()
    if version!="1.26.4 0.19.0 2.18.7 0.15.1":raise RuntimeError(f"environment mismatch: {version}")
    if shutil.disk_usage(PROJECT_ROOT).free<DISK_LIMIT_BYTES+1024**3:raise RuntimeError("less than 13 GiB free")
    write_json(run_dir/"config/environment_identity.json",{"python":str(PYTHON),"versions":version,"m1r_seal":seal})
    shutil.copy2(PROJECT_ROOT/"configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json",run_dir/"config/world_registry.json")
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Approved Gate-1 export plus objective teacher; zero training."})
    env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
    argv=[str(PYTHON),str(EXECUTOR),"--run-dir",str(run_dir)];started=time.monotonic();code=124;output=""
    try:
        done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=TIME_LIMIT_SECONDS,check=False);code=done.returncode;output=done.stdout
    except subprocess.TimeoutExpired as exc:
        output=(exc.stdout.decode() if isinstance(exc.stdout,bytes) else exc.stdout or "")+"\nTIMEOUT\n"
    duration=time.monotonic()-started;(run_dir/"logs/01_export.log").write_text(output+f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",encoding="utf-8");print(output,end="")
    summary=load_json(run_dir/"metrics/summary.json") if (run_dir/"metrics/summary.json").is_file() else {}
    shards=list((run_dir/"artifacts/dataset").glob("*/*.zarr"));pages=list((run_dir/"previews/train_cluster_pages").glob("*.png"));coverage=list((run_dir/"previews/train_recipe_coverage").glob("*.png"));size=sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    passed=bool(code==0 and duration<=TIME_LIMIT_SECONDS and summary.get("overall_status")=="PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1" and summary.get("frames")==112500 and len(shards)==90 and len(pages)==10 and len(coverage)==10 and size<=DISK_LIMIT_BYTES)
    overall="PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1" if passed else "FAIL_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1"
    write_json(run_dir/"metrics/runner_summary.json",{"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"zarr_shards":len(shards),"cluster_pages":len(pages),"coverage_figures":len(coverage),"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT_BYTES,"m1r_seal":seal,"training_samples_consumed":0,"models":0})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":"Formal range/teacher dataset only; zero training or M-TARE changes."})
    sealed=_seal_manifest(run_dir);print(json.dumps({"overall_status":overall,"sealed_files":sealed},indent=2));return 0 if passed else 2

if __name__=="__main__":raise SystemExit(main())
