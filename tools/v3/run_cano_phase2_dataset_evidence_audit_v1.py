#!/usr/bin/env python3
"""Verify, execute once and seal the approved Gate-1 corrective evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest

RUN_ID="gate1_20260812_cano_phase2_dataset_evidence_audit_v1_seed0"
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_phase2_dataset_evidence_audit_v1.py"
PYTHON=Path("/tmp/mtare_cano_e1_zarr2187/bin/python")
SOURCE_RUN=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0"
M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
TIME_LIMIT_SECONDS=3600
DISK_LIMIT_BYTES=1024**3


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def verify_seal(seal_path: Path, expected_entries: int) -> dict:
    lines=seal_path.read_text(encoding="utf-8").splitlines(); bad=[]
    for line in lines:
        expected,raw=line.split("  ",1); path=PROJECT_ROOT/raw
        if not path.is_file() or sha256(path)!=expected: bad.append(raw)
    return {"entries":len(lines),"expected_entries":expected_entries,"entry_count_passed":len(lines)==expected_entries,"mismatch_count":len(bad),"mismatches":bad,"seal_sha256":sha256(seal_path)}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--spec",required=True,type=Path); parser.add_argument("--run-dir",required=True,type=Path); args=parser.parse_args()
    spec=load_json(args.spec.resolve()); run_dir=args.run_dir.resolve()
    if run_dir.name!=RUN_ID or load_json(run_dir/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED": raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=1 or spec.get("operation")!="teacher_generation" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED": raise RuntimeError("approved Gate-1 corrective scope mismatch")
    card=load_json(PROJECT_ROOT/spec["data_card"]); proposal=load_json(PROJECT_ROOT/spec["config_path"])
    if card.get("status")!="APPROVED_FOR_ONE_EXECUTION" or card["approval"].get("authorized_operations")!=["teacher_generation"]: raise RuntimeError("approved corrective data card mismatch")
    if proposal.get("status")!="APPROVED_FOR_ONE_EXECUTION": raise RuntimeError("approved corrective proposal mismatch")
    for name,record in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]: raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected: raise RuntimeError(f"frozen source mismatch: {raw}")
    source_before=verify_seal(SOURCE_RUN/"artifacts/evidence_sha256.txt",12790)
    m1r_before=verify_seal(M1R/"artifacts/evidence_sha256.txt",1001)
    if source_before["mismatch_count"] or not source_before["entry_count_passed"] or m1r_before["mismatch_count"] or not m1r_before["entry_count_passed"]: raise RuntimeError("source seal precheck failed")
    if not PYTHON.is_file(): raise RuntimeError("approved E1+Zarr interpreter missing")
    version=subprocess.run([str(PYTHON),"-c","import numpy,open3d,zarr,numcodecs;print(numpy.__version__,open3d.__version__,zarr.__version__,numcodecs.__version__)"],text=True,capture_output=True,check=True).stdout.strip()
    if version!="1.26.4 0.19.0 2.18.7 0.15.1": raise RuntimeError(f"environment mismatch: {version}")
    if shutil.disk_usage(PROJECT_ROOT).free<2*1024**3: raise RuntimeError("less than 2 GiB free")
    write_json(run_dir/"config/environment_identity.json",{"python":str(PYTHON),"versions":version,"source_run_seal_before":source_before,"m1r_seal_before":m1r_before})
    shutil.copy2(PROJECT_ROOT/"configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json",run_dir/"config/world_registry.json")
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Corrective evidence-only audit; sealed dataset read-only; zero training."})
    env=os.environ.copy(); env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
    argv=[str(PYTHON),str(EXECUTOR),"--run-dir",str(run_dir)]; started=time.monotonic(); code=124; output=""
    try:
        done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=TIME_LIMIT_SECONDS,check=False); code=done.returncode; output=done.stdout
    except subprocess.TimeoutExpired as exc:
        output=(exc.stdout.decode() if isinstance(exc.stdout,bytes) else exc.stdout or "")+"\nTIMEOUT\n"
    duration=time.monotonic()-started; (run_dir/"logs/01_corrective_evidence_audit.log").write_text(output+f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",encoding="utf-8"); print(output,end="")
    source_after=verify_seal(SOURCE_RUN/"artifacts/evidence_sha256.txt",12790); source_unchanged=source_before==source_after
    summary=load_json(run_dir/"metrics/summary.json") if (run_dir/"metrics/summary.json").is_file() else {}
    pages=list((run_dir/"previews/train_complete_samples").glob("*.png")); coverage=list((run_dir/"previews/train_spatial_coverage").glob("*.png")); distributions=list((run_dir/"previews/train_distributions").glob("*.png")); size=sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed=bool(code==0 and duration<=TIME_LIMIT_SECONDS and summary.get("overall_status")=="PASS_CANO_PHASE2_DATASET_EVIDENCE_AUDIT_V1" and summary.get("frames")==112500 and summary.get("clusters")==22500 and summary.get("rejected_candidates")==61 and summary.get("replay_failures")==0 and summary.get("teacher_failures")==0 and len(pages)==10 and len(coverage)==10 and len(distributions)==1 and source_unchanged and size<=DISK_LIMIT_BYTES)
    overall="PASS_CANO_PHASE2_DATASET_EVIDENCE_AUDIT_V1" if passed else "FAIL_CANO_PHASE2_DATASET_EVIDENCE_AUDIT_V1"
    write_json(run_dir/"metrics/runner_summary.json",{"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"source_run_seal_before":source_before,"source_run_seal_after":source_after,"source_run_seal_unchanged":source_unchanged,"m1r_seal":m1r_before,"complete_sample_pages":len(pages),"spatial_coverage_figures":len(coverage),"distribution_figures":len(distributions),"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT_BYTES,"training_samples_consumed":0,"models":0})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":"Corrective evidence-only audit; source dataset remained sealed; zero training."})
    sealed=_seal_manifest(run_dir); print(json.dumps({"overall_status":overall,"sealed_files":sealed},indent=2)); return 0 if passed else 2


if __name__=="__main__": raise SystemExit(main())
