#!/usr/bin/env python3
"""Verify, execute once, and seal the approved Gate-2 stability audit."""

from __future__ import annotations

import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json

RUN_ID="gate2_20260812_cano_phase3_stability_contract_audit_v1_seed0"
PYTHON=Path("/tmp/mtare_phase3_torch290_zarr2187/bin/python")
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_phase3_stability_contract_audit.py"
DATASET=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0"
SOURCE=PROJECT_ROOT/"results/gate2_representation/gate2_20260812_cano_phase3_multitask_structural_semantics_v1r3_seed0"
TIME_LIMIT=900;DISK_LIMIT=50*1024**2


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def verify_seal(path:Path,expected_entries:int)->dict:
    lines=path.read_text(encoding="utf-8").splitlines();bad=[]
    for line in lines:
        expected,raw=line.split("  ",1);target=PROJECT_ROOT/raw
        if not target.is_file() or sha256(target)!=expected:bad.append(raw)
    return {"entries":len(lines),"expected_entries":expected_entries,"mismatch_count":len(bad),"mismatches":bad}


def seal_manifest(run:Path)->int:
    destination=run/"artifacts/evidence_sha256.txt";files=sorted(path for path in run.rglob("*") if path.is_file() and path!=destination)
    destination.write_text("".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),encoding="utf-8");return len(files)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve()
    if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=2 or spec.get("operation")!="audit" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED":raise RuntimeError("approved Gate-2 audit scope mismatch")
    if load_json(PROJECT_ROOT/spec["config_path"]).get("status")!="APPROVED_FOR_ONE_EXECUTION":raise RuntimeError("proposal mismatch")
    for name,item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/item["path"])!=item["sha256"]:raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected:raise RuntimeError(f"frozen source mismatch: {raw}")
    source_seal=verify_seal(SOURCE/"artifacts/evidence_sha256.txt",62);dataset_seal=verify_seal(DATASET/"artifacts/evidence_sha256.txt",19338)
    if source_seal["mismatch_count"] or dataset_seal["mismatch_count"] or source_seal["entries"]!=62 or dataset_seal["entries"]!=19338:raise RuntimeError("source seal verification failed")
    probe="import json,torch,zarr,numpy,matplotlib;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'zarr':zarr.__version__,'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,'cuda_available':torch.cuda.is_available()}))"
    identity=json.loads(subprocess.run([str(PYTHON),"-c",probe],text=True,capture_output=True,check=True).stdout)
    expected={"torch":"2.9.0+cu129","cuda":"12.9","gpu":"NVIDIA GeForce RTX 5090 D","zarr":"2.18.7","numpy":"2.1.3","matplotlib":"3.10.0","cuda_available":True}
    if identity!=expected:raise RuntimeError(f"environment mismatch: {identity}")
    write_json(run/"config/environment_identity.json",{**identity,"host":platform.platform(),"source_run_seal":source_seal,"dataset_seal":dataset_seal})
    write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"No-update CPU/CUDA audit; 2,048 validation frames, zero train/C10/M-TARE reads."})
    env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3");env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    started=time.monotonic();code=124;output=""
    try:
        done=subprocess.run([str(PYTHON),str(EXECUTOR),"--run-dir",str(run)],cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=TIME_LIMIT,check=False);code=done.returncode;output=done.stdout
    except subprocess.TimeoutExpired as exc:output=(exc.stdout.decode() if isinstance(exc.stdout,bytes) else exc.stdout or "")+"\nTIMEOUT\n"
    duration=time.monotonic()-started;(run/"logs/01_stability_contract_audit.log").write_text(output+f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",encoding="utf-8");print(output,end="")
    summary=load_json(run/"metrics/summary.json") if (run/"metrics/summary.json").is_file() else {};size=sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    passed=bool(code==0 and duration<=TIME_LIMIT and size<=DISK_LIMIT and summary.get("overall_status")=="PASS_CANO_PHASE3_STABILITY_CONTRACT_AUDIT" and summary.get("unique_validation_frame_indices_read")==2048 and summary.get("training_frames_read")==0 and summary.get("strict_test_frames_read")==0 and summary.get("mtare_frames_read")==0 and summary.get("weight_updates")==0 and summary.get("checkpoints_created")==0)
    overall="PASS_CANO_PHASE3_STABILITY_CONTRACT_AUDIT" if passed else "FAIL_CANO_PHASE3_STABILITY_CONTRACT_AUDIT"
    write_json(run/"metrics/runner_summary.json",{"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT,"source_run_seal":source_seal,"dataset_seal":dataset_seal})
    write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"scientific_result":summary.get("scientific_result"),"note":"No-update CPU/CUDA audit; no checkpoint created."})
    sealed=seal_manifest(run);print(json.dumps({"overall_status":overall,"scientific_result":summary.get("scientific_result"),"sealed_files":sealed},indent=2));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
