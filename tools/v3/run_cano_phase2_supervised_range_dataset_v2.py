#!/usr/bin/env python3
"""Verify, execute once, and seal the approved Gate-1 V2 export."""

from __future__ import annotations

import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest

RUN_ID="gate1_20260812_cano_phase2_supervised_range_dataset_v2_seed0"
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_phase2_supervised_range_dataset_v2.py"
PYTHON=Path("/tmp/mtare_cano_e1_zarr2187/bin/python")
M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
TIME_LIMIT_SECONDS=3600;DISK_LIMIT_BYTES=12*1024**3


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def verify_seal(path: Path) -> dict:
    lines=(path/"artifacts/evidence_sha256.txt").read_text().splitlines();bad=[]
    for line in lines:
        expected,raw=line.split("  ",1);target=PROJECT_ROOT/raw
        if not target.is_file() or sha256(target)!=expected: bad.append(raw)
    return {"entries":len(lines),"mismatch_count":len(bad),"mismatches":bad}


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve()
    if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED": raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=1 or spec.get("operation")!="data_export" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED": raise RuntimeError("approved Gate-1 scope mismatch")
    card=load_json(PROJECT_ROOT/spec["data_card"]);proposal=load_json(PROJECT_ROOT/spec["config_path"])
    if card.get("status")!="APPROVED_FOR_ONE_IMPLEMENTATION_AND_ONE_EXECUTION_NOT_YET_EXECUTED" or card["approval"].get("authorized_operations")!=["data_export","teacher_generation"]: raise RuntimeError("approved data card mismatch")
    if proposal.get("status")!="APPROVED_FOR_ONE_IMPLEMENTATION_AND_ONE_EXECUTION_NOT_YET_EXECUTED": raise RuntimeError("approved proposal mismatch")
    for name,record in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]: raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected: raise RuntimeError(f"frozen source mismatch: {raw}")
    m1r=verify_seal(M1R);selector=verify_seal(PROJECT_ROOT/spec["prerequisites"]["selector_v2_run"])
    if m1r["mismatch_count"] or selector["mismatch_count"]: raise RuntimeError(f"source seal mismatch: {m1r} {selector}")
    if shutil.disk_usage(PROJECT_ROOT).free<DISK_LIMIT_BYTES+1024**3: raise RuntimeError("less than 13 GiB free")
    version=subprocess.run([str(PYTHON),"-c","import numpy,open3d,zarr,numcodecs;print(numpy.__version__,open3d.__version__,zarr.__version__,numcodecs.__version__)"],text=True,capture_output=True,check=True).stdout.strip()
    if version!="1.26.4 0.19.0 2.18.7 0.15.1": raise RuntimeError(f"environment mismatch: {version}")
    write_json(run/"config/environment_identity.json",{"python":str(PYTHON),"versions":version,"m1r_seal":m1r,"selector_v2_seal":selector});shutil.copy2(PROJECT_ROOT/"configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json",run/"config/world_registry.json")
    write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Approved eligibility-first V2 data and objective teacher export; zero training."})
    env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3");argv=[str(PYTHON),str(EXECUTOR),"--run-dir",str(run)];started=time.monotonic();code=124;output=""
    try:
        done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=TIME_LIMIT_SECONDS,check=False);code=done.returncode;output=done.stdout
    except subprocess.TimeoutExpired as exc: output=(exc.stdout.decode() if isinstance(exc.stdout,bytes) else exc.stdout or "")+"\nTIMEOUT\n"
    duration=time.monotonic()-started;(run/"logs/01_export.log").write_text(output+f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",encoding="utf-8");print(output,end="")
    summary=load_json(run/"metrics/summary.json") if (run/"metrics/summary.json").is_file() else {};shards=list((run/"artifacts/dataset").glob("*/*.zarr"));pages=list((run/"previews/train_complete_samples").glob("*.png"));spatial=list((run/"previews/train_spatial_coverage").glob("*.png"));distributions=list((run/"previews/train_distributions").glob("*.png"));size=sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    counts={name:sum(1 for _ in (run/f"artifacts/{name}").open()) if (run/f"artifacts/{name}").is_file() else 0 for name in ("candidate_frame_audit.jsonl","candidate_eligibility.jsonl","manifest.jsonl","place_clusters.jsonl")}
    passed=bool(code==0 and duration<=TIME_LIMIT_SECONDS and size<=DISK_LIMIT_BYTES and summary.get("overall_status")=="PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2" and summary.get("candidate_clusters")==27247 and summary.get("candidate_frames")==136235 and summary.get("selected_clusters")==22500 and summary.get("frames")==112500 and summary.get("junction_events_covered")==634 and summary.get("terminal_events_covered")==575 and summary.get("maximum_candidate_to_selected_same_tunnel_arc_distance_m",99)<=10 and len(shards)==90 and len(pages)==10 and len(spatial)==10 and len(distributions)==1 and counts=={"candidate_frame_audit.jsonl":136235,"candidate_eligibility.jsonl":27247,"manifest.jsonl":112500,"place_clusters.jsonl":22500})
    overall="PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2" if passed else "FAIL_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2"
    write_json(run/"metrics/runner_summary.json",{"overall_status":overall,"executor_exit_code":code,"duration_seconds":duration,"zarr_shards":len(shards),"sample_pages":len(pages),"spatial_figures":len(spatial),"distribution_figures":len(distributions),"jsonl_counts":counts,"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT_BYTES,"training_samples_consumed":0,"models":0})
    write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":"Formal V2 dataset only; zero training or benchmark reads."});sealed=_seal_manifest(run);print(json.dumps({"overall_status":overall,"sealed_files":sealed},indent=2));return 0 if passed else 2


if __name__=="__main__": raise SystemExit(main())
