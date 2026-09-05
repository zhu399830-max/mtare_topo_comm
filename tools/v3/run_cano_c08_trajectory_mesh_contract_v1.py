#!/usr/bin/env python3
"""Run and seal the approved Gate-4 C08 trajectory mesh contract."""

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

RUN_ID="gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0"
E1=Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python")
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_c08_trajectory_mesh_contract_v1.py"
SOURCE=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def seal(run_dir:Path)->int:
    destination=run_dir/"artifacts/evidence_sha256.txt"
    files=sorted(p for p in run_dir.rglob("*") if p.is_file() and p!=destination)
    destination.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in files),encoding="utf-8")
    return len(files)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args()
    spec=load_json(args.spec.resolve());run_dir=args.run_dir.resolve()
    if run_dir.name!=RUN_ID or load_json(run_dir/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("run identity/state mismatch")
    if spec.get("gate")!=4 or spec.get("operation")!="topology_replay" or spec.get("trajectory_contract_only") is not True:raise RuntimeError("scope mismatch")
    data_card=load_json(PROJECT_ROOT/spec["data_card"])
    if data_card.get("approval",{}).get("status")!="APPROVED" or "topology_replay" not in data_card["approval"].get("authorized_operations",[]):raise RuntimeError("data card is not approved")
    for relative,expected in spec["frozen_inputs"].items():
        path=PROJECT_ROOT/relative
        if sha(path)!=expected:raise RuntimeError(f"frozen input drift: {relative}")
    observed={}
    for name,item in spec["frozen_tools"].items():
        path=PROJECT_ROOT/item["path"]
        observed[name]=sha(path)
        if observed[name]!=item["sha256"]:raise RuntimeError(f"frozen tool drift: {name}")
    write_json(run_dir/"config/tool_hashes.json",observed)
    identity=json.loads(subprocess.check_output([str(E1),"-c","import json,sys,numpy,open3d,matplotlib;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'open3d':open3d.__version__,'matplotlib':matplotlib.__version__}))"],text=True))
    write_json(run_dir/"config/executor_environment_identity.json",identity)
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Zero-ray C08 trajectory geometry contract."})
    env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join((str(PROJECT_ROOT/"src"),str(PROJECT_ROOT/"tools/v3")));env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    argv=[str(E1),str(EXECUTOR),"--run-dir",str(run_dir)];started=time.monotonic();completed=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300,check=False);duration=time.monotonic()-started
    (run_dir/"logs/01_trajectory_mesh_contract.log").write_text(f"argv={json.dumps(argv)}\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n{completed.stdout}\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",encoding="utf-8")
    print(completed.stdout,end="")
    summary=load_json(run_dir/"metrics/summary.json") if (run_dir/"metrics/summary.json").is_file() else {}
    artifact_count=len(list((run_dir/"artifacts").glob("*.npz")))+len(list((run_dir/"artifacts").glob("*.json")))
    preview_count=len(list((run_dir/"previews").glob("*.png")))
    passed=bool(completed.returncode==0 and summary.get("overall_status")=="PASS_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1" and summary.get("world_count")==3 and summary.get("graph_edge_count")==312 and summary.get("traversal_count")==624 and summary.get("frame_count")==4773 and summary.get("maximum_connector_m",99)<=0.5 and summary.get("minimum_connector_mesh_surface_distance_m",0)>=0.8 and summary.get("ray_count")==0 and summary.get("inference_count")==0 and artifact_count==9 and preview_count==4)
    overall="PASS_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1" if passed else "FAIL_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1"
    write_json(run_dir/"metrics/runner_summary.json",{"schema_version":"cano_c08_trajectory_mesh_contract_runner_v1","overall_status":overall,"executor_exit_code":completed.returncode,"duration_seconds":duration,"artifact_count":artifact_count,"preview_count":preview_count,"result_bytes_before_seal":sum(p.stat().st_size for p in run_dir.rglob('*') if p.is_file()),"free_disk_bytes_after":shutil.disk_usage(PROJECT_ROOT).free,"claim_boundary":"Zero-ray perception-mesh trajectory geometry only; no LiDAR, inference, graph construction, C09/C10 or M-TARE."})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"note":"Formal C08 perception-mesh trajectory contract; not dynamic navigation qualification."})
    count=seal(run_dir);print(json.dumps({"overall_status":overall,"sealed_files":count},indent=2));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
