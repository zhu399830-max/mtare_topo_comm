#!/usr/bin/env python3
"""Execute/seal the one proposal-distinctness corrective."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, time, traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

PASS="PASS_GSE_SPARSE_RELATION_PROPOSAL_DISTINCTNESS_CORRECTIVE_V1"; FAIL="FAIL_GSE_SPARSE_RELATION_PROPOSAL_DISTINCTNESS_CORRECTIVE_V1"
CARD_STATUS="APPROVED_FOR_ONE_IMMUTABLE_GSE_SPARSE_RELATION_PROPOSAL_DISTINCTNESS_CORRECTIVE_V1"
PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
READINESS=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
GEOMETRY=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_sparse_relation_geometry_shape_corrective_v1_seed0"

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def seal(run):
    target=run/"artifacts/evidence_sha256.txt"; files=sorted(p for p in run.rglob("*") if p.is_file() and p!=target)
    with target.open("w",encoding="utf-8") as stream:
        for path in files: stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--spec",required=True,type=Path); parser.add_argument("--run-dir",required=True,type=Path); args=parser.parse_args()
    spec=load_json(args.spec.resolve()); run=args.run_dir.resolve(); run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED": raise RuntimeError("proposal corrective executes exactly once")
    started=time.monotonic(); overall=FAIL; error=None; result={}; before={}; after={}; subprocesses=[]; peak=None
    try:
        card=load_json(PROJECT_ROOT/spec["data_card"]); validation=validate_data_card(card)
        if not validation.passed or card.get("status")!=CARD_STATUS: raise RuntimeError(f"proposal card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]: raise RuntimeError(f"tool drift: {record['path']}")
        for relative,expected in spec["frozen_inputs"].items():
            before[relative]=sha256(PROJECT_ROOT/relative)
            if before[relative]!=expected: raise RuntimeError(f"input drift: {relative}")
        versions=json.loads(subprocess.check_output([str(PYTHON),"-c","import json,numpy,scipy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__},sort_keys=True))"],text=True))
        expected={"python":"3.13.5","numpy":"2.1.3","scipy":"1.15.3","torch":"2.9.0+cu129","cuda":"12.9","zarr":"2.18.7"}
        if versions!=expected: raise RuntimeError(f"environment drift: {versions}")
        write_json(run/"config/environment.json",{"executable":str(PYTHON),"versions":versions,"deterministic_algorithms":True,"tf32":False,"cublas_workspace_config":":4096:8"}); write_json(run/"config/source_integrity_before.json",before); write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
        env=os.environ.copy(); env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3"); env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        with (run/"logs/00_unit_tests.log").open("w",encoding="utf-8") as stream:
            unit=subprocess.run([str(PYTHON),"-m","pytest","-q","tests/v3/unit/test_gse_sparse_circular_relation_transport.py","tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py"],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
        subprocesses.append({"stage":"unit_tests","returncode":unit.returncode})
        if unit.returncode: raise RuntimeError("proposal corrective unit tests failed")
        output=run/"metrics/corrective"; command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_sparse_relation_proposal_distinctness_corrective_v1.py"),"--dataset-root",str(DATASET/"artifacts/dataset/train"),"--sequence-manifest",str(DATASET/"artifacts/sequence_manifest.jsonl"),"--readiness-summary",str(READINESS/"metrics/summary.json"),"--geometry-corrective-summary",str(GEOMETRY/"metrics/summary.json"),"--output-dir",str(output)]
        log=run/"logs/01_corrective.log"
        with log.open("w",encoding="utf-8") as stream: completed=subprocess.run(["/usr/bin/time","-v",*command],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=1200,check=False)
        match=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",log.read_text()); peak=int(match.group(1)) if match else None; subprocesses.append({"stage":"corrective","returncode":completed.returncode,"peak_host_rss_kib":peak})
        if completed.returncode: raise RuntimeError("proposal corrective subprocess failed")
        result=load_json(output/"summary.json")
        if result.get("status")!=PASS or result.get("scientific_pass") is not True: raise RuntimeError("proposal corrective result failed")
        required=[output/"summary.json",output/"figure_source.json",output/"corrective_checks.csv",*[output/f"gse_sparse_relation_proposal_distinctness_corrective_v1.{s}" for s in ("png","pdf","svg")]]
        if any(not p.is_file() or not p.stat().st_size for p in required): raise RuntimeError("proposal evidence incomplete")
        after={relative:sha256(PROJECT_ROOT/relative) for relative in before}
        if before!=after: raise RuntimeError("frozen inputs changed")
        overall=PASS
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"; (run/"logs/failure_traceback.log").write_text(traceback.format_exc())
    write_json(run/"metrics/summary.json",{"schema_version":"gse_sparse_relation_proposal_distinctness_corrective_outer_v1","overall_status":overall,"scientific_pass":overall==PASS,"error":error,"subprocesses":subprocesses,"peak_host_rss_kib":peak,"result":result,"duration_seconds":time.monotonic()-started,"source_unchanged":bool(before and before==after),"optimizer_steps":0,"checkpoints_created":0,"c07_worlds_read":0,"c08_worlds_read":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}); write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if overall==PASS and error is None else "FAILED","overall_status":overall,"error":error})
    entries=seal(run); print(json.dumps({"overall_status":overall,"error":error,"decision":result.get("decision"),"seal_entries":entries},indent=2)); return 0 if overall==PASS and error is None else 2

if __name__=="__main__": raise SystemExit(main())
