#!/usr/bin/env python3
"""Run and seal the relation-endpoint episode-exposure audit."""

from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json

RUN_ID="gate3_20260828_gse_relation_endpoint_episode_exposure_audit_v1_seed0"; PASS="PASS_GSE_RELATION_ENDPOINT_EPISODE_EXPOSURE_AUDIT_V1"; PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def _verify(spec):
    out={}
    for rel,expected in spec["frozen_inputs"].items():
        actual=_sha(PROJECT_ROOT/rel)
        if actual!=expected: raise RuntimeError(f"episode exposure input drift: {rel}")
        out[rel]=actual
    return out
def _seal(run):
    seal=run/"artifacts/evidence_sha256.txt"; files=sorted(p for p in run.rglob("*") if p.is_file() and p!=seal)
    with seal.open("w",encoding="utf-8") as f:
        for p in files:f.write(f"{_sha(p)}  {p.relative_to(PROJECT_ROOT)}\n")
    return len(files)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--spec",required=True,type=Path);ap.add_argument("--run-dir",required=True,type=Path);a=ap.parse_args();run=a.run_dir.resolve();spec=load_json(a.spec.resolve())
    if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("episode exposure audit may execute only once")
    start=time.monotonic();overall="FAIL_SYSTEM_GSE_RELATION_ENDPOINT_EPISODE_EXPOSURE_AUDIT_V1";error=None;result={};before=after={};code=None
    try:
        card=load_json(PROJECT_ROOT/spec["data_card"])
        if card.get("status")!="APPROVED_FOR_ONE_IMMUTABLE_GSE_RELATION_ENDPOINT_EPISODE_EXPOSURE_AUDIT_V1":raise RuntimeError("episode exposure Data Card drift")
        for rec in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT/rec["path"])!=rec["sha256"]:raise RuntimeError(f"episode exposure tool drift: {rec['path']}")
        before=_verify(spec);write_json(run/"config/source_integrity_before.json",before);write_json(run/"config/environment.json",{"executable":str(PYTHON),"gpu_used":False});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
        cache=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0/scratch/action_set_cache"; endpoint=PROJECT_ROOT/"results/gate3_semantics/gate3_20260828_gse_development_relation_endpoint_support_audit_v1_seed0/artifacts/audit/endpoint_support_audit.jsonl"
        cmd=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/execute_gse_relation_endpoint_episode_exposure_audit_v1.py"),"--cache-dir",str(cache),"--endpoint-audit",str(endpoint),"--output-dir",str(run/"artifacts/audit")];(run/"config/command.txt").write_text(" ".join(cmd)+"\n",encoding="utf-8")
        env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
        with (run/"logs/audit.log").open("w",encoding="utf-8") as f:c=subprocess.run(cmd,cwd=PROJECT_ROOT,env=env,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=False)
        code=int(c.returncode)
        if code:raise RuntimeError(f"episode exposure executor failed with code {code}")
        result=load_json(run/"artifacts/audit/summary.json")
        if result.get("status")!=PASS or result.get("population",{}).get("relation_endpoints")!=98 or result.get("population",{}).get("relation_endpoint_episodes")!={"fit":286,"selection":110} or not result.get("gates",{}).get("identity_balanced_episode_corrective_justified") or any(result.get(k)!=0 for k in ("optimizer_steps","model_updates","model_inference_frames","c09_worlds_read","c10_worlds_read","mtare_worlds_read")):raise RuntimeError("episode exposure result drift")
        after=_verify(spec)
        if before!=after:raise RuntimeError("episode exposure sources changed")
        overall=PASS
    except Exception as exc:error=f"{type(exc).__name__}: {exc}";(run/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
    write_json(run/"metrics/summary.json",{"schema_version":"gse_relation_endpoint_episode_exposure_outer_v1","overall_status":overall,"error":error,"result":result,"executor_returncode":code,"duration_seconds":time.monotonic()-start,"source_unchanged":bool(before and before==after),"optimizer_steps":0,"model_updates":0,"model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if overall==PASS else "FAILED","overall_status":overall,"error":error});entries=_seal(run);print(json.dumps({"overall_status":overall,"error":error,"seal_entries":entries},indent=2));return 0 if overall==PASS else 2
if __name__=="__main__":raise SystemExit(main())
