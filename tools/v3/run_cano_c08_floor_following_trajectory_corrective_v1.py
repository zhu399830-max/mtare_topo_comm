#!/usr/bin/env python3
"""Run and seal the approved complete C08 trajectory corrective audit."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json
RUN_ID="gate4_20260813_cano_c08_floor_following_trajectory_corrective_v1_seed0"
E1=Path("/tmp/mtare_cano_e1_zarr2187/bin/python")
EXECUTOR=PROJECT_ROOT/"tools/v3/execute_cano_c08_floor_following_trajectory_corrective_v1.py"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def seal(run):
 out=run/"artifacts/evidence_sha256.txt"; files=sorted(p for p in run.rglob("*") if p.is_file() and p!=out); out.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in files)); return len(files)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--spec",type=Path,required=True); p.add_argument("--run-dir",type=Path,required=True); a=p.parse_args(); spec=load_json(a.spec.resolve()); run=a.run_dir.resolve()
 if run.name!=RUN_ID or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED": raise RuntimeError("run identity/state mismatch")
 if spec.get("gate")!=4 or spec.get("complete_trajectory_corrective_only") is not True: raise RuntimeError("scope mismatch")
 card=load_json(PROJECT_ROOT/spec["data_card"])
 if card["approval"]["status"]!="APPROVED": raise RuntimeError("card not approved")
 for rel,expected in spec["frozen_inputs"].items():
  if sha(PROJECT_ROOT/rel)!=expected: raise RuntimeError(f"input drift {rel}")
 observed={}
 for name,item in spec["frozen_tools"].items():
  observed[name]=sha(PROJECT_ROOT/item["path"])
  if observed[name]!=item["sha256"]: raise RuntimeError(f"tool drift {name}")
 write_json(run/"config/tool_hashes.json",observed); write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
 env=os.environ.copy(); env["PYTHONPATH"]=os.pathsep.join((str(PROJECT_ROOT/"src"),str(PROJECT_ROOT/"tools/v3"))); env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
 argv=[str(E1),str(EXECUTOR),"--run-dir",str(run)]; started=time.monotonic(); done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300); duration=time.monotonic()-started
 (run/"logs/01_complete_trajectory_corrective.log").write_text(f"argv={json.dumps(argv)}\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n{done.stdout}\nduration_seconds={duration}\nexit_code={done.returncode}\n"); print(done.stdout,end="")
 s=load_json(run/"metrics/summary.json") if (run/"metrics/summary.json").exists() else {}; passed=done.returncode==0 and s.get("overall_status")=="PASS_CANO_C08_FLOOR_FOLLOWING_TRAJECTORY_CORRECTIVE_V1" and s.get("frames")==4773 and s.get("horizontal_rays")==3436560 and s.get("vertical_rays")==14319 and s.get("all_frames_passed") is True and s.get("inference_frames")==s.get("graph_updates")==s.get("training_samples_consumed")==0 and s.get("c09_worlds_read")==s.get("c10_worlds_read")==s.get("mtare_worlds_read")==0
 status="PASS_CANO_C08_FLOOR_FOLLOWING_TRAJECTORY_CORRECTIVE_V1" if passed else "FAIL_CANO_C08_FLOOR_FOLLOWING_TRAJECTORY_CORRECTIVE_V1"
 write_json(run/"metrics/runner_summary.json",{"schema_version":"cano_c08_floor_following_trajectory_corrective_runner_v1","overall_status":status,"executor_exit_code":done.returncode,"duration_seconds":duration,"frames":s.get("frames",0),"all_frames_passed":s.get("all_frames_passed"),"claim_boundary":"Corrected continuous trajectory qualification only; no inference, topology graph or planner result."})
 write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":status}); print(json.dumps({"overall_status":status,"sealed_files":seal(run)})); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
