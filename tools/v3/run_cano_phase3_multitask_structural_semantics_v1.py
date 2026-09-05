#!/usr/bin/env python3
"""Execute and seal the one approved Gate-2 B1/M1 three-seed training run."""

from __future__ import annotations

import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json

RUN_ID="gate2_20260812_cano_phase3_multitask_structural_semantics_v1r3_seed0"
PYTHON=Path("/tmp/mtare_phase3_torch290_zarr2187/bin/python")
TRAINER=PROJECT_ROOT/"tools/v3/train_phase3_multitask_structural_semantics.py"
DATASET=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0"
B0=PROJECT_ROOT/"results/gate1_data/phase3_pretraining_readonly_audits/b0_validation_metrics.json"
TIME_LIMIT_SECONDS=4*3600;DISK_LIMIT_BYTES=8*1024**3
ENVIRONMENT_PROBE_ATTEMPTS=2
ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS=2.0
ENVIRONMENT_PROBE_CODE="import json,torch,torchvision,zarr,numcodecs,numpy,sklearn;print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'cuda_available':torch.cuda.is_available()}))"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def verify_dataset_seal()->dict:
    lines=(DATASET/"artifacts/evidence_sha256.txt").read_text().splitlines();bad=[]
    for line in lines:
        expected,raw=line.split("  ",1);path=PROJECT_ROOT/raw
        if not path.is_file() or sha256(path)!=expected:bad.append(raw)
    return {"entries":len(lines),"mismatch_count":len(bad),"mismatches":bad}


def seal_manifest(run_dir:Path)->int:
    destination=run_dir/"artifacts/evidence_sha256.txt";files=sorted(path for path in run_dir.rglob("*") if path.is_file() and path!=destination)
    destination.write_text("".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),encoding="utf-8");return len(files)


def probe_environment_identity(run_dir:Path)->dict:
    """Run the frozen identity probe with bounded retry and complete diagnostics."""
    logs=run_dir/"logs";logs.mkdir(parents=True,exist_ok=True);attempts=[]
    for attempt in range(1,ENVIRONMENT_PROBE_ATTEMPTS+1):
        started=time.monotonic()
        result=subprocess.run([str(PYTHON),"-c",ENVIRONMENT_PROBE_CODE],env=os.environ.copy(),text=True,capture_output=True,check=False)
        record={"attempt":attempt,"return_code":result.returncode,"duration_seconds":time.monotonic()-started,"stdout_file":f"environment_probe_attempt_{attempt}.stdout.log","stderr_file":f"environment_probe_attempt_{attempt}.stderr.log"}
        (logs/record["stdout_file"]).write_text(result.stdout,encoding="utf-8")
        (logs/record["stderr_file"]).write_text(result.stderr,encoding="utf-8")
        attempts.append(record);write_json(logs/"environment_probe_attempts.json",{"schema_version":"phase3_environment_probe_attempts_v1","maximum_attempts":ENVIRONMENT_PROBE_ATTEMPTS,"retry_delay_seconds":ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS,"attempts":attempts})
        if result.returncode==0:
            try:return json.loads(result.stdout)
            except json.JSONDecodeError as exc:raise RuntimeError(f"environment identity probe returned invalid JSON on attempt {attempt}") from exc
        if attempt<ENVIRONMENT_PROBE_ATTEMPTS:time.sleep(ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"environment identity probe failed after {ENVIRONMENT_PROBE_ATTEMPTS} attempts; see {logs/'environment_probe_attempts.json'}")


def training_environment()->dict[str,str]:
    env=os.environ.copy()
    env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3")
    env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    return env


def plot_summary(b0:dict,runs:list[dict],path:Path)->None:
    m1=sorted((x for x in runs if x["mode"]=="M1"),key=lambda x:x["seed"]);b1=sorted((x for x in runs if x["mode"]=="B1"),key=lambda x:x["seed"])
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);seeds=[0,1,2];width=.25;x=np.arange(3)
    axes[0,0].axhline(b0["f1"],color="black",linestyle="--",label="B0 frozen")
    axes[0,0].bar(x-width,[r["direction_f1"] for r in b1],width,label="B1");axes[0,0].bar(x,[r["direction_f1"] for r in m1],width,label="M1");axes[0,0].set_xticks(x,seeds);axes[0,0].set_title("Validation direction F1");axes[0,0].legend()
    axes[0,1].bar(x-width/2,[r["role_macro_f1"] for r in m1],width,label="role macro-F1");axes[0,1].bar(x+width/2,[r["count_1_4_macro_f1"] for r in m1],width,label="count 1-4 macro-F1");axes[0,1].set_xticks(x,seeds);axes[0,1].set_title("M1 explicit semantics");axes[0,1].legend()
    axes[1,0].bar(x-width/2,[r["same_cluster_mean"] for r in m1],width,label="same-cluster mean");axes[1,0].bar(x+width/2,[r["same_cluster_p05"] for r in m1],width,label="same-cluster p05");axes[1,0].set_xticks(x,seeds);axes[1,0].set_title("M1 z_role five-view consistency");axes[1,0].legend()
    axes[1,1].bar(x,[r["masking_cosine_mean"] for r in m1],width,label="fixed masking cosine");axes[1,1].set_xticks(x,seeds);axes[1,1].set_title("M1 perturbation stability");axes[1,1].legend()
    fig.suptitle("Gate 2 — validation C09 only — B0/B1/M1 three-seed evidence");fig.savefig(path,dpi=150);plt.close(fig)


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--spec",type=Path,required=True);p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();spec=load_json(a.spec.resolve());run=a.run_dir.resolve();started=time.monotonic()
    if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=2 or spec.get("operation")!="training" or spec.get("user_authorization",{}).get("status")!="APPROVED":raise RuntimeError("approved Gate-2 scope mismatch")
    card=load_json(PROJECT_ROOT/spec["data_card"]);proposal=load_json(PROJECT_ROOT/spec["config_path"])
    if card.get("status")!="APPROVED_FOR_ONE_DETERMINISTIC_REPLACEMENT_EXECUTION_NOT_YET_EXECUTED" or card["approval"].get("authorized_operations")!=["training"] or card["approval"].get("authorized_gates")!=[2]:raise RuntimeError("approved deterministic replacement training card mismatch")
    if proposal.get("status")!="APPROVED_FOR_ONE_DETERMINISTIC_REPLACEMENT_EXECUTION_NOT_YET_EXECUTED":raise RuntimeError("approved deterministic replacement proposal mismatch")
    for name,record in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]:raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected:raise RuntimeError(f"frozen source mismatch: {raw}")
    seal=verify_dataset_seal()
    if seal["mismatch_count"]:raise RuntimeError(f"dataset seal mismatch: {seal}")
    identity=probe_environment_identity(run)
    expected={"torch":"2.9.0+cu129","torchvision":"0.24.0+cu129","cuda":"12.9","gpu":"NVIDIA GeForce RTX 5090 D","zarr":"2.18.7","numcodecs":"0.15.1","numpy":"2.1.3","sklearn":"1.6.1","cuda_available":True}
    if identity!=expected:raise RuntimeError(f"training environment mismatch: {identity}")
    write_json(run/"config/environment_identity.json",{**identity,"dataset_seal":seal,"host":platform.platform()});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"Approved B1/M1 seeds 0/1/2; C10/M-TARE/graph/planner reads zero."})
    env=training_environment();runs=[];exit_code=0
    for mode in ("B1","M1"):
        for seed in (0,1,2):
            child=run/f"artifacts/models/{mode.lower()}_seed{seed}";argv=[str(PYTHON),str(TRAINER),"--dataset-run",str(DATASET),"--output-dir",str(child),"--mode",mode,"--seed",str(seed),"--epochs","30","--batch-size","128","--learning-rate","0.0003","--weight-decay","0.0001","--patience","6","--workers","0"]
            child_started=time.monotonic();done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=max(1,TIME_LIMIT_SECONDS-int(time.monotonic()-started)),check=False);(run/f"logs/{mode.lower()}_seed{seed}.log").write_text(done.stdout+f"\nduration_seconds={time.monotonic()-child_started:.6f}\nexit_code={done.returncode}\n");print(done.stdout,end="",flush=True)
            if done.returncode!=0:exit_code=done.returncode;break
            summary=load_json(child/"summary.json");validation=summary["best_validation"];stability=summary["stability"];runs.append({"mode":mode,"seed":seed,"direction_f1":validation["direction"]["f1"],"direction_precision":validation["direction"]["precision"],"direction_recall":validation["direction"]["recall"],"angular_error_deg":validation["direction"]["mean_matched_angular_error_deg"],"role_macro_f1":validation["role"]["macro_f1_present"],"role_recall":validation["role"]["recall"],"count_1_4_macro_f1":validation["count"]["macro_f1_count_1_to_4"],"same_cluster_mean":validation["representation"]["same_cluster_cosine"]["mean"],"same_cluster_p05":validation["representation"]["same_cluster_cosine"]["p05"],"masking_cosine_mean":stability["fixed_masking_z_role_cosine_mean"],"rotation_direction_max_error":stability["rotation_direction_max_absolute_logit_error"],"rotation_z_min_cosine":stability["rotation_z_role_minimum_cosine"],"epochs":summary["epochs_completed"],"best_epoch":summary["best_epoch"]})
        if exit_code:break
    b0=load_json(B0);m1=[x for x in runs if x["mode"]=="M1"]
    integrity=exit_code==0 and len(runs)==6 and len(m1)==3 and time.monotonic()-started<=TIME_LIMIT_SECONDS
    if integrity:
        direction_median=float(np.median([x["direction_f1"] for x in m1]));direction_floor=min(x["direction_f1"] for x in m1);role_median=float(np.median([x["role_macro_f1"] for x in m1]));role_recall_medians=np.median(np.asarray([x["role_recall"] for x in m1]),axis=0).tolist();count_median=float(np.median([x["count_1_4_macro_f1"] for x in m1]));same_mean=float(np.median([x["same_cluster_mean"] for x in m1]));same_p05=float(np.median([x["same_cluster_p05"] for x in m1]));masking=float(np.median([x["masking_cosine_mean"] for x in m1]));rotation_error=max(x["rotation_direction_max_error"] for x in m1);rotation_cos=min(x["rotation_z_min_cosine"] for x in m1)
        direction_pass=direction_median>=b0["f1"] and direction_floor>=.75;aux_pass=role_median>=.70 and min(role_recall_medians)>=.60 and count_median>=.70 and same_mean>=.90 and same_p05>=.75 and masking>=.85 and rotation_error<=2e-5 and rotation_cos>=.999
        gate_result="GATE_PASS" if direction_pass and aux_pass else "GATE_MIXED" if direction_pass else "GATE_FAIL"
        aggregate={"direction_median_f1":direction_median,"direction_seed_floor_f1":direction_floor,"frozen_b0_f1":b0["f1"],"role_median_macro_f1":role_median,"role_recall_medians":role_recall_medians,"count_1_4_median_macro_f1":count_median,"same_cluster_median_mean_cosine":same_mean,"same_cluster_median_p05_cosine":same_p05,"masking_median_mean_cosine":masking,"rotation_max_logit_error":rotation_error,"rotation_min_z_cosine":rotation_cos,"direction_pass":direction_pass,"auxiliary_representation_pass":aux_pass}
    else:gate_result="GATE_FAIL";aggregate={}
    (run/"previews").mkdir(exist_ok=True)
    if len(runs)==6:plot_summary(b0,runs,run/"previews/three_seed_summary.png")
    result_size=sum(path.stat().st_size for path in run.rglob("*") if path.is_file());passed=integrity and result_size<=DISK_LIMIT_BYTES
    overall="PASS_CANO_PHASE3_MULTITASK_TRAINING_EXECUTION" if passed else "FAIL_CANO_PHASE3_MULTITASK_TRAINING_EXECUTION"
    summary={"schema_version":"cano_phase3_multitask_training_summary_v1","overall_status":overall,"gate_result":gate_result,"integrity_passed":integrity,"b0":b0,"runs":runs,"aggregate":aggregate,"duration_seconds":time.monotonic()-started,"result_bytes_before_seal":result_size,"disk_limit_bytes":DISK_LIMIT_BYTES,"train_worlds":80,"validation_worlds":10,"strict_test_worlds_read":0,"mtare_worlds_read":0,"graph_runs":0,"planner_changes":0}
    write_json(run/"metrics/summary.json",summary);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"gate_result":gate_result,"note":"B1/M1 training only; no C10/M-TARE/graph/planner."});sealed=seal_manifest(run);print(json.dumps({"overall_status":overall,"gate_result":gate_result,"sealed_files":sealed},indent=2));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
