#!/usr/bin/env python3
"""Execute and seal the one approved Gate-2 masking corrective run."""

from __future__ import annotations

import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,write_json

RUN_ID="gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r2_seed0"
PYTHON=Path("/tmp/mtare_phase3_torch290_zarr2187/bin/python");TRAINER=PROJECT_ROOT/"tools/v3/train_phase3_multitask_ray_column_dropout.py";SMOKE=PROJECT_ROOT/"tools/v3/smoke_phase3_ray_column_dropout.py"
DATASET=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0";SOURCE=PROJECT_ROOT/"results/gate2_representation/gate2_20260812_cano_phase3_multitask_structural_semantics_v1r3_seed0";AUDIT=PROJECT_ROOT/"results/gate2_representation/gate2_20260812_cano_phase3_stability_contract_audit_v1_seed0";B0=PROJECT_ROOT/"results/gate1_data/phase3_pretraining_readonly_audits/b0_validation_metrics.json"
TIME_LIMIT=3*3600;DISK_LIMIT=4*1024**3
ENVIRONMENT_PROBE_ATTEMPTS=2;ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS=2.0


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


def probe_environment_identity(run:Path,probe:str)->dict:
    attempts=[]
    for attempt in range(1,ENVIRONMENT_PROBE_ATTEMPTS+1):
        started=time.monotonic();result=subprocess.run([str(PYTHON),"-c",probe],text=True,capture_output=True,check=False)
        record={"attempt":attempt,"return_code":result.returncode,"duration_seconds":time.monotonic()-started,"stdout_file":f"environment_probe_attempt_{attempt}.stdout.log","stderr_file":f"environment_probe_attempt_{attempt}.stderr.log"}
        (run/"logs"/record["stdout_file"]).write_text(result.stdout,encoding="utf-8");(run/"logs"/record["stderr_file"]).write_text(result.stderr,encoding="utf-8");attempts.append(record)
        write_json(run/"logs/environment_probe_attempts.json",{"schema_version":"phase3_environment_probe_attempts_v1","maximum_attempts":ENVIRONMENT_PROBE_ATTEMPTS,"retry_delay_seconds":ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS,"attempts":attempts})
        if result.returncode==0:
            try:return json.loads(result.stdout)
            except json.JSONDecodeError as exc:raise RuntimeError(f"environment probe returned invalid JSON on attempt {attempt}") from exc
        if attempt<ENVIRONMENT_PROBE_ATTEMPTS:time.sleep(ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"environment identity probe failed after {ENVIRONMENT_PROBE_ATTEMPTS} attempts")


def plot_comparison(old:list[dict],new:list[dict],path:Path)->None:
    labels=["direction F1","role F1","count 1-4 F1","same p05","mask cosine"];old_values=[np.median([r["direction_f1"] for r in old]),np.median([r["role_macro_f1"] for r in old]),np.median([r["count_1_4_macro_f1"] for r in old]),np.median([r["same_cluster_p05"] for r in old]),np.median([r["masking_cosine_mean"] for r in old])];new_values=[np.median([r["direction_f1"] for r in new]),np.median([r["role_macro_f1"] for r in new]),np.median([r["count_1_4_macro_f1"] for r in new]),np.median([r["same_cluster_p05"] for r in new]),np.median([r["masking_cosine_mean_cpu"] for r in new])]
    x=np.arange(len(labels));width=.36;fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);ax.bar(x-width/2,old_values,width,label="sealed v1r3 M1");ax.bar(x+width/2,new_values,width,label="M1D corrective");ax.axhline(.85,color="black",linestyle="--",alpha=.6,label="mask threshold 0.85");ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,1.05);ax.set_title("Gate 2 masking corrective — clean metrics and frozen perturbation");ax.legend();fig.savefig(path,dpi=150);plt.close(fig)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();spec=load_json(args.spec.resolve());run=args.run_dir.resolve();started=time.monotonic()
    if run.name!=RUN_ID or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED":raise RuntimeError("one-time run state mismatch")
    if spec.get("gate")!=2 or spec.get("operation")!="training" or spec.get("seed")!=0 or spec.get("user_authorization",{}).get("status")!="APPROVED":raise RuntimeError("approved corrective scope mismatch")
    card=load_json(PROJECT_ROOT/spec["data_card"]);proposal=load_json(PROJECT_ROOT/spec["config_path"])
    expected_status="APPROVED_FOR_ONE_MASKING_CORRECTIVE_EXECUTION_NOT_YET_EXECUTED"
    if card.get("status")!=expected_status or proposal.get("status")!=expected_status or card.get("approval",{}).get("authorized_operations")!=["training"] or card.get("approval",{}).get("authorized_gates")!=[2]:raise RuntimeError("corrective card/proposal mismatch")
    for name,item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT/item["path"])!=item["sha256"]:raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw,expected in spec["frozen_source_files"].items():
        if sha256(PROJECT_ROOT/raw)!=expected:raise RuntimeError(f"frozen source mismatch: {raw}")
    dataset_seal=verify_seal(DATASET/"artifacts/evidence_sha256.txt",19338);source_seal=verify_seal(SOURCE/"artifacts/evidence_sha256.txt",62);audit_seal=verify_seal(AUDIT/"artifacts/evidence_sha256.txt",13)
    if any(item["mismatch_count"] or item["entries"]!=item["expected_entries"] for item in (dataset_seal,source_seal,audit_seal)):raise RuntimeError("source seal verification failed")
    probe="import json,torch,zarr,numcodecs,numpy,sklearn;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'cuda_available':torch.cuda.is_available()}))";identity=probe_environment_identity(run,probe)
    expected={"torch":"2.9.0+cu129","cuda":"12.9","gpu":"NVIDIA GeForce RTX 5090 D","zarr":"2.18.7","numcodecs":"0.15.1","numpy":"2.1.3","sklearn":"1.6.1","cuda_available":True}
    if identity!=expected:raise RuntimeError(f"environment mismatch: {identity}")
    write_json(run/"config/environment_identity.json",{**identity,"host":platform.platform(),"dataset_seal":dataset_seal,"source_run_seal":source_seal,"stability_audit_seal":audit_seal});write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING","note":"M1D seeds 0/1/2 train-only ray-column dropout; zero C10/M-TARE/graph/planner."})
    env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3");env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    smoke=subprocess.run([str(PYTHON),str(SMOKE)],cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False);(run/"logs/00_no_update_smoke.log").write_text(smoke.stdout,encoding="utf-8")
    if smoke.returncode!=0 or "PASS_PHASE3_RAY_COLUMN_DROPOUT_NO_UPDATE_SMOKE" not in smoke.stdout:raise RuntimeError("no-update corrective smoke failed")
    runs=[];exit_code=0
    for seed in (0,1,2):
        child=run/f"artifacts/models/m1d_seed{seed}";argv=[str(PYTHON),str(TRAINER),"--dataset-run",str(DATASET),"--output-dir",str(child),"--mode","M1D","--seed",str(seed),"--epochs","30","--batch-size","128","--learning-rate","0.0003","--weight-decay","0.0001","--patience","6","--workers","0"]
        remaining=max(1,TIME_LIMIT-int(time.monotonic()-started));child_started=time.monotonic();done=subprocess.run(argv,cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remaining,check=False);(run/f"logs/m1d_seed{seed}.log").write_text(done.stdout+f"\nduration_seconds={time.monotonic()-child_started:.6f}\nexit_code={done.returncode}\n",encoding="utf-8");print(done.stdout,end="",flush=True)
        if done.returncode!=0 or "does not have a deterministic implementation" in done.stdout:exit_code=done.returncode or 3;break
        summary=load_json(child/"summary.json");validation=summary["best_validation"];cpu=summary["stability"]["cpu"];cuda=summary["stability"]["cuda"];augmentation=summary["augmentation"]
        runs.append({"mode":"M1D","seed":seed,"direction_f1":validation["direction"]["f1"],"angular_error_deg":validation["direction"]["mean_matched_angular_error_deg"],"role_macro_f1":validation["role"]["macro_f1_present"],"role_recall":validation["role"]["recall"],"count_1_4_macro_f1":validation["count"]["macro_f1_count_1_to_4"],"same_cluster_mean":validation["representation"]["same_cluster_cosine"]["mean"],"same_cluster_p05":validation["representation"]["same_cluster_cosine"]["p05"],"masking_cosine_mean_cpu":cpu["fixed_masking_z_role_cosine_mean"],"masking_cosine_p05_cpu":cpu["fixed_masking_z_role_cosine_p05"],"masking_cosine_mean_cuda":cuda["fixed_masking_z_role_cosine_mean"],"rotation_direction_max_error_cpu":cpu["rotation_direction_max_absolute_logit_error"],"rotation_direction_max_error_cuda":cuda["rotation_direction_max_absolute_logit_error"],"rotation_z_min_cosine_cpu":cpu["rotation_z_role_minimum_cosine"],"augmentation":augmentation,"epochs":summary["epochs_completed"],"best_epoch":summary["best_epoch"]})
    integrity=exit_code==0 and len(runs)==3 and time.monotonic()-started<=TIME_LIMIT and all(r["augmentation"]["scope"]=="train_only" and r["augmentation"]["samples"]==100000*r["epochs"] for r in runs)
    b0=load_json(B0);old=load_json(SOURCE/"metrics/summary.json")["runs"];old_m1=[item for item in old if item["mode"]=="M1"]
    if integrity:
        direction_median=float(np.median([r["direction_f1"] for r in runs]));direction_floor=min(r["direction_f1"] for r in runs);role_median=float(np.median([r["role_macro_f1"] for r in runs]));role_recall_medians=np.median(np.asarray([r["role_recall"] for r in runs]),axis=0).tolist();count_median=float(np.median([r["count_1_4_macro_f1"] for r in runs]));same_mean=float(np.median([r["same_cluster_mean"] for r in runs]));same_p05=float(np.median([r["same_cluster_p05"] for r in runs]));masking=float(np.median([r["masking_cosine_mean_cpu"] for r in runs]));rotation_error=max(r["rotation_direction_max_error_cpu"] for r in runs);rotation_cos=min(r["rotation_z_min_cosine_cpu"] for r in runs)
        direction_pass=direction_median>=b0["f1"] and direction_floor>=.75;aux_pass=role_median>=.70 and min(role_recall_medians)>=.60 and count_median>=.70 and same_mean>=.90 and same_p05>=.75 and masking>=.85 and rotation_error<=2e-5 and rotation_cos>=.999;gate_result="GATE_PASS" if direction_pass and aux_pass else "GATE_MIXED" if direction_pass else "GATE_FAIL";aggregate={"direction_median_f1":direction_median,"direction_seed_floor_f1":direction_floor,"frozen_b0_f1":b0["f1"],"role_median_macro_f1":role_median,"role_recall_medians":role_recall_medians,"count_1_4_median_macro_f1":count_median,"same_cluster_median_mean_cosine":same_mean,"same_cluster_median_p05_cosine":same_p05,"masking_median_mean_cosine_cpu":masking,"rotation_max_logit_error_cpu":rotation_error,"rotation_min_z_cosine_cpu":rotation_cos,"direction_pass":direction_pass,"auxiliary_representation_pass":aux_pass}
    else:gate_result="GATE_FAIL";aggregate={}
    if len(runs)==3:plot_comparison(old_m1,runs,run/"previews/v1r3_vs_m1d_corrective.png")
    size=sum(path.stat().st_size for path in run.rglob("*") if path.is_file());passed=integrity and size<=DISK_LIMIT;overall="PASS_CANO_PHASE3_MASKING_CORRECTIVE_EXECUTION" if passed else "FAIL_CANO_PHASE3_MASKING_CORRECTIVE_EXECUTION"
    summary={"schema_version":"cano_phase3_masking_corrective_summary_v1","overall_status":overall,"gate_result":gate_result,"integrity_passed":integrity,"runs":runs,"aggregate":aggregate,"duration_seconds":time.monotonic()-started,"result_bytes_before_seal":size,"disk_limit_bytes":DISK_LIMIT,"unique_train_frames":100000,"validation_frames":12500,"strict_test_frames_read":0,"mtare_frames_read":0,"graph_runs":0,"planner_changes":0}
    write_json(run/"metrics/summary.json",summary);write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall,"gate_result":gate_result,"note":"M1D masking corrective only; no C10/M-TARE/graph/planner."});sealed=seal_manifest(run);print(json.dumps({"overall_status":overall,"gate_result":gate_result,"sealed_files":sealed},indent=2));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
