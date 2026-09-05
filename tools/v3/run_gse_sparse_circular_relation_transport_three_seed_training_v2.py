#!/usr/bin/env python3
"""Execute/seal one sparse relation transport V2 three-seed run."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, time, traceback
from pathlib import Path
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

PASS="PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2"; FAIL="FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2"; INNER_PASS="PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_SELECTION_V2"
CARD_STATUS="APPROVED_FOR_ONE_IMMUTABLE_GSE_SPARSE_RELATION_TRANSPORT_THREE_SEED_TRAINING_V2"
PYTHON=Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
DATASET=PROJECT_ROOT/"results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
SLOT=PROJECT_ROOT/"results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def peak_rss(path):
    match=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",path.read_text()); return int(match.group(1)) if match else None
def execute(command,log,environment,timeout):
    with log.open("w",encoding="utf-8") as stream: completed=subprocess.run(["/usr/bin/time","-v",*command],cwd=PROJECT_ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=timeout,check=False)
    return completed.returncode,peak_rss(log)
def verify_seal(path):
    count=0
    for line in path.read_text().splitlines():
        expected,relative=line.split("  ",1); target=PROJECT_ROOT/relative
        if not target.is_file() or sha256(target)!=expected: raise RuntimeError(f"sealed source drift: {relative}")
        count+=1
    return count
def seal(run):
    target=run/"artifacts/evidence_sha256.txt"; files=sorted(p for p in run.rglob("*") if p.is_file() and p!=target)
    with target.open("w",encoding="utf-8") as stream:
        for path in files: stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--spec",required=True,type=Path); parser.add_argument("--run-dir",required=True,type=Path); args=parser.parse_args(); spec=load_json(args.spec.resolve()); run=args.run_dir.resolve(); run_id=f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name!=run_id or load_json(run/"RUN_STATE.json").get("state")!="CREATED_NOT_EXECUTED": raise RuntimeError("sparse relation training executes exactly once")
    started=time.monotonic(); overall=FAIL; error=None; selection={}; before={}; after={}; processes=[]; commands=[]; verified=0
    try:
        card=load_json(PROJECT_ROOT/spec["data_card"]); validation=validate_data_card(card)
        if not validation.passed or card.get("status")!=CARD_STATUS: raise RuntimeError(f"training card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT/record["path"])!=record["sha256"]: raise RuntimeError(f"tool drift: {record['path']}")
        for relative,expected in spec["frozen_inputs"].items():
            before[relative]=sha256(PROJECT_ROOT/relative)
            if before[relative]!=expected: raise RuntimeError(f"input drift: {relative}")
        verified=verify_seal(DATASET/"artifacts/evidence_sha256.txt")
        versions=json.loads(subprocess.check_output([str(PYTHON),"-c","import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"],text=True)); expected={"python":"3.13.5","numpy":"2.1.3","torch":"2.9.0+cu129","cuda":"12.9","zarr":"2.18.7","gpu":"NVIDIA GeForce RTX 5090 D"}
        if versions!=expected: raise RuntimeError(f"environment drift: {versions}")
        write_json(run/"config/environment.json",{"executable":str(PYTHON),"versions":versions,"deterministic_algorithms":True,"tf32":False,"cublas_workspace_config":":4096:8"}); write_json(run/"config/source_integrity_before.json",before); write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"RUNNING"})
        env=os.environ.copy(); env.update({"PYTHONPATH":str(PROJECT_ROOT/"src")+os.pathsep+str(PROJECT_ROOT/"tools/v3"),"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"})
        tests=["tests/v3/unit/test_gse_sparse_circular_relation_transport.py","tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py","tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py","tests/v3/unit/test_evaluate_gse_sparse_circular_relation_transport_v2.py"]
        with (run/"logs/00_unit_tests.log").open("w",encoding="utf-8") as stream: unit=subprocess.run([str(PYTHON),"-m","pytest","-q",*tests],cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600,check=False)
        processes.append({"stage":"unit_tests","returncode":unit.returncode})
        if unit.returncode: raise RuntimeError("sparse relation unit tests failed")
        models=run/"artifacts/models"; models.mkdir(parents=True)
        for seed in range(3):
            command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/train_gse_sparse_circular_relation_transport_v2.py"),"--dataset-root",str(DATASET/"artifacts/dataset/train"),"--sequence-manifest",str(DATASET/"artifacts/sequence_manifest.jsonl"),"--predecessor-checkpoint",str(SLOT/f"artifacts/models/seed{seed}/best.pt"),"--output-dir",str(models/f"seed{seed}"),"--seed",str(seed),"--epochs","10","--batch-size","128","--evaluation-batch-size","256","--learning-rate","0.0003","--weight-decay","0.0001"]
            commands.append(command); code,rss=execute(command,run/f"logs/{seed+1:02d}_seed{seed}_training.log",env,43200); processes.append({"stage":"training","seed":seed,"returncode":code,"peak_host_rss_kib":rss})
            if code: raise RuntimeError(f"seed{seed} sparse relation training failed")
            summary=load_json(models/f"seed{seed}/summary.json")
            if summary.get("schema_version")!="gse_sparse_circular_relation_transport_training_seed_v2" or summary.get("parameters")!=784513 or summary.get("optimizer_steps")!=12230 or summary.get("core_optimizer_steps")!=11370 or summary.get("descriptor_optimizer_steps")!=860 or summary.get("c08_checkpoint_observations")!=0 or summary.get("development_output_observations")!=45942 or summary.get("predecessor_backbone",{}).get("source_seed")!=seed: raise RuntimeError(f"seed{seed} population/backbone drift")
            if summary.get("peak_gpu_memory_bytes",0)>16*1024**3 or summary.get("peak_nvidia_process_memory_bytes",0)>16*1024**3 or rss is None or rss>16*1024**2: raise RuntimeError(f"seed{seed} resource contract failed")
        evaluation=run/"metrics/selection"; command=[str(PYTHON),str(PROJECT_ROOT/"tools/v3/evaluate_gse_sparse_circular_relation_transport_v2.py"),"--dataset-root",str(DATASET/"artifacts/dataset/train"),"--sequence-manifest",str(DATASET/"artifacts/sequence_manifest.jsonl"),"--association-pairs",str(DATASET/"artifacts/association_pairs_numeric.jsonl"),"--output-dir",str(evaluation)]
        for seed in range(3): command.extend(("--prediction-root",str(models/f"seed{seed}/development_predictions")))
        commands.append(command); write_json(run/"config/commands.json",commands); code,rss=execute(command,run/"logs/04_selection.log",env,14400); processes.append({"stage":"selection","returncode":code,"peak_host_rss_kib":rss})
        if code not in (0,2): raise RuntimeError("sparse relation evaluator program failure")
        selection=load_json(evaluation/"summary.json"); passed=selection.get("status")==INNER_PASS and selection.get("scientific_pass") is True
        if (code==0)!=passed: raise RuntimeError("selection status/return mismatch")
        parents=sorted(path.stem for path in (DATASET/"artifacts/dataset/train").glob("*_C0[78].zarr")); required=[*[models/f"seed{seed}/{name}" for seed in range(3) for name in ("best.pt","history.json","summary.json")],*[models/f"seed{seed}/development_predictions/{parent}.npz" for seed in range(3) for parent in parents],evaluation/"summary.json",evaluation/"per_world_metrics.csv",evaluation/"figure_source.json",*[evaluation/f"gse_sparse_circular_relation_transport_selection_v2.{suffix}" for suffix in ("png","pdf","svg")]]
        if any(not path.is_file() or not path.stat().st_size for path in required): raise RuntimeError("sparse relation training evidence incomplete")
        after={relative:sha256(PROJECT_ROOT/relative) for relative in before}
        if before!=after: raise RuntimeError("frozen sparse relation source changed")
        overall=PASS if passed else FAIL
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"; (run/"logs/failure_traceback.log").write_text(traceback.format_exc())
    successful=sum(p.get("stage")=="training" and p.get("returncode")==0 for p in processes)
    write_json(run/"metrics/summary.json",{"schema_version":"gse_sparse_circular_relation_transport_three_seed_training_outer_v2","overall_status":overall,"scientific_pass":overall==PASS,"error":error,"subprocesses":processes,"selection":selection,"duration_seconds":time.monotonic()-started,"source_unchanged":bool(before and before==after),"verified_source_seal_entries":verified,"optimizer_steps":successful*12230,"core_optimizer_steps":successful*11370,"descriptor_optimizer_steps":successful*860,"checkpoint_selection_forward_observations":successful*215480,"final_inference_observations":successful*45942,"c08_checkpoint_observations":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}); write_json(run/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":run_id,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error})
    entries=seal(run); print(json.dumps({"overall_status":overall,"error":error,"optimizer_steps":successful*12230,"decision":selection.get("decision"),"seal_entries":entries},indent=2)); return 0 if overall==PASS and error is None else 2
if __name__=="__main__": raise SystemExit(main())
