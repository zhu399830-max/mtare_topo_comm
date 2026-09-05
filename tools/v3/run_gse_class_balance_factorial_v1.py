#!/usr/bin/env python3
"""Paired 2x2 class-contribution experiment; reuse 00, never select a best run."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_partial_training_inputs import load_training_inputs
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_class_balance_factorial import validate_class_balance_factorial_card
from mtare_topo.representation.gse_class_balance_training import train_class_balanced_structure
from mtare_topo.representation.gse_partial_structure_training import BRANCHES, PartialStructureExample, PartialTrainingConfig
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionQueryHead
from run_gse_partial_structure_training_v1 import evaluate_and_save, slice_target
from run_gse_geometry_bound_rank_v1 import analyze_and_save as analyze_rank
from run_gse_supported_construction_teacher_v1 import sha


FACTORS = ("10", "01", "11")


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("path outside project")
    return path


def load_inputs(root, card):
    """Six bound payloads. Old weights are never opened, even for SHA checks."""
    baseline = {}
    reads = {}
    for key in ("baseline_geometry_summary", "baseline_rank_summary"):
        source = card["sources"][key]
        path = (Path(root) / source["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or sha(path) != source["sha256"]:
            raise ValueError("baseline source drift: " + key)
        baseline[key] = load_json(path)
        reads[source["path"]] = source["sha256"]
    data = load_training_inputs(root, card["base_card"]["base_training_card"])
    reads.update(data["read_hashes"])
    expected = {s["path"]:s["sha256"] for s in card["sources"].values()}
    if reads != expected:
        raise ValueError("six exact payloads required")
    geometry = baseline["baseline_geometry_summary"]
    rank = baseline["baseline_rank_summary"]
    if (geometry.get("status") != "GEOMETRY_BOUND_FIXED_BUDGET_COMPLETE"
            or rank.get("status") != "GEOMETRY_BOUND_RANK_DIAGNOSTIC_COMPLETE"
            or any(x.get("error") is not None or x.get("scientific_gate_pass") is not False for x in (geometry, rank))):
        raise ValueError("completed, non-scientific baseline statuses required")
    g, r = geometry["result"], rank["result"]
    for key,value in (("optimizer_steps",900),("head_inference_windows",1080),("backbone_windows",0),("new_sensor_frames",0)):
        if type(g.get(key)) is not int or g[key] != value:
            raise ValueError("baseline count drift: " + key)
    if g.get("same_initial_state_verified") is not True or g.get("same_schedule_verified") is not True:
        raise ValueError("baseline pairing evidence missing")
    if r.get("original_scores_reproduced") is not True or r.get("cached_prediction_observations") != 540:
        raise ValueError("baseline rank reproduction missing")
    for key in ("model_inference", "checkpoint_reads", "optimizer_steps"):
        if type(r.get(key)) is not int or r[key] != 0:
            raise ValueError("baseline rank may not train")
    return data, geometry, rank, reads


def rank_saved_final(run, data, outputs):
    """Reload only OUR generated predictions, not historical NPZ or weights."""
    predictions, saved = {}, {}
    for branch in BRANCHES:
        with np.load(run / "artifacts" / (branch + "__final__predictions.npz"), allow_pickle=False) as archive:
            arrays = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
        predictions[branch] = RegionPrediction(**{f.name: arrays[f.name] for f in fields(RegionPrediction)})
        saved[branch] = {key:arrays[key] for key in ("unique_center_query", "scored_member_mask")}
    return analyze_rank(run, {"data":data,"predictions":predictions,"saved_scoring":saved,
        "training_summary":{"result":{"evaluation":outputs}}})


def run_factor(run, data, settings, evaluation, factor, member_counts, event_counts,
               baseline_initial, baseline_initial_sha, *, step_callback=None):
    if factor not in FACTORS:
        raise ValueError("only new factors10/01/11 may train")
    config = PartialTrainingConfig(settings["seed"],settings["steps_per_branch"],settings["batch_size"],settings["lr"],settings["device"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        initial = RegionQueryHead(hidden=settings["hidden"])
    if sum(p.numel() for p in initial.parameters()) != settings["parameters_per_head"]:
        raise ValueError("parameter count drift")
    initial_state = {k:v.detach().clone() for k,v in initial.state_dict().items()}
    initial_path = run / "artifacts/shared_initial_state.pt"
    torch.save(initial_state, initial_path)
    if sha(initial_path) != baseline_initial_sha:
        raise ValueError("shared initial state serialization differs from sealed baseline")
    output = {stage:{} for stage in ("initial","final")}
    for branch in BRANCHES:
        initial.use_relations = settings["use_relations"][branch]
        output["initial"][branch] = evaluate_and_save(run,branch,"initial",initial.to(config.device),data,config,evaluation["member_threshold"])
        if output["initial"][branch] != baseline_initial[branch]:
            raise ValueError("initial independent scores differ from baseline: " + branch)
    del initial
    examples = {key:[PartialStructureExample(data["axes"][key][i:i+1],slice_target(data["targets"][key],slice(i,i+1)))
        for i in range(len(data["manifest"]))] for key in ("gt","predicted")}
    trained = train_class_balanced_structure({"gt_axes":examples["gt"],"predicted_axes":examples["predicted"],
        "predicted_no_relations":examples["predicted"]},config,hidden=settings["hidden"],
        member_class_counts=tuple(member_counts) if factor[0]=="1" else None,
        event_class_counts=tuple(event_counts) if factor[1]=="1" else None,on_step=step_callback)
    if not all(torch.equal(trained.initial_state[k],v) for k,v in initial_state.items()):
        raise ValueError("training initial state mismatch")
    totals = {}
    for branch in BRANCHES:
        history = trained.history[branch]
        if len(history)!=config.steps or [r["sample_indices"] for r in history]!=trained.schedule:
            raise ValueError("paired budget/schedule drift")
        totals[branch] = {group:{k:sum(r[group][k] for r in history) for k in history[0][group]}
            for group in ("counts","geometry_counts","supervision_counts")}
        torch.save({"state_dict":{k:v.detach().cpu() for k,v in trained.heads[branch].state_dict().items()},
            "factor":factor,"branch":branch,"settings":settings,"step":config.steps,
            "member_class_counts":member_counts if factor[0]=="1" else None,
            "event_class_counts":event_counts if factor[1]=="1" else None},run/"artifacts"/(branch+"__final.pt"))
        output["final"][branch] = evaluate_and_save(run,branch,"final",trained.heads[branch],data,config,evaluation["member_threshold"])
    write_json(run/"artifacts/shared_schedule.json",trained.schedule)
    write_json(run/"artifacts/training_history.json",trained.history)
    write_json(run/"artifacts/manifest.json",data["manifest"])
    del trained
    ranks = rank_saved_final(run,data,output)
    return {"evaluation":output,"rank":ranks,"supervision_totals":totals,
        "optimizer_steps":len(BRANCHES)*config.steps,"head_inference_windows":2*len(BRANCHES)*len(data["manifest"]),
        "same_initial_state_verified":True,"same_schedule_verified":True,"baseline_initial_sha_verified":True}


def metric_vector(aggregate, rank):
    return {"center_mean_m":aggregate["unique_assigned_distance"]["mean_m"],
        "member_f1":aggregate["members"]["known_scored_f1"],
        "event_macro_f1":aggregate["events"]["two_class_macro_f1"],
        "member_ap":rank["members"]["average_precision"],"member_auc":rank["members"]["auroc"],
        "junction_ap":rank["junction"]["average_precision"],"junction_auc":rank["junction"]["auroc"]}


def factorial_effects(factors):
    """Report all paired contrasts, undefined populations remain undefined."""
    output = {}
    for branch in BRANCHES:
        parents = factors["00"]["evaluation"]["final"][branch]["parents"]
        output[branch] = {}
        for group in ("all",*sorted(parents)):
            values = {}
            for factor, result in factors.items():
                evaluation = result["evaluation"]["final"][branch]
                aggregate = evaluation["aggregate"] if group=="all" else evaluation["parents"][group]
                values[factor] = metric_vector(aggregate,result["rank"]["branches"][branch][group])
            contrasts = {}
            for name,terms in {"10-00":(("10",1),("00",-1)),"01-00":(("01",1),("00",-1)),
                "11-01":(("11",1),("01",-1)),"11-10":(("11",1),("10",-1)),
                "interaction_11-10-01+00":(("11",1),("10",-1),("01",-1),("00",1))}.items():
                contrasts[name] = {metric:None if any(values[f][metric] is None for f,_ in terms)
                    else sum(values[f][metric]*coefficient for f,coefficient in terms) for metric in values["00"]}
            output[branch][group] = {"values":values,"contrasts":contrasts}
    return output


def run_factorial(run, data, card, baseline, baseline_rank, *, step_callback=None):
    factors = {"00":{"evaluation":baseline["result"]["evaluation"],"rank":baseline_rank["result"],"reused_not_retrained":True}}
    policy = card["balance_policy"]
    for factor in FACTORS:
        root = run/"artifacts/factors"/factor
        for name in ("artifacts","metrics","logs","previews"):
            (root/name).mkdir(parents=True,exist_ok=False)
        callback = None if step_callback is None else lambda branch,record:step_callback(factor,branch,record)
        factors[factor] = run_factor(root,data,card["training"],card["evaluation"],factor,
            policy["member_class_counts"],policy["event_class_counts"],baseline["result"]["evaluation"]["initial"],
            card["baseline_initial_state_sha256"],step_callback=callback)
        write_json(root/"metrics/summary.json",factors[factor])
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    effects = factorial_effects(factors)
    write_json(run/"metrics/factorial_effects.json",effects)
    capacity = {}
    for factor,result in factors.items():
        final = result["evaluation"]["final"]
        capacity[factor] = {"gt_member_f1_at_least_0_90":final["gt_axes"]["aggregate"]["members"]["known_scored_f1"]>=.9,
            "predicted_member_f1_above_no_relations":final["predicted_axes"]["aggregate"]["members"]["known_scored_f1"]>
                final["predicted_no_relations"]["aggregate"]["members"]["known_scored_f1"],
            "available_event_fit_macro_at_least_0_95":{b:final[b]["aggregate"]["events"]["two_class_macro_f1"] is not None and
                final[b]["aggregate"]["events"]["two_class_macro_f1"]>=.95 for b in BRANCHES},
            "full_three_class_ready":False,"scientific_gate_pass":False}
    return {"factors":factors,"capacity_diagnostics":capacity,"selected_factor":None,
        "optimizer_steps":sum(factors[f]["optimizer_steps"] for f in FACTORS),
        "head_inference_windows":sum(factors[f]["head_inference_windows"] for f in FACTORS),
        "backbone_windows":0,"new_sensor_frames":0,"old_checkpoint_reads":0,"threshold_search":False}


def execute(spec, run):
    run = Path(run).resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec)
            or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec):
        raise RuntimeError("requires fresh exact run; no overwrite/retry")
    started,error,result,reads = time.monotonic(),None,{},{}
    write_json(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    def deadline(signum,frame): raise TimeoutError("fixed1800s factorial deadline exceeded")
    old_handler = signal.signal(signal.SIGALRM,deadline);signal.alarm(1800)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_class_balance_factorial_card(card)
        if (not report.passed or spec["operation"]!="training" or spec["wall_time_cap_s"]!=1800
                or load_json(run/"config/data_card.json")!=card
                or any(spec[k]!=card[k] for k in ("training","evaluation","balance_policy","rank_policy"))):
            raise ValueError("frozen factorial scope drift: "+str(report.errors))
        frozen = {**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest: raise ValueError("frozen source drift: "+path)
        versions = {"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"zarr":zarr.__version__,"scipy":scipy.__version__}
        if versions!=spec["expected_versions"]: raise ValueError("environment drift")
        if not torch.cuda.is_available(): raise RuntimeError("required CUDA unavailable; no CPU fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8": raise ValueError("deterministic CUDA environment missing")
        torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        write_json(run/"config/execution_environment.json",{**versions,"platform":platform.platform(),
            "device":card["training"]["device"],"gpu_name":torch.cuda.get_device_name(0),"deterministic_algorithms":True})
        data,baseline,rank,reads = load_inputs(PROJECT_ROOT,card)
        with (run/"logs/training.jsonl").open("x") as log:
            def callback(factor,branch,record):
                log.write(json.dumps({"factor":factor,"branch":branch,"elapsed_s":time.monotonic()-started,**record})+"\n");log.flush()
                if record["step"]==1 or record["step"]%100==0:
                    print(json.dumps({"factor":factor,"branch":branch,"step":record["step"],"loss":record["loss"]["total"],"elapsed_s":time.monotonic()-started}),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3 or torch.cuda.max_memory_reserved()>4*1024**3:
                    raise RuntimeError("host/GPU4GiB exceeded")
            result = run_factorial(run,data,card,baseline,rank,step_callback=callback)
        for key in ("optimizer_steps","head_inference_windows","backbone_windows","new_sensor_frames"):
            if result[key]!=spec["expected_counts"][key]: raise ValueError("execution count drift: "+key)
        xy = list((run/"artifacts/factors").glob("*/previews/*__initial__*.svg")) + list((run/"artifacts/factors").glob("*/previews/*__final__*.svg"))
        all_plots = list((run/"artifacts/factors").glob("*/previews/*.svg"))
        result.update(xy_svg_figures=len(xy),rank_svg_figures=len(all_plots)-len(xy))
        for key in ("xy_svg_figures","rank_svg_figures"):
            if result[key]!=spec["expected_counts"][key]: raise ValueError("plot evidence count drift: "+key)
        result.update(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        if (result["peak_rss_bytes"]>4*1024**3 or result["peak_gpu_reserved_bytes"]>4*1024**3
                or time.monotonic()-started>1800 or sum(p.stat().st_size for p in run.rglob("*") if p.is_file())>499_000_000):
            raise RuntimeError("resource cap exceeded")
        for path,digest in {**frozen,**reads}.items():
            if sha(contained(path))!=digest: raise ValueError("source changed during training: "+path)
    except Exception:
        error = traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
    write_json(run/"artifacts/source_reads_sha256.json",reads)
    status = "CLASS_BALANCE_FACTORIAL_FIXED_BUDGET_COMPLETE" if error is None else "CLASS_BALANCE_FACTORIAL_FAIL"
    write_json(run/"metrics/summary.json",{"status":status,"elapsed_s":time.monotonic()-started,
        "result":result,"error":error,"scientific_gate_pass":False,"full_three_class_ready":False})
    write_json(run/"RUN_STATE.json",{"state":"COMPLETED" if error is None else "FAILED","run_id":run.name,"error":error})
    seal = run/"artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"status":status,"error":error,"elapsed_s":time.monotonic()-started,"seal_sha256":sha(seal)}),flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec",type=Path,required=True);parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    return execute(load_json(args.spec),args.run_dir)


if __name__=="__main__": raise SystemExit(main())
