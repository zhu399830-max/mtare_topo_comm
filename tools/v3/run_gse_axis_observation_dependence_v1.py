#!/usr/bin/env python3
"""One immutable zero-training cached-output correspondence diagnostic."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import time
import traceback
import numpy as np
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from mtare_topo.evaluation.gse_axis_observation_dependence import same_parent_derangement,score_observation,summarize,compare
from run_gse_point_axis_evidence_v1 import restore_teacher,equivalent
from run_gse_point_axis_evidence_v1r import write_json_with_parent as write_json
from run_gse_composition_field_recovery_v1 import sha,contained

METHODS=("initial","raw_no_offset","raw_slot_offset","legacy_frozen")


def plot_summary(run,results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    labels=("Initial","Raw","Offset","Frozen old")
    fields=("coordinate_mae_m","transverse_rms_m","undirected_direction_error_deg")
    for ax,field,title in zip(axs,fields,("Coordinate MAE (m)","Transverse RMS (m)","Undirected angle (deg)")):
        for j,condition in enumerate(("correct","shuffled")):
            values=[]
            for method in METHODS:
                m=results[method][condition]["macro"]
                values.append(m[field] if field=="coordinate_mae_m" else m["layout"][field]["mean"])
            ax.bar(np.arange(4)+(j-.5)*.35,[np.nan if v is None else v for v in values],.35,label=condition)
        ax.set_xticks(np.arange(4),labels);ax.set_ylabel(title);ax.legend()
    fig.suptitle("C01 10 parents / same180 FIT observations; fixed within-parent shift9\nCached output correspondence, not a generalization or detection score",fontsize=10)
    fig.tight_layout();fig.savefig(run/"previews/correspondence_comparison.svg");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args();spec=load_json(args.spec);run=args.run_dir.resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec):raise ValueError("exact fresh run required")
    start=time.monotonic();result={};error=None
    def guard():
        if time.monotonic()-start>120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise RuntimeError("120s/4GiB resource cap")
    def timeout(signum,frame):raise TimeoutError("120s CPU diagnostic cap")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    write_json(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        versions={"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","scipy","matplotlib","zarr")}}
        write_json(run/"environment/execution_environment.json",{"versions":versions,"device":"cpu"})
        if versions!=spec["expected_versions"]:raise ValueError("environment drift")
        card=load_json(contained(spec["data_card"]));report=validate_scoped_inventory_card(card)
        if not report.passed or spec["operation"]!="audit" or load_json(run/"config/data_card.json")!=card:raise ValueError("scope drift")
        frozen={**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError(f"source drift:{path}")
        def read(path):
            if path not in card["sealed_sources"]:raise ValueError("undeclared artifact")
            return load_json(contained(path))
        base=spec["training_run"];manifest=read(base+"/artifacts/sample_manifest.json")
        if [(r["task"],r["row_index"]) for r in manifest]!=[(r["task"],r["row_index"]) for r in card["selected_rows"]]:raise ValueError("selection drift")
        mapping=same_parent_derangement(manifest)
        if mapping!=spec["shuffled_prediction_indices"]:raise ValueError("pre-frozen derangement drift")
        write_json(run/"artifacts/correspondence_manifest.json",[{"target":r,"prediction_source":manifest[mapping[i]],"target_index":i,"prediction_index":mapping[i]} for i,r in enumerate(manifest)])
        cache=restore_teacher(read(spec["coordinate_rows"]),manifest)
        path=base+"/artifacts/all_predictions.npz"
        if path not in card["sealed_sources"]:raise ValueError("unfrozen predictions")
        with np.load(contained(path),allow_pickle=False) as source:axes={k:source[k] for k in source.files}
        if set(axes)!=set(METHODS) or any(v.shape!=(180,32,3,3) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in axes.values()):raise ValueError("prediction content drift")
        old=read(base+"/artifacts/observation_metrics.json")
        with (run/"logs/progress.jsonl").open("x") as log:
            for method in METHODS:
                result[method]={}
                for condition in ("correct","shuffled"):
                    indices=range(180) if condition=="correct" else mapping
                    rows=[]
                    for i,source in enumerate(indices):
                        guard();item=cache[i]
                        rows.append(score_observation(axes[method][source],item["target"],item["mask"]))
                    if condition=="correct" and not equivalent([r["fit"] for r in rows],old[method]):raise ValueError("original score/matching mismatch")
                    write_json(run/f"artifacts/{method}_{condition}.json",rows)
                    result[method][condition]=summarize(rows,manifest)
                    log.write(json.dumps({"method":method,"condition":condition,"observations":180,"elapsed_s":time.monotonic()-start})+"\n");log.flush()
                result[method]["comparison"]=compare(result[method]["correct"],result[method]["shuffled"])
        plot_summary(run,result)
        write_json(run/"artifacts/source_reads_sha256.json",frozen)
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError("source changed")
        guard()
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:signal.alarm(0)
    summary={"status":"DIAGNOSTIC_COMPLETE" if not error else "DIAGNOSTIC_FAIL","result":result,"error":error,
        "elapsed_s":time.monotonic()-start,"peak_host_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "optimizer_steps":0,"model_inference":0,"checkpoint_reads":0,"raw_sensor_frames":0,"new_labels":0,
        "scientific_gate_pass":False,"original_offset_decision":"STOP_BEFORE_EXPANSION"}
    write_json(run/"metrics/summary.json",summary);write_json(run/"RUN_STATE.json",{"state":"COMPLETED" if not error else "FAILED","run_id":run.name,"error":error})
    seal=run/"artifacts/evidence_sha256.txt";seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"error":error,"comparisons":{k:v.get("comparison") for k,v in result.items()},"elapsed_s":summary["elapsed_s"],"seal_sha256":sha(seal)}))
    return int(error is not None)


if __name__=="__main__":raise SystemExit(main())
