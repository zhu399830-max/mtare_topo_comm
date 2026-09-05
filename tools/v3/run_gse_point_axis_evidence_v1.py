#!/usr/bin/env python3
"""Complete evidence after plotting failure, without touching the trained run."""
import argparse
import json
import platform
import importlib.metadata
from pathlib import Path
import resource
import signal
import time
import traceback

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, build_run_id
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from mtare_topo.representation.gse_point_axis_probe import decide, sample_schedule
from mtare_topo.representation.gse_point_axis_loss import axis_set_metrics
from run_gse_point_axis_probe_v1 import save_preview
from run_gse_composition_field_recovery_v1 import sha, contained


def restore_result(evaluations,manifest):
    result=decide(evaluations["initial"],evaluations["raw_no_offset"],evaluations["raw_slot_offset"],evaluations["legacy_frozen"],[r["task"] for r in manifest])
    result["parent_count"]=len(result["parents"])
    return result


def restore_teacher(coordinate_rows,manifest):
    if len(manifest)!=180 or len(coordinate_rows)!=180:
        raise ValueError("exact180 cached observations required")
    keys=[(r["task"],r["row_index"]) for r in coordinate_rows]
    wanted=[(r["task"],r["row_index"]) for r in manifest]
    if len(set(keys))!=180 or len(set(wanted))!=180 or keys!=wanted:
        raise ValueError("cached teacher population/order identity mismatch")
    lookup=dict(zip(keys,coordinate_rows));cache=[];total=0
    for index,key in enumerate(wanted):
        row=lookup[key];target=np.zeros((1,32,3,3),np.float32);mask=np.zeros((1,32),bool);seen=set()
        for point in row["coordinate_support"]["control_points"]:
            slot,control=point["teacher_slot_scoring_only"],point["control_index"]
            if type(slot) is not int or type(control) is not int or not 0<=slot<32 or not 0<=control<3 or (slot,control) in seen:
                raise ValueError("invalid/duplicate cached control index")
            xyz=np.asarray(point["query_current_sensor_m"],dtype=np.float64)
            if xyz.shape!=(3,) or not np.isfinite(xyz).all() or (np.abs(xyz)>np.finfo(np.float32).max).any():
                raise ValueError("invalid/nonfinite cached teacher point")
            target[0,slot,control]=xyz;mask[0,slot]=True;seen.add((slot,control))
        if not mask.any() or any((int(slot),control) not in seen for slot in np.flatnonzero(mask[0]) for control in range(3)):
            raise ValueError("missing control in active cached primitive")
        if type(manifest[index].get("visible_fragments")) is not int or manifest[index]["visible_fragments"]!=int(mask.sum()):
            raise ValueError("manifest visible fragment count mismatch")
        total+=len(seen)
        cache.append({"target":torch.from_numpy(target),"mask":torch.from_numpy(mask),"task":key[0]})
    if total!=4356: raise ValueError("cached teacher total must remain4356 controls/1452 fragments")
    return cache


def equivalent(a,b):
    """CPU/GPU double reductions: exact keys/identities, 64-eps numeric bound."""
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if type(a) is float or type(b) is float:
        return (type(a) in (int,float) and type(b) in (int,float) and np.isfinite(a) and np.isfinite(b)
                and abs(a-b)<=64*np.finfo(np.float64).eps*max(1,abs(a),abs(b)))
    return type(a) is type(b) and a==b


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args();spec=load_json(args.spec);run=args.run_dir.resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec): raise RuntimeError("fresh exact evidence run required")
    started=time.monotonic();error=None;result={}
    def timeout(signum,frame):raise TimeoutError("120s evidence deadline")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    write_json(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        versions={"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","scipy","matplotlib","zarr")}}
        write_json(run/"environment/execution_environment.json",{"versions":versions,"device":"cpu","optimizer_steps":0})
        if versions!=spec["expected_versions"]:raise ValueError("evidence environment drift")
        card=load_json(contained(spec["data_card"]));report=validate_scoped_inventory_card(card)
        if not report.passed or spec["operation"]!="audit" or load_json(run/"config/data_card.json")!=card:
            raise ValueError("evidence-only card drift")
        frozen={**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError(f"source drift: {path}")
        def read(path):
            if path not in card["sealed_sources"]:raise ValueError("undeclared artifact")
            return json.loads(contained(path).read_text())
        base=spec["training_run"]
        old=read(base+"/metrics/summary.json")
        if (read(base+"/RUN_STATE.json")["state"]!="FAILED" or "TypeError: 'int' object is not iterable" not in old["error"]
                or old["result"]["steps_completed"]!={"raw_no_offset":540,"raw_slot_offset":540}):
            raise ValueError("not the frozen post-training plotting failure")
        manifest=read(base+"/artifacts/sample_manifest.json")
        if [(r["task"],r["row_index"]) for r in manifest]!=[(r["task"],r["row_index"]) for r in card["selected_rows"]]:
            raise ValueError("training manifest differs from approved selection")
        schedule=read(base+"/artifacts/sample_schedule.json")
        if schedule!=sample_schedule():raise ValueError("schedule drift")
        progress=[json.loads(line) for line in contained(base+"/logs/progress.jsonl").read_text().splitlines()]
        for variant in ("raw_no_offset","raw_slot_offset"):
            rows=[r for r in progress if r.get("stage")=="train" and r["variant"]==variant]
            if ([r["step"] for r in rows]!=list(range(1,541)) or [r["observation_index"] for r in rows]!=schedule
                    or not all(np.isfinite(r["loss_coordinate_l1_div50"]) for r in rows)):
                raise ValueError("raw training ledger drift")
        cache=restore_teacher(read(spec["coordinate_rows"]),manifest)
        path=base+"/artifacts/all_predictions.npz"
        if path not in card["sealed_sources"]:raise ValueError("unfrozen predictions")
        with np.load(contained(path),allow_pickle=False) as source:axes={k:source[k] for k in source.files}
        if set(axes)!={"initial","raw_no_offset","raw_slot_offset","legacy_frozen"} or any(v.shape!=(180,32,3,3) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in axes.values()):
            raise ValueError("prediction shape/content drift")
        old_metrics=read(base+"/artifacts/observation_metrics.json");rescored={}
        for name,value in axes.items():
            rescored[name]=[axis_set_metrics(torch.from_numpy(value[i:i+1]),item["target"],item["mask"])[0] for i,item in enumerate(cache)]
        if not equivalent(old_metrics,rescored):raise ValueError("saved metrics/matching do not reproduce within64eps")
        result=restore_result(old_metrics,manifest)
        if result["checks"]!=old["result"]["checks"] or result["decision"]!=old["result"]["decision"] or result["macro_observation_errors"]!=old["result"]["macro_observation_errors"]:
            raise ValueError("original scientific decision changed")
        save_preview(run,cache,axes,result)
        write_json(run/"artifacts/restored_parent_metrics.json",result["parents"])
        result.update({"original_run_remains_failed":True,"original_gate_decision_preserved":True,"matching_and_metric_reconstruction":True,
                       "optimizer_steps":0,"model_inference":0,"checkpoint_reads":0,"raw_sensor_frames":0,"new_labels":0,"observations":180})
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError("source changed during evidence completion")
        result["peak_host_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if result["peak_host_rss_bytes"]>4*1024**3 or time.monotonic()-started>120:raise RuntimeError("evidence resource cap")
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:signal.alarm(0)
    write_json(run/"metrics/summary.json",{"status":"EVIDENCE_COMPLETE" if not error else "EVIDENCE_FAIL","elapsed_s":time.monotonic()-started,"result":result,"error":error,"scientific_gate_pass":False})
    write_json(run/"RUN_STATE.json",{"state":"COMPLETED" if not error else "FAILED","run_id":run.name,"error":error})
    seal=run/"artifacts/evidence_sha256.txt";seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"error":error,"result":result,"seal_sha256":sha(seal)}));return int(error is not None)


if __name__=="__main__":raise SystemExit(main())
