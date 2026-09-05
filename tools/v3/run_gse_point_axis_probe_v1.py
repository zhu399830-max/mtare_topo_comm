#!/usr/bin/env python3
"""One fixed C01 geometry fit probe; paired variants, no event training."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import torch
import zarr
import scipy
import matplotlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout
from mtare_topo.representation.gse_point_axis_probe import (
    VARIANTS, sample_schedule, configure_variant, evaluate, fit, decide,
)
from mtare_topo.representation.gse_point_axis_loss import axis_set_metrics, axis_set_loss
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from run_gse_composition_field_recovery_v1 import sha, contained, load_model, predict, state_sha

GEOMETRY_FIELDS=frozenset(("frame_row","source_global_sequence_index","primitive_mask","axis_control_current_sensor_m"))


@torch.no_grad()
def build_cache(model, reader, records, reference_root, source_expected, log, guard):
    """Read only exact sensor/geometry fields; teacher remains outside student."""
    entries, legacy, targets, total_frames = [], [], [], 0
    device=next(model.parameters()).device
    for task in sorted(reader.selection):
        guard()
        batch=reader.read_task(task)
        ids=np.asarray([r["row_index"] for r in reader.selection[task]],dtype=np.int64)
        group=reader._open(reader.teacher_root, task, GEOMETRY_FIELDS)
        target=np.asarray(group["axis_control_current_sensor_m"].oindex[ids])
        mask=np.asarray(group["primitive_mask"].oindex[ids])
        if (target.shape!=(18,32,3,3) or target.dtype!=np.float32 or mask.shape!=(18,32)
                or not np.isin(mask,(0,1)).all() or not mask.any(axis=1).all()
                or not np.isfinite(target[mask.astype(bool)]).all()
                or not np.array_equal(np.asarray(group["frame_row"].oindex[ids]),batch.frame_rows)
                or not np.array_equal(np.asarray(group["source_global_sequence_index"].oindex[ids]),batch.source_sequence_indices)):
            raise ValueError("existing geometry/frame contract drift")
        path=reference_root/(task+".npz")
        if source_expected.get(str(path))!=sha(path):
            raise ValueError("unsealed/drifted previous prediction reference")
        reader.opened[str(path)]=source_expected[str(path)]
        raw,_,_=predict(model,batch.student)
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(raw) or any(not np.array_equal(raw[k],old[k]) for k in raw):
                raise ValueError("sealed seed0 batch18 prediction parity failed")
        values=[torch.from_numpy(np.stack([getattr(row,key) for row in batch.student])).to(device)
                for key in ("range_valid","relative_translation_current_sensor_m","relative_yaw_current_sensor_deg")]
        memory,memory_valid,memory_xyz,_=model._memory(*values)
        slots=model.slot_decoder(model.slot_query[None].expand(18,-1,-1),memory,memory_key_padding_mask=~memory_valid)
        queries=model.control_query(slots).reshape(18,32,3,128)
        logits=torch.einsum("bscd,bnd->bscn",queries,memory)/math.sqrt(128)
        weights=torch.softmax(logits.masked_fill(~memory_valid[:,None,None],-torch.inf),dim=-1)
        reconstructed=torch.einsum("bscn,bnd->bscd",weights,memory_xyz)
        if not np.array_equal(reconstructed.cpu().numpy(),raw["axis_control_current_sensor_m"]):
            raise ValueError("cached memory/slots do not reproduce original control readout")
        points,valid=register_causal_lidar_points(*values)
        memory, memory_xyz, slots = [v.detach().cpu() for v in (memory,memory_xyz,slots)]
        points,valid=points.reshape(18,-1,3).cpu(),valid.reshape(18,-1).cpu()
        indices=(torch.arange(5)[:,None,None]*180+torch.arange(720)[None,None]//4).expand(5,16,720).reshape(1,-1)
        for i in range(18):
            inputs=(points[i:i+1],valid[i:i+1],memory[i:i+1],memory_xyz[i:i+1],indices,slots[i:i+1])
            entries.append({"student":inputs,"target":torch.from_numpy(target[i:i+1]),
                            "mask":torch.from_numpy(mask[i:i+1].astype(bool)),"task":task})
            legacy.append(raw["axis_control_current_sensor_m"][i])
            targets.append({"task":task,"row_index":int(ids[i]),"visible_fragments":int(mask[i].sum())})
        total_frames+=len(np.unique(batch.frame_rows))
        log.write(json.dumps({"stage":"cache","task":task,"rows":18,"sealed_prediction_parity_exact":True})+"\n");log.flush()
    expected=[(r["task"],r["row_index"]) for r in records]
    if ([(r["task"],r["row_index"]) for r in targets]!=expected or len(entries)!=180 or total_frames!=900
            or sum(r["visible_fragments"] for r in targets)!=1452):
        raise ValueError("exact cache population/order drift")
    guard()
    return entries,np.stack(legacy),targets


def save_preview(run, cache, axes, result):
    # All180 observations and all32 queries retained in arrays and each view.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=result["parents"]
    fig,ax=plt.subplots(figsize=(10,4))
    for j,(name,label) in enumerate((("legacy_frozen","Frozen old"),("raw_no_offset","Raw, no offset"),("raw_slot_offset","Raw + axis offset"))):
        ax.bar(np.arange(10)+(j-1)*.25,[v[name] for v in p],.25,label=label)
    ax.set_xticks(np.arange(10),[v["parent"][:3] for v in p]);ax.set_ylabel("Coordinate MAE (m), observation mean")
    ax.set_title("C01 same180 FIT probe; geometry matching, not detection/generalization")
    ax.legend();fig.tight_layout();fig.savefig(run/"previews/parent_geometry_mae.svg");plt.close(fig)
    for task in sorted({c["task"] for c in cache}):
        indices=[i for i,c in enumerate(cache) if c["task"]==task]
        fig,axs=plt.subplots(6,6,figsize=(18,18))
        for row,index in enumerate(indices):
            item=cache[index];target=item["target"][0][item["mask"][0]].numpy()
            for view,dimensions in enumerate(((0,1),(0,2))):
                ax=axs.flat[row*2+view]
                for name,color in (("raw_no_offset","#3587ba"),("raw_slot_offset","#df7726")):
                    for primitive in axes[name][index]:
                        ax.plot(primitive[:,dimensions[0]],primitive[:,dimensions[1]],color=color,alpha=.3,lw=.6)
                for primitive in target:
                    ax.plot(primitive[:,dimensions[0]],primitive[:,dimensions[1]],color="black",lw=1.)
                ax.set_title(f"observation {index} {'XY' if view==0 else 'XZ'} (m)",fontsize=8)
                ax.set_aspect("equal",adjustable="datalim");ax.tick_params(labelsize=6)
        fig.suptitle(f"{task}: all18 observations/all32 queries; black existing teacher, blue raw, orange offset\nFIT geometry only; not detection, no query filtering",fontsize=11)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(run/f"previews/{task}.svg");plt.close(fig)


def main():
    from mtare_topo.governance_point_axis import validate_point_axis_training_card
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args();spec=load_json(args.spec);run=args.run_dir.resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec)
            or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec):
        raise RuntimeError("requires exact fresh run, no overwrite or retry")
    started=time.monotonic();error=None;result={"steps_completed":{}};reader=None
    def timeout(signum,frame): raise TimeoutError("execution interrupted or frozen1200s deadline")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    signal.signal(signal.SIGTERM,timeout)
    def guard():
        if (time.monotonic()-started>1200 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3
                or (torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>4*1024**3)):
            raise RuntimeError("probe resource cap exceeded")
    write_json(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        card=load_json(contained(spec["data_card"]));report=validate_point_axis_training_card(card)
        if (not report.passed or spec["operation"]!="training" or load_json(run/"config/data_card.json")!=card
                or spec["training"]!=card["training"] or spec.get("wall_time_cap_s")!=1200):
            raise ValueError(f"training card/scope drift: {report.errors}")
        frozen={**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest: raise ValueError(f"frozen source drift: {path}")
        versions={"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"zarr":zarr.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}
        if versions!=spec["expected_versions"]: raise ValueError("environment drift")
        expected={}
        for seal in spec["source_seals"]:
            if seal not in frozen: raise ValueError("unfrozen seal")
            for line in contained(seal).read_text().splitlines():
                digest,path=line.split(None,1);path=str(contained(path))
                if path in expected and expected[path]!=digest: raise ValueError("conflicting source seals")
                expected[path]=digest
        records=card["selected_rows"]
        reader=ScopedCompositionModelReader(contained(spec["sensor_root"]),contained(spec["teacher_root"]),records,expected_sha256=expected)
        model=load_model(contained(card["checkpoint"]["path"]));before=state_sha(model)
        write_json(run/"config/execution_environment.json",{**versions,"device":torch.cuda.get_device_name(0),"tf32":False,"platform":platform.platform(),"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
        with (run/"logs/progress.jsonl").open("x") as log:
            cache,legacy,manifest=build_cache(model,reader,records,contained(spec["prediction_reference_root"]),expected,log,guard)
            if state_sha(model)!=before: raise ValueError("backbone drift before fitting")
            schedule=sample_schedule();write_json(run/"artifacts/sample_schedule.json",schedule)
            write_json(run/"artifacts/sample_manifest.json",manifest)
            torch.manual_seed(0);initial=PointAxisReadout().to("cuda")
            torch.save(initial.state_dict(),run/"artifacts/initial_head.pt")
            readiness=[]
            for variant in VARIANTS:
                dry=configure_variant(initial,variant)
                item=cache[0];inputs=[v.to("cuda") for v in item["student"]]
                output=dry(*inputs).votes.axis_control_m
                loss=axis_set_loss(output,item["target"].to("cuda"),item["mask"].to("cuda"))
                loss.backward();guard()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in dry.parameters() if p.requires_grad):
                    raise ValueError("real-input readiness missing/nonfinite gradient")
                readiness.append({"variant":variant,"loss":float(loss.detach()),"trainable_parameters":sum(p.numel() for p in dry.parameters() if p.requires_grad),"optimizer_steps":0})
                del dry,output,loss,inputs
            write_json(run/"artifacts/pretraining_readiness.json",readiness)
            result["trainable_parameters_by_variant"]={r["variant"]:r["trainable_parameters"] for r in readiness}
            axes={"legacy_frozen":legacy};evaluations={}
            initial_axes=None
            for variant in VARIANTS:
                head=configure_variant(initial,variant)
                current,initial_metrics=evaluate(head,cache)
                if initial_axes is None:
                    initial_axes=current;axes["initial"]=current;evaluations["initial"]=initial_metrics
                elif not np.array_equal(current,initial_axes): raise ValueError("paired initialization differs")
                def log_step(row):
                    result["steps_completed"][variant]=row["step"]
                    log.write(json.dumps({"stage":"train","variant":variant,**row})+"\n");log.flush()
                trained=fit(head,cache,schedule,log_step,guard=guard)
                if trained["steps"]!=540: raise ValueError("step count drift")
                if variant=="raw_no_offset" and any(bool(p.any()) for layer in (head.offset,head.slot_offset) for p in layer.parameters()):
                    raise ValueError("no-offset baseline changed offsets")
                axes[variant],evaluations[variant]=evaluate(head,cache)
                torch.save({"schema_version":"gse_point_axis_fit_probe_v1","variant":variant,"seed":0,
                            "head_state_dict":head.state_dict(),"training":trained,"backbone_state_sha256":before},run/f"artifacts/{variant}_final.pt")
                del head
                guard()
            legacy_metrics=[]
            for i,item in enumerate(cache):
                legacy_metrics.extend(axis_set_metrics(torch.from_numpy(legacy[i:i+1]),item["target"],item["mask"]))
            evaluations["legacy_frozen"]=legacy_metrics
            result.update(decide(evaluations["initial"],evaluations["raw_no_offset"],evaluations["raw_slot_offset"],legacy_metrics,[c["task"] for c in cache]))
            if state_sha(model)!=before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("frozen backbone state/gradient changed")
            result.update({"observations":180,"parents":10,"sensor_frames":900,"visible_fragments":1452,
                           "backbone_state_sha256":before,"backbone_unchanged":True,"old_prediction_parity_exact":True,
                           "new_labels":0,"optimizer_steps":1080,"trained_head_parameters":35460})
            result["computation_ledger"]={"old_full_model_windows":180,"cached_backbone_memory_windows":180,
                "head_training_windows":1080,"head_evaluation_windows":720,"head_zero_update_readiness_windows":2}
            np.savez_compressed(run/"artifacts/all_predictions.npz",**axes)
            write_json(run/"artifacts/observation_metrics.json",evaluations)
            save_preview(run,cache,axes,result)
            for path,digest in {**frozen,**reader.opened}.items():
                if sha(contained(path))!=digest: raise ValueError("source drift after fit")
            guard()
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:
        signal.alarm(0)
    if reader is not None:
        write_json(run/"artifacts/source_reads_sha256.json",{str(Path(p).relative_to(PROJECT_ROOT)):h for p,h in sorted(reader.opened.items())})
    result.update({"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0})
    write_json(run/"metrics/summary.json",{"status":"SYSTEM_FAIL" if error else result["decision"],"elapsed_s":time.monotonic()-started,
                                         "error":error,"result":result,"scientific_gate_pass":False})
    write_json(run/"RUN_STATE.json",{"state":"FAILED" if error else "COMPLETED","run_id":run.name,"error":error})
    seal=run/"artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"error":error,"result":result,"seal_sha256":sha(seal)}))
    return int(error is not None)


if __name__=="__main__": raise SystemExit(main())
