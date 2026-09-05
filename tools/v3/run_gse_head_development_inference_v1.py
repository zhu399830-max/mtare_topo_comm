#!/usr/bin/env python3
"""One frozen C02 head-development comparison, with no weight updates."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback
import numpy as np
import torch
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_head_inference import validate_head_development_inference_card
from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout
from mtare_topo.evaluation.gse_head_development_inference import infer_task,read_geometry,evaluate_outputs,plot_development
from run_gse_composition_field_recovery_v1 import sha,contained,load_model,state_sha
from run_gse_head_development_metadata_v1 import write


def load_frozen_pair(card):
    model=load_model(contained(card["checkpoints"]["backbone"]["path"]))
    saved=torch.load(contained(card["checkpoints"]["head"]["path"]),map_location="cpu",weights_only=False)
    if (saved.get("schema_version")!="gse_point_axis_fit_probe_v1" or saved.get("variant")!="raw_no_offset"
            or saved.get("seed")!=0 or saved.get("training",{}).get("steps")!=540
            or saved.get("backbone_state_sha256")!=state_sha(model)):
        raise ValueError("raw head/backbone provenance mismatch")
    head=PointAxisReadout();head.load_state_dict(saved["head_state_dict"],strict=True)
    head.eval().requires_grad_(False).to("cuda")
    if any(bool(p.any()) for layer in (head.offset,head.slot_offset) for p in layer.parameters()):raise ValueError("raw-only head has nonzero offset")
    return model,head


def source_index(card):
    fields={"sensor":{"range_m","valid_mask"},"teacher":{"frame_row","source_global_sequence_index","primitive_mask","axis_control_current_sensor_m","relative_translation_current_sensor_m","relative_yaw_current_sensor_deg"}}
    expected={}
    for kind,seal in card["source_seals"].items():
        prefixes=[card["source_roots"][kind]+"/"+task+".zarr/" for task in card["tasks"]]
        with contained(seal).open() as stream:
            for line in stream:
                digest,path=line.strip().split(None,1)
                suffix=next((path[len(p):] for p in prefixes if path.startswith(p)),None)
                if suffix is not None and (suffix in (".zgroup",".zattrs") or suffix.split("/")[0] in fields[kind]):
                    key=str(contained(path))
                    if key in expected and expected[key]!=digest:raise ValueError("conflicting input seals")
                    expected[key]=digest
    return expected


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args();spec=load_json(args.spec);run=args.run_dir.resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec):raise ValueError("exact fresh inference run required")
    start=time.monotonic();result={};error=None;reader=None;ledger={"raw_primary_observations":0,"raw_repeat_observations":0,"legacy_primary_observations":0,"legacy_repeat_observations":0,"cached_backbone_windows":0,"optimizer_steps":0}
    def guard():
        if (time.monotonic()-start>120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3
                or (torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>4*1024**3)):raise RuntimeError("120s/4GiB host/GPU cap")
    def timeout(signum,frame):raise TimeoutError("120s inference limit")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);write(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        card=load_json(contained(spec["data_card"]));report=validate_head_development_inference_card(card)
        if not report.passed or spec["operation"]!="data_export" or load_json(run/"config/data_card.json")!=card:raise ValueError("inference approval/scope drift")
        frozen={**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError("frozen input/tool drift")
        versions={"python":platform.python_version(),**{name:importlib.metadata.version(name) for name in ("numpy","torch","zarr","scipy","matplotlib")}}
        if versions!=spec["expected_versions"]:raise ValueError("inference version drift")
        records=json.loads(contained(card["metadata_selection_path"]).read_text())
        if records!=card["selected_rows"]:raise ValueError("selection mismatch")
        expected=source_index(card)
        reader=ScopedCompositionModelReader(contained(card["source_roots"]["sensor"]),contained(card["source_roots"]["teacher"]),records,expected_sha256=expected)
        model,head=load_frozen_pair(card);before={"backbone":state_sha(model),"head":state_sha(head)}
        write(run/"config/execution_environment.json",{"versions":versions,"device":str(next(model.parameters()).device),"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"tf32":False})
        outputs={"raw_no_offset":[],"legacy_frozen":[]};targets=[];masks=[];frames=0
        with (run/"logs/inference.jsonl").open("x") as log:
            for task in card["tasks"]:
                guard();batch=reader.read_task(task);rows=[r for r in records if r["task"]==task]
                truth,mask=read_geometry(reader,task,batch,rows)
                repeated=task==card["inference"]["repeat_task"]
                values=infer_task(model,head,batch.student,guard,repeat=repeated)
                for method in outputs:outputs[method].append(values[method])
                targets.append(truth);masks.append(mask);frames+=len(np.unique(batch.frame_rows))
                ledger["raw_primary_observations"]+=18;ledger["legacy_primary_observations"]+=18;ledger["cached_backbone_windows"]+=18
                if repeated:ledger["raw_repeat_observations"]+=18;ledger["legacy_repeat_observations"]+=18
                log.write(json.dumps({"task":task,"observations":18,"visible_fragments":int(mask.sum()),"repeated_exact":repeated,"elapsed_s":time.monotonic()-start})+"\n");log.flush()
        for key,value in ledger.items():
            if card["inference"][key]!=value:raise ValueError("inference ledger drift")
        after={"backbone":state_sha(model),"head":state_sha(head)}
        if before!=after or any(p.requires_grad or p.grad is not None for module in (model,head) for p in module.parameters()):raise ValueError("frozen parameter/gradient drift")
        axes={k:np.concatenate(v) for k,v in outputs.items()};targets=np.concatenate(targets);masks=np.concatenate(masks)
        if frames!=900 or int(masks.sum())!=1489:raise ValueError("metadata target/frame count drift")
        details,result=evaluate_outputs(axes,targets,masks,records)
        np.savez_compressed(run/"artifacts/axis_predictions.npz",**{k+"_axis_control_m":v for k,v in axes.items()})
        np.savez_compressed(run/"artifacts/existing_scoring_targets.npz",axis_control_m=targets,mask=masks)
        for name,rows in details.items():write(run/f"artifacts/{name}_observation_metrics.json",rows)
        write(run/"artifacts/selected_rows.json",records)
        plot_development(run,axes,targets,masks,records,result)
        result.update({"observations":180,"parents":10,"sensor_frames":frames,"visible_fragments":int(masks.sum()),"frozen_states":before,"frozen_states_unchanged":True,"head_exposure":card["head_exposure"]})
        for path,digest in {**frozen,**reader.opened}.items():
            if sha(contained(path))!=digest:raise ValueError("source changed during inference")
        guard()
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:signal.alarm(0)
    if reader is not None:write(run/"artifacts/source_reads_sha256.json",{str(Path(p).relative_to(PROJECT_ROOT)):h for p,h in reader.opened.items()})
    write(run/"metrics/summary.json",{"status":"DEVELOPMENT_EVALUATED" if not error else "SYSTEM_FAIL","result":result,"error":error,"ledger":ledger,"elapsed_s":time.monotonic()-start,
        "peak_host_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,"scientific_gate_pass":False,"new_labels":0})
    write(run/"RUN_STATE.json",{"state":"COMPLETED" if not error else "FAILED","run_id":run.name,"error":error})
    seal=run/"artifacts/evidence_sha256.txt";seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"error":error,"comparison":result.get("comparison"),"ledger":ledger,"seal_sha256":sha(seal)}));return int(error is not None)


if __name__=="__main__":raise SystemExit(main())
