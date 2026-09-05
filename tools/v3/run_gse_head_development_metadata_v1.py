#!/usr/bin/env python3
"""Freeze deterministic C02 row references without geometry/model access."""
import argparse
import json
from pathlib import Path
import platform
import resource
import signal
import time
import traceback
import numpy as np
import zarr
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_head_development import validate_head_development_metadata_card
from mtare_topo.data.gse_head_development_selection import HeadDevelopmentMetadataReader
from run_gse_composition_inventory_v1 import sha,contained


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--spec",required=True,type=Path);parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args();spec=load_json(args.spec);run=args.run_dir.resolve()
    if (run!=PROJECT_ROOT/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
            or load_json(run/"config/run_spec.json")!=spec):raise ValueError("exact fresh metadata run required")
    start=time.monotonic();result={};error=None
    def deadline(signum,frame):raise TimeoutError("120s metadata limit")
    signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
    write(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        card=load_json(contained(spec["data_card"]));report=validate_head_development_metadata_card(card)
        if not report.passed or spec["operation"]!="audit" or load_json(run/"config/data_card.json")!=card:raise ValueError("metadata card mismatch")
        frozen={**card["sealed_sources"],**spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path))!=digest:raise ValueError("metadata/source drift")
        versions={"python":platform.python_version(),"numpy":np.__version__,"zarr":zarr.__version__}
        if versions!=spec["expected_versions"]:raise ValueError("version drift")
        write(run/"config/execution_environment.json",versions)
        # Seal index is provenance only; open only explicit C02 index/mask keys.
        seals={}
        prefixes=[card["source_roots"]["teacher"]+"/"+task+".zarr/" for task in card["tasks"]]
        for seal in card["source_seals"].values():
            with contained(seal).open() as stream:
                for line in stream:
                    digest,path=line.strip().split(None,1)
                    suffix=next((path[len(prefix):] for prefix in prefixes if path.startswith(prefix)),None)
                    if suffix is not None and (suffix in (".zgroup",".zattrs") or suffix.split("/")[0] in card["fields"]):
                        seals[str(contained(path))]=digest
        reader=HeadDevelopmentMetadataReader(contained(card["source_roots"]["teacher"]),card["tasks"],seals)
        selected=[];parents=[]
        with (run/"logs/metadata.jsonl").open("x") as log:
            for task in card["tasks"]:
                rows,summary=reader.read_task(task);selected.extend(rows);parents.append(summary)
                log.write(json.dumps(summary)+"\n");log.flush()
        if len(selected)!=180 or len({(r["task"],r["source_global_sequence_index"]) for r in selected})!=180:raise ValueError("population drift")
        result={"observations":180,"parents":10,"frame_references":900,"unique_source_frames":sum(p["unique_source_frames"] for p in parents),
            "visible_fragments":sum(p["visible_fragments"] for p in parents),"selected_traversals":sum(p["selected_traversals"] for p in parents),
            "available_sequences":sum(p["available_sequences"] for p in parents),"by_parent":parents,
            "head_exposure":card["head_exposure"],"metadata_files_read":len(reader.opened),
            "model_inference":0,"checkpoint_reads":0,"sensor_frames_decoded":0,"geometry_targets_read":0,"optimizer_steps":0,"new_labels":0,
            "training_ready":False,"scientific_gate_pass":False}
        write(run/"artifacts/selected_rows.json",selected)
        write(run/"artifacts/source_reads_sha256.json",{str(Path(p).relative_to(PROJECT_ROOT)):h for p,h in reader.opened.items()})
        result["selection_sha256"]=sha(run/"artifacts/selected_rows.json")
        for path,digest in {**frozen,**reader.opened}.items():
            if sha(contained(path))!=digest:raise ValueError("input changed during metadata audit")
        result["peak_host_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if result["peak_host_rss_bytes"]>1024**3 or time.monotonic()-start>120:raise RuntimeError("1GiB/120s metadata cap")
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:signal.alarm(0)
    write(run/"metrics/summary.json",{"status":"METADATA_COMPLETE" if not error else "METADATA_FAIL","result":result,"error":error,"elapsed_s":time.monotonic()-start})
    write(run/"RUN_STATE.json",{"state":"COMPLETED" if not error else "FAILED","run_id":run.name,"error":error})
    seal=run/"artifacts/evidence_sha256.txt";seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps({"error":error,"result":result,"seal_sha256":sha(seal)}));return int(error is not None)


if __name__=="__main__":raise SystemExit(main())
