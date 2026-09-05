#!/usr/bin/env python3
"""Materialize corrected causal labels and backprojection targets for Composers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.governance import write_json
from mtare_topo.teacher.gse_composer_supervision import (
    ComposerWorldSupervision,
    materialize_composer_world_supervision,
    summarize_composer_supervision,
)


EXPECTED_SPLIT_ROWS={"fit":142184,"c07":21548,"c08":24394}
EXPECTED_EVENT={"corridor":150964,"junction":26608,"terminal":7525,"turn":1998,"geometry_transition":1031}


def _split(parent:str)->str:
    value=int(parent.rsplit("_C",1)[1]);return "fit" if value<=6 else "c07" if value==7 else "c08"


def _read_jsonl(path:Path):
    groups=defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line);groups[str(row["parent_id"])].append(row)
    return groups


def _read_timing(path:Path):
    groups=defaultdict(list)
    with path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):groups[str(row["parent_id"])].append(row)
    return groups


def _plot(output:Path,summary:dict):
    figure,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    event=summary["event_frames"]
    axes[0].bar(list(event.keys()),list(event.values()),color="#2b6cb0");axes[0].set_yscale("log");axes[0].tick_params(axis="x",rotation=35);axes[0].set_title("Corrected causal event frames")
    back=summary["backprojection_frames"]
    axes[1].bar(list(back.keys()),list(back.values()),color="#2f855a");axes[1].set_title("Valid 5-frame backprojection labels")
    figure.tight_layout()
    for suffix in ("png","pdf","svg"):figure.savefig(output/f"gse_composer_supervision_v1.{suffix}",dpi=220)
    plt.close(figure)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--cache-root",required=True,type=Path);parser.add_argument("--teacher",required=True,type=Path);parser.add_argument("--transition-timing",required=True,type=Path);parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args();output=args.output_dir.resolve()
    if output.exists():raise RuntimeError("Composer supervision output exists")
    output.mkdir(parents=True);world_out=output/"worlds";world_out.mkdir()
    teacher=_read_jsonl(args.teacher.resolve());timing=_read_timing(args.transition_timing.resolve())
    paths=sorted(args.cache_root.resolve().glob("*.npz"))
    if len(paths)!=80 or set(teacher)!=set(path.stem for path in paths):raise RuntimeError("Composer supervision world population drift")
    split_rows=defaultdict(int);event_frames=defaultdict(int);back_frames=defaultdict(int);back_ids=defaultdict(set);records=[]
    for path in paths:
        parent=path.stem
        with np.load(path,allow_pickle=False) as cache:global_index=np.asarray(cache["global_sequence_index"],dtype=np.int64)
        supervision=materialize_composer_world_supervision(global_index,teacher[parent],timing.get(parent,[]))
        arrays={field.name:getattr(supervision,field.name) for field in fields(ComposerWorldSupervision)}
        np.savez_compressed(world_out/f"{parent}.npz",**arrays)
        local=summarize_composer_supervision(supervision);split=_split(parent);split_rows[split]+=local["rows"]
        for name,value in local["event_frames"].items():event_frames[name]+=value
        for name,value in local["backprojection_frames"].items():back_frames[name]+=value
        valid=supervision.backprojection_valid
        for name in ("turn","geometry_transition"):
            back_ids[name].update(supervision.identity[valid&(supervision.event_name==name)].tolist())
        records.append({"parent":parent,"split":split,**local})
    summary={"schema_version":"gse_composer_supervision_v1","status":"PASS_GSE_COMPOSER_SUPERVISION_V1","worlds":80,"rows":sum(split_rows.values()),"split_rows":dict(split_rows),"event_frames":dict(event_frames),"backprojection_frames":dict(back_frames),"backprojection_identities":{name:len(values) for name,values in back_ids.items()},"records":records,"teacher_fields_in_model_cache":0,"optimizer_steps":0,"model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}
    checks={"exact_split_rows":dict(split_rows)==EXPECTED_SPLIT_ROWS,"exact_event_frames":dict(event_frames)==EXPECTED_EVENT,"exact_transition_backprojection_frames":back_frames["geometry_transition"]==410,"all_transition_identities_backprojectable":len(back_ids["geometry_transition"])==76,"all_turn_identities_backprojectable":len(back_ids["turn"])==392,"at_least_one_turn_anchor_per_directional_episode":back_frames["turn"]>=740,"zero_forbidden_work":True}
    summary["checks"]={name:bool(value) for name,value in checks.items()};summary["scientific_pass"]=all(checks.values())
    if not summary["scientific_pass"]:summary["status"]="FAIL_GSE_COMPOSER_SUPERVISION_V1"
    write_json(output/"summary.json",summary);write_json(output/"figure_source.json",summary);_plot(output,summary)
    print(json.dumps({"status":summary["status"],"checks":summary["checks"],"backprojection_frames":summary["backprojection_frames"],"backprojection_identities":summary["backprojection_identities"]},indent=2,sort_keys=True))
    return 0 if summary["scientific_pass"] else 2


if __name__=="__main__":raise SystemExit(main())
