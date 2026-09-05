#!/usr/bin/env python3
"""Execute the approved zero-ray Phase-2 selector-V2 feasibility audit."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_phase2_dataset import ROLE_ORDER, candidate_clusters, spline_arrays
from mtare_topo.data.cano_phase2_evidence_audit import event_coverage
from mtare_topo.data.cano_phase2_selector_v2 import MAXIMUM_COVERAGE_RADIUS_M, select_spatially_balanced_clusters
from mtare_topo.data.cano_sensor_smoke import interpolate_polyline
from mtare_topo.governance import load_json, write_json

REGISTRY=PROJECT_ROOT/"configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json"
M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
V1_RUN=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r",encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]


def v1_spatial_audit(candidates: list[dict], selected: list[dict]) -> dict:
    by_selected=defaultdict(list)
    for item in selected:by_selected[int(item["tunnel_id"])].append(float(item["center_arc_m"]))
    candidate_tunnels={int(item["tunnel_id"]) for item in candidates};missing=sorted(candidate_tunnels-set(by_selected));finite=[]
    for item in candidates:
        tunnel=int(item["tunnel_id"])
        if tunnel in by_selected:finite.append(min(abs(float(item["center_arc_m"])-arc) for arc in by_selected[tunnel]))
    return {"tunnels_without_selected":len(missing),"missing_tunnel_ids":missing,"maximum_represented_tunnel_gap_m":max(finite,default=float("inf"))}


def plot_recipe(recipe: str, worlds: list[dict], path: Path) -> None:
    fig,axes=plt.subplots(2,4,figsize=(20,10),constrained_layout=True);colors={"interior":"#1f77b4","junction":"#d62728","terminal":"#2ca02c"}
    for axis,item in zip(axes.ravel(),worlds):
        for points in item["splines"].values():axis.plot(points[:,0],points[:,1],color="0.78",linewidth=.7)
        for role in ROLE_ORDER:
            points=[]
            for cluster in item["selected"]:
                if cluster["primary_role"]!=role:continue
                point,_=interpolate_polyline(item["splines"][int(cluster["tunnel_id"])],float(cluster["center_arc_m"]));points.append(point[:2])
            if points:
                values=np.asarray(points);axis.scatter(values[:,0],values[:,1],s=5,c=colors[role],label=role,alpha=.85)
        axis.set_aspect("equal",adjustable="box");axis.set_xlabel("world x (m)");axis.set_ylabel("world y (m)")
        axis.set_title(f"{item['parent_id']} | V1 gap={item['v1_gap']:.0f}m/miss={item['v1_missing']} → V2 gap={item['v2_gap']:.0f}m/miss=0",fontsize=8)
    axes.ravel()[0].legend(loc="best",fontsize=7)
    fig.suptitle(f"Gate 1 selector-V2 feasibility — TRAIN ONLY — {recipe} length-balanced selected clusters")
    fig.savefig(path,dpi=130);plt.close(fig)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();run_dir=args.run_dir.resolve();started=time.monotonic()
    (run_dir/"artifacts").mkdir(parents=True,exist_ok=True);(run_dir/"previews/train_spatial_coverage").mkdir(parents=True,exist_ok=True)
    registry=load_json(REGISTRY);schema=registry["sampling_contract"]["row_schema"];rows=[dict(zip(schema,row)) for row in registry["rows"]]
    v1=defaultdict(list)
    for item in read_jsonl(V1_RUN/"artifacts/place_clusters.jsonl"):v1[item["parent_id"]].append(item)
    manifest=(run_dir/"artifacts/selector_v2_clusters.jsonl").open("w",encoding="utf-8");metrics=[];recipe_worlds=defaultdict(list);total_selected=0;total_candidates=0;deterministic_failures=0
    for index,row in enumerate(rows):
        parent=row["parent_id"];split=row["split"];source=M1R/f"artifacts/meshes/{parent}/primary";graph=load_json(source/"graph.json");spl_doc=load_json(source/"splines.json");splines=spline_arrays(spl_doc)
        candidates=candidate_clusters(parent,graph,spl_doc);quota={role:int(row[f"quota_{role}"]) for role in ROLE_ORDER}
        selected,audit=select_spatially_balanced_clusters(candidates,quota);replayed,replay_audit=select_spatially_balanced_clusters(candidates,quota)
        deterministic=[item["cluster_id"] for item in selected]==[item["cluster_id"] for item in replayed];deterministic_failures+=int(not deterministic)
        if not deterministic or audit!=replay_audit:raise RuntimeError(f"{parent}: selector replay drift")
        events=event_coverage(selected);old=v1_spatial_audit(candidates,v1[parent])
        for item in selected:manifest.write(json.dumps({**item,"split":split},separators=(",",":"))+"\n")
        metric={"parent_id":parent,"split":split,"candidate_clusters":len(candidates),"selected_clusters":len(selected),"selector_audit":audit,"deterministic_replay":deterministic,"event_coverage":events,"v1_spatial_baseline":old}
        write_json(run_dir/f"metrics/{parent}.json",metric);metrics.append(metric);total_selected+=len(selected);total_candidates+=len(candidates)
        if split=="train":recipe_worlds[parent[:3]].append({"parent_id":parent,"splines":splines,"selected":selected,"v1_gap":old["maximum_represented_tunnel_gap_m"],"v1_missing":old["tunnels_without_selected"],"v2_gap":audit["maximum_candidate_to_selected_same_tunnel_arc_distance_m"]})
        print(json.dumps({"world":parent,"index":index+1,"of":len(rows),"selected":len(selected),"v1_missing":old["tunnels_without_selected"],"v1_gap":old["maximum_represented_tunnel_gap_m"],"v2_gap":audit["maximum_candidate_to_selected_same_tunnel_arc_distance_m"]}),flush=True)
    manifest.close()
    for recipe_index in range(1,11):
        recipe=f"S{recipe_index:02d}"
        if len(recipe_worlds[recipe])!=8:raise RuntimeError(f"{recipe}: train world count != 8")
        plot_recipe(recipe,recipe_worlds[recipe],run_dir/f"previews/train_spatial_coverage/{recipe}.png")
    junction=sum(x["event_coverage"]["junction_event_count"] for x in metrics);terminal=sum(x["event_coverage"]["terminal_event_count"] for x in metrics);max_gap=max(x["selector_audit"]["maximum_candidate_to_selected_same_tunnel_arc_distance_m"] for x in metrics);missing=sum(not x["selector_audit"]["all_tunnels_represented"] for x in metrics)
    v1_missing_worlds=sum(x["v1_spatial_baseline"]["tunnels_without_selected"]>0 for x in metrics);v1_missing_tunnels=sum(x["v1_spatial_baseline"]["tunnels_without_selected"] for x in metrics);v1_max=max(x["v1_spatial_baseline"]["maximum_represented_tunnel_gap_m"] for x in metrics)
    passed=bool(len(rows)==90 and total_candidates==27247 and total_selected==22500 and deterministic_failures==0 and missing==0 and max_gap<=MAXIMUM_COVERAGE_RADIUS_M and junction==634 and terminal==575)
    summary={"schema_version":"cano_phase2_selector_v2_feasibility_audit_v1","overall_status":"PASS_CANO_PHASE2_SELECTOR_V2_FEASIBILITY_AUDIT" if passed else "FAIL_CANO_PHASE2_SELECTOR_V2_FEASIBILITY_AUDIT","worlds":90,"train_worlds":80,"validation_worlds":10,"development_test_worlds_read":0,"mtare_worlds_read":0,"mesh_worlds_read":0,"raycasts":0,"candidate_clusters":total_candidates,"selected_clusters":total_selected,"frames_proposed_not_generated":total_selected*5,"deterministic_replay_failures":deterministic_failures,"worlds_with_missing_tunnels":missing,"maximum_candidate_to_selected_same_tunnel_arc_distance_m":max_gap,"junction_events_covered":junction,"terminal_events_covered":terminal,"v1_baseline":{"worlds_with_missing_tunnels":v1_missing_worlds,"missing_tunnels":v1_missing_tunnels,"maximum_represented_tunnel_gap_m":v1_max},"train_spatial_figures":10,"data_exported":False,"training_samples_consumed":0,"models":0,"duration_seconds":time.monotonic()-started,"world_metrics":metrics}
    write_json(run_dir/"metrics/summary.json",summary)
    if not passed:raise RuntimeError(f"selector V2 feasibility failed: {summary}")
    return 0


if __name__=="__main__":raise SystemExit(main())
