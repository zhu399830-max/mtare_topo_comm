#!/usr/bin/env python3
"""Execute the approved eligibility-first Phase-2 Cano V2 export."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import zarr
from numcodecs import Blosc

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_phase2_dataset import (
    ROLE_ORDER, candidate_clusters, cast_ranges, evaluate_frame, frame_contract,
    spline_arrays,
)
from mtare_topo.data.cano_phase2_evidence_audit import (
    compare_scan_replay, event_coverage, frame_failure_reasons,
)
from mtare_topo.data.cano_phase2_selector_v2 import select_spatially_balanced_clusters
from mtare_topo.data.cano_sensor_smoke import interpolate_polyline, lidar_local_directions, world_directions
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline

REGISTRY=PROJECT_ROOT/"configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json"
M1R=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
SELECTOR=PROJECT_ROOT/"results/gate1_data/gate1_20260812_cano_phase2_selector_v2_feasibility_audit_seed0"
EXPECTED_CANDIDATES=27247
EXPECTED_CANDIDATE_FRAMES=136235
EXPECTED_SELECTED=22500
EXPECTED_FRAMES=112500
REPLAY_THRESHOLD_M=1e-6


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles(): raise RuntimeError(f"empty mesh: {path}")
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return scene


def create_shard(path: Path, frames: int) -> zarr.Group:
    group=zarr.open_group(str(path),mode="w");codec=Blosc(cname="zstd",clevel=5,shuffle=Blosc.BITSHUFFLE)
    for name,shape,chunks,dtype in (
        ("range_m",(frames,16,720),(10,16,720),"f4"),("valid_mask",(frames,16,720),(20,16,720),"u1"),
        ("exit_target",(frames,720),(100,720),"f4"),("pose_world_sensor",(frames,7),(1000,7),"f8"),
        ("ray_origin_world",(frames,3),(1000,3),"f8"),("branch_count",(frames,),(4000,),"u1"),
    ): group.create_dataset(name,shape=shape,chunks=chunks,dtype=dtype,compressor=codec)
    return group


def pose7(frame: dict) -> np.ndarray:
    half=np.radians(float(frame["yaw_deg"]))/2.0
    return np.asarray([*frame["sensor_xyz_m"],0.0,0.0,np.sin(half),np.cos(half)],dtype=np.float64)


def plot_samples(samples: list[dict], path: Path, page: int) -> None:
    fig,axes=plt.subplots(5,2,figsize=(18,16),constrained_layout=True)
    for axis,sample in zip(axes.ravel(),samples):
        values=np.ma.masked_where(sample["valid"]==0,sample["range"]);image=axis.imshow(values,aspect="auto",origin="lower",vmin=.3,vmax=50,cmap="viridis")
        for heading in sample["headings"]: axis.axvline((heading%360.0)/.5,color="white",linewidth=1)
        axis.plot(np.arange(720),13+2*sample["target"]/max(float(sample["target"].max()),1e-12),color="red",linewidth=.8)
        axis.set_title(f"{sample['parent_id']} | {sample['role']} | b={len(sample['headings'])} | LOS=PASS | valid={sample['valid'].mean():.3f}",fontsize=8)
        axis.set_xlabel("azimuth bin");axis.set_ylabel("LiDAR ring")
    fig.colorbar(image,ax=axes,shrink=.5,label="range (m); grey=invalid; red=teacher; white=headings")
    fig.suptitle(f"Gate 1 V2 — TRAIN ONLY — complete samples — page {page}/10");fig.savefig(path,dpi=130);plt.close(fig)


def plot_spatial(recipe: str, worlds: list[dict], path: Path) -> None:
    fig,axes=plt.subplots(2,4,figsize=(20,10),constrained_layout=True);colors={"interior":"#1f77b4","junction":"#d62728","terminal":"#2ca02c"}
    for axis,item in zip(axes.ravel(),worlds):
        for points in item["splines"].values(): axis.plot(points[:,0],points[:,1],color=".75",linewidth=.7)
        for role in ROLE_ORDER:
            xy=[interpolate_polyline(item["splines"][int(c["tunnel_id"])],float(c["center_arc_m"]))[0][:2] for c in item["clusters"] if c["primary_role"]==role]
            if xy: axis.scatter(np.asarray(xy)[:,0],np.asarray(xy)[:,1],s=5,c=colors[role],label=role)
        axis.set_aspect("equal",adjustable="box");axis.set_title(item["parent_id"],fontsize=8)
    axes.ravel()[0].legend(fontsize=7);fig.suptitle(f"Gate 1 V2 — TRAIN ONLY — {recipe} selected spatial coverage");fig.savefig(path,dpi=130);plt.close(fig)


def plot_distributions(roles: Counter, branches: Counter, ratios: list[float], hist: np.ndarray, edges: np.ndarray, path: Path) -> None:
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    axes[0,0].bar(ROLE_ORDER,[roles[x] for x in ROLE_ORDER]);axes[0,0].set_title("Train cluster roles")
    keys=sorted(branches);axes[0,1].bar([str(x) for x in keys],[branches[x] for x in keys]);axes[0,1].set_title("Train teacher branch counts")
    axes[1,0].hist(ratios,bins=30);axes[1,0].set_title("Train valid-pixel ratio")
    axes[1,1].plot((edges[:-1]+edges[1:])/2,hist);axes[1,1].set_title("Train valid-return ranges");axes[1,1].set_xlabel("range (m)")
    fig.suptitle("Gate 1 V2 — TRAIN ONLY — distributions");fig.savefig(path,dpi=140);plt.close(fig)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();run=args.run_dir.resolve();started=time.monotonic()
    dataset_root=run/"artifacts/dataset";scratch=run/"artifacts/_scratch_candidate_cache"
    for path in (dataset_root,scratch,run/"previews/train_complete_samples",run/"previews/train_spatial_coverage",run/"previews/train_distributions"): path.mkdir(parents=True,exist_ok=True)
    registry=load_json(REGISTRY);schema=registry["sampling_contract"]["row_schema"];rows=[dict(zip(schema,row)) for row in registry["rows"]]
    frame_stream=(run/"artifacts/manifest.jsonl").open("w",encoding="utf-8");cluster_stream=(run/"artifacts/place_clusters.jsonl").open("w",encoding="utf-8")
    audit_stream=(run/"artifacts/candidate_frame_audit.jsonl").open("w",encoding="utf-8");eligibility_stream=(run/"artifacts/candidate_eligibility.jsonl").open("w",encoding="utf-8");rejected_stream=(run/"artifacts/rejected_candidates.jsonl").open("w",encoding="utf-8")
    baseline=RangeExitBaseline();directions_local=lidar_local_directions();world_metrics=[];recipe_worlds=defaultdict(list);preview_pool=defaultdict(lambda:defaultdict(list))
    train_roles=Counter();train_branches=Counter();train_ratios=[];range_edges=np.linspace(.3,50,101);range_hist=np.zeros(100,dtype=np.int64)
    totals=Counter();junction_events=0;terminal_events=0;global_gap=0.0;deterministic_failures=0
    for world_index,row in enumerate(rows):
        parent=row["parent_id"];split=row["split"];source=M1R/f"artifacts/meshes/{parent}/primary"
        graph=load_json(source/"graph.json");spl_doc=load_json(source/"splines.json");splines=spline_arrays(spl_doc);fta=float(load_json(source/"geometry_parameters.json")["fta_distance_m"])
        candidates=candidate_clusters(parent,graph,spl_doc);quota={role:int(row[f"quota_{role}"]) for role in ROLE_ORDER}
        frozen=load_json(SELECTOR/f"metrics/{parent}.json")["selector_audit"];tunnel_quota={int(key):int(value) for key,value in frozen["tunnel_quota"].items()}
        scene_a=scene_from_mesh(source/"mesh.obj");scene_b=scene_from_mesh(source/"mesh.obj");candidate_frames=len(candidates)*5;cache=create_shard(scratch/f"{parent}.zarr",candidate_frames)
        evaluations={};eligible=[];cursor=0
        for cluster in candidates:
            cluster_evaluations=[];cluster_reasons=set()
            for frame in frame_contract(cluster,spl_doc,fta):
                result=evaluate_frame(scene_a,frame,splines,o3d);directions=world_directions(directions_local,float(frame["yaw_deg"]));second_range,second_valid=cast_ranges(scene_b,np.asarray(frame["sensor_xyz_m"]),directions,o3d)
                replay=compare_scan_replay(result["range_m"],result["valid_mask"],result["range_m"],result["valid_mask"],second_range,second_valid,REPLAY_THRESHOLD_M);reasons=frame_failure_reasons(result,replay);cluster_reasons.update(reasons)
                cache["range_m"][cursor]=result["range_m"];cache["valid_mask"][cursor]=result["valid_mask"];cache["exit_target"][cursor]=result["exit_target"];cache["pose_world_sensor"][cursor]=pose7(frame);cache["ray_origin_world"][cursor]=frame["sensor_xyz_m"];cache["branch_count"][cursor]=result["branch_count"]
                metadata={"frame_index":frame["frame_index"],"arc_m":frame["arc_m"],"yaw_deg":frame["yaw_deg"],"branch_count":result["branch_count"],"headings_robot_deg":result["headings_robot_deg"],"minimum_horizontal_clearance_m":result["minimum_horizontal_clearance_m"],"branch_los":result["branch_los"],"replay":replay,"failure_reasons":reasons,"cache_row":cursor}
                cluster_evaluations.append(metadata);audit_stream.write(json.dumps({"frame_id":f"{cluster['cluster_id']}_f{frame['frame_index']}","cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,**metadata},separators=(",",":"))+"\n");cursor+=1
            is_eligible=not cluster_reasons;evaluations[cluster["cluster_id"]]=cluster_evaluations
            eligibility_stream.write(json.dumps({"cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"eligible":is_eligible,"failure_reasons":sorted(cluster_reasons)},separators=(",",":"))+"\n")
            if is_eligible: eligible.append(cluster)
            else: rejected_stream.write(json.dumps({"cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"primary_role":cluster["primary_role"],"failure_reasons":sorted(cluster_reasons),"frames":cluster_evaluations},separators=(",",":"))+"\n")
        if cursor!=candidate_frames: raise RuntimeError(f"{parent}: candidate frame mismatch")
        selected,selector_audit=select_spatially_balanced_clusters(eligible,quota,tunnel_quota=tunnel_quota,constraint_reference=candidates)
        replayed,replay_audit=select_spatially_balanced_clusters(eligible,quota,tunnel_quota=tunnel_quota,constraint_reference=candidates)
        deterministic=([x["cluster_id"] for x in selected]==[x["cluster_id"] for x in replayed] and selector_audit==replay_audit);deterministic_failures+=int(not deterministic)
        if not deterministic: raise RuntimeError(f"{parent}: selector replay mismatch")
        shard=create_shard(dataset_root/split/f"{parent}.zarr",len(selected)*5);shard.attrs.update({"parent_id":parent,"split":split,"student_fields":["range_m","valid_mask"],"teacher_fields":["exit_target"],"forbidden_student_fields":["pose_world_sensor","ray_origin_world","branch_count"]})
        out=0;role_counts=Counter();branch_counts=Counter();baseline_match=0
        for cluster in selected:
            role_counts[cluster["primary_role"]]+=1;cluster_stream.write(json.dumps({**cluster,"split":split,"eligible":True},separators=(",",":"))+"\n");items=evaluations[cluster["cluster_id"]]
            rows5=[item["cache_row"] for item in items]
            if rows5!=list(range(rows5[0],rows5[0]+5)): raise RuntimeError(f"{parent}: non-contiguous cache rows")
            for name in ("range_m","valid_mask","exit_target","pose_world_sensor","ray_origin_world","branch_count"): shard[name][out:out+5]=cache[name][rows5[0]:rows5[0]+5]
            for offset,item in enumerate(items):
                rr=cache["range_m"][item["cache_row"]];vv=cache["valid_mask"][item["cache_row"]];tt=cache["exit_target"][item["cache_row"]]
                pred=baseline.predict(rr,vv,np.arange(-15,16,2,dtype=np.float64));baseline_match+=int(int(pred["branch_count"])==item["branch_count"]);branch_counts[item["branch_count"]]+=1
                record={"frame_id":f"{cluster['cluster_id']}_f{item['frame_index']}","cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"primary_role":cluster["primary_role"],"near_junction":cluster["near_junction"],"near_terminal":cluster["near_terminal"],"tunnel_id":cluster["tunnel_id"],"frame_in_cluster":item["frame_index"],"arc_m":item["arc_m"],"yaw_deg":item["yaw_deg"],"branch_count":item["branch_count"],"headings_robot_deg":item["headings_robot_deg"],"minimum_horizontal_clearance_m":item["minimum_horizontal_clearance_m"],"branch_los":item["branch_los"],"replay":item["replay"],"candidate_eligible":True,"zarr_row":out+offset}
                frame_stream.write(json.dumps(record,separators=(",",":"))+"\n")
                if split=="train":
                    train_branches[item["branch_count"]]+=1;train_ratios.append(float(vv.mean()));range_hist+=np.histogram(rr[vv.astype(bool)],bins=range_edges)[0]
                    role=cluster["primary_role"];recipe=parent[:3]
                    if item["frame_index"]==2 and len(preview_pool[recipe][role])<({"interior":4,"junction":3,"terminal":3}[role]): preview_pool[recipe][role].append({"parent_id":parent,"role":role,"range":rr,"valid":vv,"target":tt,"headings":item["headings_robot_deg"]})
            out+=5
        events=event_coverage(selected);junction_events+=events["junction_event_count"];terminal_events+=events["terminal_event_count"];global_gap=max(global_gap,selector_audit["maximum_candidate_to_selected_same_tunnel_arc_distance_m"])
        metric={"parent_id":parent,"split":split,"candidate_clusters":len(candidates),"candidate_frames":candidate_frames,"eligible_clusters":len(eligible),"rejected_clusters":len(candidates)-len(eligible),"selected_clusters":len(selected),"frames":out,"role_counts":dict(role_counts),"branch_counts":dict(branch_counts),"baseline_branch_count_accuracy":baseline_match/out,"selector_audit":selector_audit,"selector_deterministic_replay":deterministic,"event_coverage":events}
        write_json(run/f"metrics/{parent}.json",metric);world_metrics.append(metric);totals.update({"candidate_clusters":len(candidates),"candidate_frames":candidate_frames,"eligible_clusters":len(eligible),"rejected_clusters":len(candidates)-len(eligible),"selected_clusters":len(selected),"frames":out})
        if split=="train": train_roles.update(role_counts);recipe_worlds[parent[:3]].append({"parent_id":parent,"splines":splines,"clusters":selected})
        shutil.rmtree(scratch/f"{parent}.zarr");print(json.dumps({"world":parent,"index":world_index+1,"of":90,"eligible":len(eligible),"rejected":len(candidates)-len(eligible),"selected":len(selected)}),flush=True)
    for stream in (frame_stream,cluster_stream,audit_stream,eligibility_stream,rejected_stream): stream.close()
    if totals["candidate_clusters"]!=EXPECTED_CANDIDATES or totals["candidate_frames"]!=EXPECTED_CANDIDATE_FRAMES or totals["selected_clusters"]!=EXPECTED_SELECTED or totals["frames"]!=EXPECTED_FRAMES: raise RuntimeError(f"aggregate cardinality mismatch: {dict(totals)}")
    if junction_events!=634 or terminal_events!=575 or deterministic_failures or global_gap>10+1e-9: raise RuntimeError("aggregate selector evidence mismatch")
    shutil.rmtree(scratch)
    for index,recipe in enumerate(sorted(preview_pool),1):
        samples=preview_pool[recipe]["interior"]+preview_pool[recipe]["junction"]+preview_pool[recipe]["terminal"]
        if len(samples)!=10: raise RuntimeError(f"{recipe}: incomplete stratified preview {len(samples)}")
        plot_samples(samples,run/f"previews/train_complete_samples/page_{index:02d}_{recipe}.png",index)
    for recipe,worlds in sorted(recipe_worlds.items()):
        if len(worlds)!=8: raise RuntimeError(f"{recipe}: expected 8 train worlds")
        plot_spatial(recipe,worlds,run/f"previews/train_spatial_coverage/{recipe}.png")
    plot_distributions(train_roles,train_branches,train_ratios,range_hist,range_edges,run/"previews/train_distributions/train_distribution.png")
    summary={"schema_version":"cano_phase2_dataset_summary_v2","overall_status":"PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2","worlds":90,"train_worlds":80,"validation_worlds":10,"development_test_worlds_read":0,"mtare_worlds_read":0,**dict(totals),"dual_scene_candidate_scans":EXPECTED_CANDIDATE_FRAMES*2,"junction_events_covered":junction_events,"terminal_events_covered":terminal_events,"maximum_candidate_to_selected_same_tunnel_arc_distance_m":global_gap,"selector_deterministic_replay_failures":deterministic_failures,"selected_replay_failures":0,"selected_teacher_failures":0,"teacher_labels":EXPECTED_FRAMES,"training_samples_consumed":0,"models":0,"duration_seconds":time.monotonic()-started,"world_summaries":world_metrics}
    write_json(run/"metrics/summary.json",summary);write_json(run/"artifacts/world_manifest.json",{"worlds":world_metrics});return 0


if __name__=="__main__": raise SystemExit(main())
