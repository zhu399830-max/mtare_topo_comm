#!/usr/bin/env python3
"""Execute the approved Phase-2 Cano range/teacher export exactly as frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
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
    select_valid_clusters, spline_arrays,
)
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline

REGISTRY = PROJECT_ROOT / "configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json"
M1R = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
EXPECTED_TOTAL_FRAMES = 112500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def create_shard(path: Path, frames: int) -> zarr.Group:
    group = zarr.open_group(str(path), mode="w")
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group.create_dataset("range_m", shape=(frames,16,720), chunks=(16,16,720), dtype="f4", compressor=codec)
    group.create_dataset("valid_mask", shape=(frames,16,720), chunks=(32,16,720), dtype="u1", compressor=codec)
    group.create_dataset("exit_target", shape=(frames,720), chunks=(128,720), dtype="f4", compressor=codec)
    group.create_dataset("pose_world_sensor", shape=(frames,7), chunks=(1024,7), dtype="f8", compressor=codec)
    group.create_dataset("ray_origin_world", shape=(frames,3), chunks=(1024,3), dtype="f8", compressor=codec)
    group.create_dataset("branch_count", shape=(frames,), chunks=(4096,), dtype="u1", compressor=codec)
    return group


def pose7(frame: dict) -> np.ndarray:
    yaw = np.radians(float(frame["yaw_deg"])) / 2.0
    return np.asarray([*frame["sensor_xyz_m"],0.0,0.0,np.sin(yaw),np.cos(yaw)], dtype=np.float64)


def preview_page(records: list[dict], ranges: list[np.ndarray], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20,7), constrained_layout=True)
    for axis, record, values in zip(axes.ravel(), records, ranges):
        axis.imshow(values, aspect="auto", origin="lower", vmin=0.3, vmax=50.0, cmap="viridis")
        axis.set_title(f"{record['parent_id']}\n{record['primary_role']} b={record['branch_count']}", fontsize=8)
        axis.set_xlabel("azimuth column"); axis.set_ylabel("ring")
    fig.suptitle(title)
    fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir=args.run_dir.resolve(); started=time.monotonic()
    registry=load_json(REGISTRY); schema=registry["sampling_contract"]["row_schema"]
    rows=[dict(zip(schema,row)) for row in registry["rows"]]
    mesh_manifest=load_json(M1R/"artifacts/mesh_manifest.json")
    mesh_records={item["parent_id"]:item for item in mesh_manifest["parents"]}
    dataset_root=run_dir/"artifacts/dataset"; dataset_root.mkdir(parents=True,exist_ok=True)
    (run_dir/"previews/train_recipe_coverage").mkdir(parents=True,exist_ok=True)
    (run_dir/"previews/train_cluster_pages").mkdir(parents=True,exist_ok=True)
    frame_manifest=(run_dir/"artifacts/manifest.jsonl").open("w",encoding="utf-8")
    cluster_manifest=(run_dir/"artifacts/place_clusters.jsonl").open("w",encoding="utf-8")
    world_summaries=[]; preview_records=[];preview_ranges=[];preview_counts=Counter(); baseline=RangeExitBaseline(); total_frames=0
    for world_index,row in enumerate(rows):
        parent=row["parent_id"]; split=row["split"]
        source=M1R/f"artifacts/meshes/{parent}/primary"
        graph=load_json(source/"graph.json"); spl_doc=load_json(source/"splines.json")
        fta=float(load_json(source/"geometry_parameters.json")["fta_distance_m"])
        candidates=candidate_clusters(parent,graph,spl_doc)
        capacity=Counter(item["primary_role"] for item in candidates)
        expected=[row[f"capacity_{role}"] for role in ROLE_ORDER]
        if [capacity[role] for role in ROLE_ORDER] != expected:
            raise RuntimeError(f"{parent}: registry capacity mismatch")
        scene1=scene_from_mesh(source/"mesh.obj"); scene2=scene_from_mesh(source/"mesh.obj")
        splines=spline_arrays(spl_doc); cache={}; second_local=lidar_local_directions()
        def validate(cluster):
            frames=frame_contract(cluster,spl_doc,fta); evaluated=[]
            for frame in frames:
                first=evaluate_frame(scene1,frame,splines,o3d)
                directions=world_directions(second_local,float(frame["yaw_deg"]))
                second_range,second_valid=cast_ranges(scene2,np.asarray(frame["sensor_xyz_m"]),directions,o3d)
                common=first["valid_mask"].astype(bool)&second_valid.astype(bool)
                replay=bool(np.array_equal(first["valid_mask"],second_valid) and (not np.any(common) or np.max(np.abs(first["range_m"][common]-second_range[common]))<=1e-6))
                first["independent_scene_replay_passed"]=replay; evaluated.append((frame,first))
            ok=all(item[1]["eligible"] and item[1]["independent_scene_replay_passed"] for item in evaluated)
            cache[cluster["cluster_id"]]=evaluated
            return ok
        quota={role:int(row[f"quota_{role}"]) for role in ROLE_ORDER}
        selected,rejected=select_valid_clusters(candidates,quota,validate)
        shard_frames=len(selected)*5; shard=create_shard(dataset_root/split/f"{parent}.zarr",shard_frames)
        shard.attrs.update({"parent_id":parent,"split":split,"student_fields":["range_m","valid_mask"],"teacher_fields":["exit_target"],"forbidden_student_fields":["pose_world_sensor","ray_origin_world","branch_count"]})
        role_counts=Counter(); branch_counts=Counter(); baseline_match=0; cursor=0
        for cluster in selected:
            role_counts[cluster["primary_role"]]+=1
            cluster_manifest.write(json.dumps({k:v for k,v in cluster.items() if k not in ()},separators=(",",":"))+"\n")
            for frame,result in cache[cluster["cluster_id"]]:
                shard["range_m"][cursor]=result["range_m"];shard["valid_mask"][cursor]=result["valid_mask"]
                shard["exit_target"][cursor]=result["exit_target"];shard["pose_world_sensor"][cursor]=pose7(frame)
                shard["ray_origin_world"][cursor]=frame["sensor_xyz_m"];shard["branch_count"][cursor]=result["branch_count"]
                pred=baseline.predict(result["range_m"],result["valid_mask"],np.arange(-15,16,2,dtype=np.float64))
                baseline_match+=int(int(pred["branch_count"])==result["branch_count"]);branch_counts[result["branch_count"]]+=1
                record={"frame_id":f"{cluster['cluster_id']}_f{frame['frame_index']}","cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"primary_role":cluster["primary_role"],"near_junction":cluster["near_junction"],"near_terminal":cluster["near_terminal"],"tunnel_id":cluster["tunnel_id"],"frame_in_cluster":frame["frame_index"],"arc_m":frame["arc_m"],"yaw_deg":frame["yaw_deg"],"branch_count":result["branch_count"],"headings_robot_deg":result["headings_robot_deg"],"minimum_horizontal_clearance_m":result["minimum_horizontal_clearance_m"],"branch_los":result["branch_los"]}
                frame_manifest.write(json.dumps(record,separators=(",",":"))+"\n")
                recipe=parent[:3]
                if split=="train" and preview_counts[recipe]<10 and frame["frame_index"]==2:
                    preview_records.append(record);preview_ranges.append(result["range_m"])
                    preview_counts[recipe]+=1
                cursor+=1
        if cursor!=shard_frames: raise RuntimeError(f"{parent}: frame write mismatch")
        summary={"parent_id":parent,"split":split,"candidate_clusters":len(candidates),"selected_clusters":len(selected),"rejected_clusters":len(rejected),"frames":shard_frames,"role_counts":dict(role_counts),"branch_counts":dict(branch_counts),"baseline_branch_count_accuracy":baseline_match/shard_frames,"zarr_path":str((dataset_root/split/f'{parent}.zarr').relative_to(PROJECT_ROOT))}
        world_summaries.append(summary);write_json(run_dir/f"metrics/{parent}.json",summary);total_frames+=shard_frames
        print(json.dumps({"world":parent,"index":world_index+1,"of":len(rows),"frames":shard_frames,"rejected":len(rejected)}),flush=True)
    frame_manifest.close();cluster_manifest.close()
    if total_frames!=EXPECTED_TOTAL_FRAMES: raise RuntimeError(f"frame total {total_frames}")
    for page in range(10): preview_page(preview_records[page*10:(page+1)*10],preview_ranges[page*10:(page+1)*10],run_dir/f"previews/train_cluster_pages/page_{page+1:02d}.png",f"GATE1 TRAIN ONLY fixed page {page+1}/10")
    # Recipe coverage is machine-backed and train-only: one bar chart per recipe.
    for recipe in range(1,11):
        prefix=f"S{recipe:02d}_"; subset=[x for x in world_summaries if x["split"]=="train" and x["parent_id"].startswith(prefix)]
        fig,ax=plt.subplots(figsize=(10,4)); ax.bar(range(len(subset)),[x["selected_clusters"] for x in subset]);ax.set_xticks(range(len(subset)),[x["parent_id"].split("_")[-1] for x in subset]);ax.set_ylabel("selected clusters");ax.set_title(f"Gate1 train-only recipe S{recipe:02d} coverage");fig.tight_layout();fig.savefig(run_dir/f"previews/train_recipe_coverage/S{recipe:02d}.png",dpi=130);plt.close(fig)
    summary={"schema_version":"cano_phase2_dataset_summary_v1","overall_status":"PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V1","worlds":90,"train_worlds":80,"validation_worlds":10,"development_test_worlds_read":0,"mtare_worlds_read":0,"clusters":22500,"frames":total_frames,"teacher_labels":total_frames,"training_samples_consumed":0,"models":0,"duration_seconds":time.monotonic()-started,"world_summaries":world_summaries}
    write_json(run_dir/"metrics/summary.json",summary);write_json(run_dir/"artifacts/world_manifest.json",{"worlds":world_summaries})
    return 0

if __name__ == "__main__": raise SystemExit(main())
