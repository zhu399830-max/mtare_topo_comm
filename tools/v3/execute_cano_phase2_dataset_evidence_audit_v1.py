#!/usr/bin/env python3
"""Audit the sealed Phase-2 dataset without modifying or re-exporting it."""

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
import open3d as o3d
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_phase2_dataset import (
    ROLE_ORDER, candidate_clusters, cast_ranges, evaluate_frame, frame_contract,
    select_valid_clusters, spline_arrays,
)
from mtare_topo.data.cano_phase2_evidence_audit import (
    compare_scan_replay, event_coverage, frame_failure_reasons,
    selection_identity_audit,
)
from mtare_topo.data.cano_sensor_smoke import interpolate_polyline, lidar_local_directions, world_directions
from mtare_topo.governance import load_json, write_json

REGISTRY = PROJECT_ROOT / "configs/v3/gate1/manifests/cano_phase2_world_registry_v1.json"
M1R = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
SOURCE_RUN = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v1_seed0"
EXPECTED_FRAMES = 112500
EXPECTED_CLUSTERS = 22500
EXPECTED_REJECTED = 61
REPLAY_THRESHOLD_M = 1e-6


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def angle_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    delta = (np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta)))


def plot_preview_page(samples: list[dict], path: Path, page: int) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), constrained_layout=True)
    for index, (axis, sample) in enumerate(zip(axes.ravel(), samples)):
        values = np.ma.masked_where(sample["valid"] == 0, sample["range"])
        image = axis.imshow(values, aspect="auto", origin="lower", vmin=0.3, vmax=50.0, cmap="viridis")
        for heading in sample["headings"]:
            axis.axvline((float(heading) % 360.0) / 0.5, color="white", linewidth=1.1, alpha=0.9)
        target = sample["target"]
        scaled = 13.0 + 2.0 * target / max(float(target.max()), 1e-12)
        axis.plot(np.arange(720), scaled, color="red", linewidth=0.9)
        axis.set_title(
            f"{sample['parent_id']} | {sample['role']} | b={len(sample['headings'])} | "
            f"LOS={'PASS' if all(sample['los']) else 'FAIL'} | valid={sample['valid'].mean():.3f}",
            fontsize=8,
        )
        axis.set_xlabel("azimuth bin (0.5 deg)")
        axis.set_ylabel("LiDAR ring")
        if index == 0:
            axis.text(5, 14.4, "red=teacher target; white=teacher headings; grey=invalid", color="white", fontsize=7, bbox={"facecolor":"black","alpha":0.5})
    fig.colorbar(image, ax=axes, shrink=0.5, label="range (m); invalid pixels masked grey")
    fig.suptitle(f"Gate 1 corrective audit — TRAIN ONLY — range + valid + teacher + LOS — page {page}/10")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial_coverage(recipe: str, worlds: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    colors = {"interior":"#1f77b4", "junction":"#d62728", "terminal":"#2ca02c"}
    for axis, item in zip(axes.ravel(), worlds):
        splines = item["splines"]
        all_z = []
        for points in splines.values():
            axis.plot(points[:, 0], points[:, 1], color="0.75", linewidth=0.7)
            all_z.extend(points[:, 2].tolist())
        for role in ROLE_ORDER:
            xy = []
            for cluster in item["clusters"]:
                if cluster["primary_role"] != role:
                    continue
                point, _ = interpolate_polyline(splines[int(cluster["tunnel_id"])], float(cluster["center_arc_m"]))
                xy.append(point[:2])
            if xy:
                values = np.asarray(xy)
                axis.scatter(values[:, 0], values[:, 1], s=5, c=colors[role], label=role, alpha=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{item['parent_id']} | z span={max(all_z)-min(all_z):.1f} m", fontsize=8)
        axis.set_xlabel("world x (m)"); axis.set_ylabel("world y (m)")
    axes.ravel()[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Gate 1 corrective audit — TRAIN ONLY — {recipe} spline and selected-cluster spatial coverage")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(role_counts: Counter, branch_counts: Counter, valid_ratios: list[float], range_hist: np.ndarray, range_edges: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0,0].bar(ROLE_ORDER, [role_counts[x] for x in ROLE_ORDER]); axes[0,0].set_title("Train cluster primary-role counts")
    branch_keys = sorted(branch_counts); axes[0,1].bar([str(x) for x in branch_keys], [branch_counts[x] for x in branch_keys]); axes[0,1].set_title("Train frame teacher branch counts")
    axes[1,0].hist(valid_ratios, bins=30); axes[1,0].set_xlabel("valid pixel ratio / frame"); axes[1,0].set_title("Train valid-ratio distribution")
    centers=(range_edges[:-1]+range_edges[1:])/2; axes[1,1].plot(centers,range_hist); axes[1,1].set_xlabel("valid return range (m)"); axes[1,1].set_ylabel("pixel count"); axes[1,1].set_title("Train valid-return range distribution")
    fig.suptitle("Gate 1 corrective audit — TRAIN ONLY — dataset distributions")
    fig.savefig(path,dpi=140); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); started = time.monotonic()
    (run_dir/"artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir/"previews/train_complete_samples").mkdir(parents=True, exist_ok=True)
    (run_dir/"previews/train_spatial_coverage").mkdir(parents=True, exist_ok=True)
    (run_dir/"previews/train_distributions").mkdir(parents=True, exist_ok=True)

    registry = load_json(REGISTRY); schema=registry["sampling_contract"]["row_schema"]
    rows=[dict(zip(schema,row)) for row in registry["rows"]]
    sealed_clusters=read_jsonl(SOURCE_RUN/"artifacts/place_clusters.jsonl")
    sealed_frames=read_jsonl(SOURCE_RUN/"artifacts/manifest.jsonl")
    source_world_summaries={item["parent_id"]:item for item in load_json(SOURCE_RUN/"metrics/summary.json")["world_summaries"]}
    clusters_by_parent=defaultdict(list); frames_by_parent=defaultdict(list)
    for item in sealed_clusters: clusters_by_parent[item["parent_id"]].append(item)
    for item in sealed_frames: frames_by_parent[item["parent_id"]].append(item)

    replay_stream=(run_dir/"artifacts/replay_audit.jsonl").open("w",encoding="utf-8")
    rejected_stream=(run_dir/"artifacts/rejected_candidates.jsonl").open("w",encoding="utf-8")
    world_metrics=[]; total_frames=0; total_clusters=0; total_rejected=0
    global_max_stored=0.0; global_max_independent=0.0; replay_failures=0; teacher_failures=0
    recipe_worlds=defaultdict(list); preview_pool=defaultdict(lambda: defaultdict(list))
    train_roles=Counter(); train_branches=Counter(); train_valid_ratios=[]
    range_edges=np.linspace(0.3,50.0,101); range_hist=np.zeros(100,dtype=np.int64)

    for world_index,row in enumerate(rows):
        parent=row["parent_id"]; split=row["split"]
        source=M1R/f"artifacts/meshes/{parent}/primary"
        graph=load_json(source/"graph.json"); spl_doc=load_json(source/"splines.json")
        splines=spline_arrays(spl_doc); fta=float(load_json(source/"geometry_parameters.json")["fta_distance_m"])
        scene_a=scene_from_mesh(source/"mesh.obj"); scene_b=scene_from_mesh(source/"mesh.obj")
        directions_local=lidar_local_directions(); candidates=candidate_clusters(parent,graph,spl_doc); cache={}

        def validate(cluster):
            evaluated=[]
            for frame in frame_contract(cluster,spl_doc,fta):
                result=evaluate_frame(scene_a,frame,splines,o3d)
                directions=world_directions(directions_local,float(frame["yaw_deg"]))
                second_range,second_valid=cast_ranges(scene_b,np.asarray(frame["sensor_xyz_m"]),directions,o3d)
                independent=compare_scan_replay(result["range_m"],result["valid_mask"],result["range_m"],result["valid_mask"],second_range,second_valid,REPLAY_THRESHOLD_M)
                reasons=frame_failure_reasons(result,independent)
                evaluated.append((frame,result,second_range,second_valid,independent,reasons))
            cache[cluster["cluster_id"]]=evaluated
            return all(not item[5] for item in evaluated)

        quota={role:int(row[f"quota_{role}"]) for role in ROLE_ORDER}
        selected,rejected=select_valid_clusters(candidates,quota,validate)
        identity=selection_identity_audit(selected,clusters_by_parent[parent])
        if not identity["passed"]:
            raise RuntimeError(f"{parent}: selected identity mismatch: {identity}")
        expected_rejected=int(source_world_summaries[parent]["rejected_clusters"])
        if len(rejected)!=expected_rejected:
            raise RuntimeError(f"{parent}: rejected count mismatch {len(rejected)} != {expected_rejected}")

        shard=zarr.open_group(str(SOURCE_RUN/f"artifacts/dataset/{split}/{parent}.zarr"),mode="r")
        stored_range=shard["range_m"][:]; stored_valid=shard["valid_mask"][:]; stored_target=shard["exit_target"][:]; stored_branch=shard["branch_count"][:]
        sealed_world_frames=frames_by_parent[parent]
        if stored_range.shape[0]!=len(selected)*5 or len(sealed_world_frames)!=len(selected)*5:
            raise RuntimeError(f"{parent}: sealed frame cardinality mismatch")
        cursor=0; world_stored_max=0.0; world_independent_max=0.0; world_replay_failures=0; world_teacher_failures=0
        for cluster in selected:
            for frame,result,second_range,second_valid,independent,_ in cache[cluster["cluster_id"]]:
                record=sealed_world_frames[cursor]
                if record["frame_id"]!=f"{cluster['cluster_id']}_f{frame['frame_index']}":
                    raise RuntimeError(f"{parent}: frame identity mismatch at {cursor}")
                replay=compare_scan_replay(stored_range[cursor],stored_valid[cursor],result["range_m"],result["valid_mask"],second_range,second_valid,REPLAY_THRESHOLD_M)
                target_max=float(np.max(np.abs(stored_target[cursor]-result["exit_target"])))
                heading_max=angle_difference(record["headings_robot_deg"],result["headings_robot_deg"])
                teacher_pass=bool(target_max<=1e-6 and int(stored_branch[cursor])==int(result["branch_count"]) and int(record["branch_count"])==int(result["branch_count"]) and heading_max<=1e-9 and record["branch_los"]==result["branch_los"] and abs(float(record["minimum_horizontal_clearance_m"])-float(result["minimum_horizontal_clearance_m"]))<=1e-6)
                audit_record={"frame_id":record["frame_id"],"cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"replay":replay,"teacher_passed":teacher_pass,"teacher_target_max_difference":target_max,"teacher_heading_max_difference_deg":heading_max}
                replay_stream.write(json.dumps(audit_record,separators=(",",":"))+"\n")
                replay_failures+=int(not replay["passed"]); world_replay_failures+=int(not replay["passed"]); teacher_failures+=int(not teacher_pass); world_teacher_failures+=int(not teacher_pass)
                world_stored_max=max(world_stored_max,replay["stored_to_a_max_difference_m"]); world_independent_max=max(world_independent_max,replay["scene_a_to_b_max_difference_m"])
                if split=="train":
                    train_branches[int(result["branch_count"])]+=1; train_valid_ratios.append(replay["valid_ratio"])
                    valid_values=stored_range[cursor][stored_valid[cursor].astype(bool)]; range_hist+=np.histogram(valid_values,bins=range_edges)[0]
                    if frame["frame_index"]==2 and len(preview_pool[parent[:3]][cluster["primary_role"]])<4:
                        preview_pool[parent[:3]][cluster["primary_role"]].append({"parent_id":parent,"role":cluster["primary_role"],"range":stored_range[cursor],"valid":stored_valid[cursor],"target":stored_target[cursor],"headings":result["headings_robot_deg"],"los":result["branch_los"]})
                cursor+=1
        for cluster in rejected:
            frames=[]
            for frame,result,_,_,independent,reasons in cache[cluster["cluster_id"]]:
                frames.append({"frame_index":frame["frame_index"],"arc_m":frame["arc_m"],"eligible":result["eligible"],"branch_count":result["branch_count"],"minimum_horizontal_clearance_m":result["minimum_horizontal_clearance_m"],"branch_los":result["branch_los"],"replay":independent,"failure_reasons":reasons})
            reasons=sorted({reason for item in frames for reason in item["failure_reasons"]})
            if not reasons: raise RuntimeError(f"{parent}: rejected candidate without reason: {cluster['cluster_id']}")
            rejected_stream.write(json.dumps({"cluster_id":cluster["cluster_id"],"parent_id":parent,"split":split,"primary_role":cluster["primary_role"],"failure_reasons":reasons,"frames":frames},separators=(",",":"))+"\n")

        coverage=event_coverage(selected)
        metric={"parent_id":parent,"split":split,"candidate_clusters":len(candidates),"selected_clusters":len(selected),"rejected_clusters":len(rejected),"selected_identity":identity,"frames":cursor,"replay_failures":world_replay_failures,"teacher_failures":world_teacher_failures,"stored_to_a_max_difference_m":world_stored_max,"scene_a_to_b_max_difference_m":world_independent_max,"event_coverage":coverage}
        write_json(run_dir/f"metrics/{parent}.json",metric); world_metrics.append(metric)
        total_frames+=cursor; total_clusters+=len(selected); total_rejected+=len(rejected)
        global_max_stored=max(global_max_stored,world_stored_max); global_max_independent=max(global_max_independent,world_independent_max)
        if split=="train":
            train_roles.update(cluster["primary_role"] for cluster in selected)
            recipe_worlds[parent[:3]].append({"parent_id":parent,"splines":splines,"clusters":selected})
        print(json.dumps({"world":parent,"index":world_index+1,"of":len(rows),"selected":len(selected),"rejected":len(rejected),"stored_max":world_stored_max,"independent_max":world_independent_max}),flush=True)

    replay_stream.close(); rejected_stream.close()
    for recipe_index in range(1,11):
        recipe=f"S{recipe_index:02d}"; pool=preview_pool[recipe]
        samples=pool["interior"][:4]+pool["junction"][:3]+pool["terminal"][:3]
        if len(samples)!=10: raise RuntimeError(f"{recipe}: incomplete stratified preview sample set {len(samples)}")
        plot_preview_page(samples,run_dir/f"previews/train_complete_samples/{recipe}.png",recipe_index)
        if len(recipe_worlds[recipe])!=8: raise RuntimeError(f"{recipe}: train spatial world count != 8")
        plot_spatial_coverage(recipe,recipe_worlds[recipe],run_dir/f"previews/train_spatial_coverage/{recipe}.png")
    plot_distributions(train_roles,train_branches,train_valid_ratios,range_hist,range_edges,run_dir/"previews/train_distributions/aggregate.png")

    junction_total=sum(item["event_coverage"]["junction_event_count"] for item in world_metrics)
    terminal_total=sum(item["event_coverage"]["terminal_event_count"] for item in world_metrics)
    passed=bool(len(rows)==90 and total_frames==EXPECTED_FRAMES and total_clusters==EXPECTED_CLUSTERS and total_rejected==EXPECTED_REJECTED and replay_failures==0 and teacher_failures==0 and global_max_stored<=REPLAY_THRESHOLD_M and global_max_independent<=REPLAY_THRESHOLD_M and junction_total==634 and terminal_total==575)
    summary={"schema_version":"cano_phase2_dataset_evidence_audit_v1","overall_status":"PASS_CANO_PHASE2_DATASET_EVIDENCE_AUDIT_V1" if passed else "FAIL_CANO_PHASE2_DATASET_EVIDENCE_AUDIT_V1","source_dataset_mutated":False,"worlds":len(rows),"train_worlds":80,"validation_worlds":10,"development_test_worlds_read":0,"mtare_worlds_read":0,"clusters":total_clusters,"frames":total_frames,"rejected_candidates":total_rejected,"replay_failures":replay_failures,"teacher_failures":teacher_failures,"stored_to_a_global_max_difference_m":global_max_stored,"scene_a_to_b_global_max_difference_m":global_max_independent,"junction_events_covered":junction_total,"terminal_events_covered":terminal_total,"complete_sample_pages":10,"spatial_coverage_figures":10,"distribution_figures":1,"training_samples_consumed":0,"models":0,"duration_seconds":time.monotonic()-started,"world_metrics":world_metrics}
    write_json(run_dir/"metrics/summary.json",summary)
    if not passed: raise RuntimeError(f"corrective evidence audit failed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
