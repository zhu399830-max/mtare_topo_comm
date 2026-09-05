#!/usr/bin/env python3
"""Read-only 80-parent P0 geometry-variant and resource inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, cross_section_area, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph


RUN_ID = "gate3_20260830_geometry_variant_inventory_v1r2_seed0"
PASS = "PASS_GEOMETRY_VARIANT_INVENTORY_V1R2_P1_PROVENANCE_ACCELERATION_REQUIRED"
FAIL = "FAIL_GEOMETRY_VARIANT_INVENTORY_V1R2"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
OLD_DATA_ROOT = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/artifacts/dataset/train"
HISTORICAL_FRAMES = 252430
HISTORICAL_SEQUENCES = 188126
HISTORICAL_NATIVE_RENDER_FRAMES = 112500
HISTORICAL_NATIVE_RENDER_SECONDS = 1932.1662316770016
PROVENANCE_PROOF_RAYS = 69120
PROVENANCE_PROOF_TWO_PASS_SECONDS = 110.70951770700049
RAYS_PER_FRAME = 16 * 720


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _plot(metrics: dict, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    axes[0].bar(["parents", "realizations", "frames / 10k", "sequences / 10k"], [metrics["parent_worlds"], metrics["planned_realizations"], metrics["planned_frames"]/1e4, metrics["planned_sequences"]/1e4], color=["#4776b4", "#62a66f", "#d78b44", "#9a69ad"])
    axes[0].set_title("Frozen P1 data scale"); axes[0].set_ylabel("count (frames/sequences scaled by 10k)"); axes[0].tick_params(axis="x", rotation=18)
    paths = metrics["resource_projection"]
    axes[1].bar(["native range\nrenderer", "current Python\nprovenance field"], [paths["native_render_hours"], paths["python_provenance_hours"]], color=["#62a66f", "#c95252"])
    axes[1].set_yscale("log"); axes[1].set_ylabel("projected serial hours (log scale)"); axes[1].set_title("Measured-throughput projection")
    axes[1].text(0, paths["native_render_hours"]*1.25, f"{paths['native_render_hours']:.1f} h", ha="center")
    axes[1].text(1, paths["python_provenance_hours"]*1.15, f"{paths['python_provenance_hours']/24:.1f} d", ha="center")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("inventory executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization",{}).get("status") != "APPROVED": raise RuntimeError("scope/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT/relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT/record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        card = load_json(PROJECT_ROOT/spec["data_card"]); worlds = tuple(card["worlds"]["train"])
        if len(worlds) != 80 or any(name.endswith(("C09", "C10")) for name in worlds): raise RuntimeError("expected exactly 80 C01-C08 parent worlds")
        rows = []; edge_total = 0; node_total = 0; source_lengths = 0.; all_axes=[]; all_exponents=[]; maximum_area_error=0.
        offset_endpoint_count = 0; maximum_attachment_error_over_radius = 0.0
        node_degree_mismatch_count = 0
        existing_bytes = 0
        for world in worlds:
            primary = MESH_ROOT/world/"primary"
            graph=load_json(primary/"graph.json"); splines=load_json(primary/"splines.json"); geometry=load_json(primary/"geometry_parameters.json")
            construction=build_primitive_construction_graph(
                graph,
                splines,
                geometry,
                endpoint_attachment_mode="free_space_overlap",
                node_degree_source="edge_incidence",
            )
            edge_total += len(construction.primitives); node_total += len(graph["nodes"]); source_lengths += sum(value.length_m for value in construction.primitives)
            world_offset_endpoints = sum(
                error > .01
                for primitive in construction.primitives
                for error in primitive.endpoint_projection_error_m
            )
            offset_endpoint_count += world_offset_endpoints
            node_degree_mismatch_count += len(construction.node_degree_mismatches)
            maximum_attachment_error_over_radius = max(
                maximum_attachment_error_over_radius,
                max(
                    error / primitive.radius_m
                    for primitive in construction.primitives
                    for error in primitive.endpoint_projection_error_m
                ),
            )
            variant_counts={}
            for realization in GeometryRealization:
                variants=realize_construction(world,construction,realization); variant_counts[realization.value]=len(variants)
                for source,variant in zip(construction.primitives,variants):
                    for axes, exponent in zip(variant.endpoint_half_axes_m,variant.endpoint_shape_exponent):
                        all_axes.extend(axes); all_exponents.append(exponent)
                        maximum_area_error=max(maximum_area_error,abs(cross_section_area(axes,exponent)-math.pi*source.radius_m**2))
            zarr=OLD_DATA_ROOT/f"{world}.zarr"
            if not zarr.is_dir(): raise RuntimeError(f"missing historical train shard: {world}")
            shard_bytes=_directory_bytes(zarr); existing_bytes += shard_bytes
            rows.append({"world":world,"nodes":len(graph["nodes"]),"edges":len(construction.primitives),"length_m":sum(value.length_m for value in construction.primitives),"variant_primitives":variant_counts,"offset_endpoints_gt_0p01m":world_offset_endpoints,"node_degree_mismatches":len(construction.node_degree_mismatches),"historical_zarr_bytes":shard_bytes})
        planned_frames=3*HISTORICAL_FRAMES; planned_sequences=3*HISTORICAL_SEQUENCES; planned_rays=planned_frames*RAYS_PER_FRAME
        native_rate=HISTORICAL_NATIVE_RENDER_FRAMES/HISTORICAL_NATIVE_RENDER_SECONDS
        provenance_rate=2*PROVENANCE_PROOF_RAYS/PROVENANCE_PROOF_TWO_PASS_SECONDS
        resource={
            "measured_historical_native_frames_per_second":native_rate,
            "measured_current_python_provenance_rays_per_second":provenance_rate,
            "native_render_hours":planned_frames/native_rate/3600,
            "python_provenance_hours":planned_rays/provenance_rate/3600,
            "python_provenance_days":planned_rays/provenance_rate/86400,
            "historical_80_world_zarr_bytes":existing_bytes,
            "projected_three_realization_zarr_bytes_same_compression":3*existing_bytes,
            "filesystem_available_bytes_at_preflight":int(__import__("shutil").disk_usage(PROJECT_ROOT).free),
            "current_python_provenance_scalable":planned_rays/provenance_rate/86400 <= 7.0,
        }
        metrics={"schema_version":"geometry_variant_inventory_v1r2","parent_worlds":len(worlds),"planned_realizations":3*len(worlds),"source_edges":edge_total,"planned_variant_primitives":3*edge_total,"source_nodes":node_total,"source_edge_length_m":source_lengths,"planned_frames":planned_frames,"planned_sequences":planned_sequences,"planned_rays":planned_rays,"offset_endpoints_gt_0p01m":offset_endpoint_count,"node_degree_mismatches":node_degree_mismatch_count,"maximum_attachment_error_over_radius":maximum_attachment_error_over_radius,"half_axis_min_m":min(all_axes),"half_axis_max_m":max(all_axes),"shape_exponent_min":min(all_exponents),"shape_exponent_max":max(all_exponents),"maximum_area_preservation_error_m2":maximum_area_error,"resource_projection":resource}
        checks={"exact_80_parent_worlds":len(worlds)==80,"exact_8039_source_edges":edge_total==8039,"exact_5_offset_endpoints":offset_endpoint_count==5,"exact_8_stale_node_degree_records":node_degree_mismatch_count==8,"all_attachment_anchors_inside_declared_free_space":maximum_attachment_error_over_radius<1.0,"exact_240_realizations":metrics["planned_realizations"]==240,"exact_757290_frames":planned_frames==757290,"exact_564378_sequences":planned_sequences==564378,"all_three_variants_per_edge":metrics["planned_variant_primitives"]==3*8039,"area_preservation_le_1e10m2":maximum_area_error<=1e-10,"disk_projection_fits_available":resource["projected_three_realization_zarr_bytes_same_compression"]<.5*resource["filesystem_available_bytes_at_preflight"],"native_render_projection_le_12h":resource["native_render_hours"]<=12,"python_provenance_correctly_classified_unscalable":resource["current_python_provenance_scalable"] is False}
        overall=PASS if all(checks.values()) else FAIL
        metrics.update({"overall_status":overall,"scientific_pass":overall==PASS,"checks":checks,"decision":"REQUIRE_ACCELERATED_PROVENANCE_BACKEND_BEFORE_P1_EXPORT","c09_c10_worlds_read":0,"optimizer_steps":0,"model_inference_frames":0,"graph_replays":0,"duration_seconds":time.monotonic()-started,"error":None})
        write_json(run_dir/"artifacts/world_variant_inventory.json",{"schema_version":"geometry_variant_world_inventory_v1","worlds":rows})
        write_json(run_dir/"metrics/summary.json",metrics); _plot(metrics,run_dir/"previews/geometry_variant_inventory")
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"; (run_dir/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8"); write_json(run_dir/"metrics/summary.json",{"overall_status":FAIL,"scientific_pass":False,"error":error})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error})
    entries=_seal(run_dir); print(json.dumps({"overall_status":overall,"error":error,"evidence_files":entries},indent=2)); return 0 if error is None else 2


if __name__=="__main__": raise SystemExit(main())
