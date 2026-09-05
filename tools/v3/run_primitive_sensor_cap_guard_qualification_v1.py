#!/usr/bin/env python3
"""Formal full-population qualification of deterministic sloped-cap guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import world_unique_frame_poses
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, cross_section_area, guard_vertical_sensor_offset_at_caps, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


RUN_ID = "gate3_20260830_primitive_sensor_cap_guard_qualification_v1_seed0"
PASS = "PASS_PRIMITIVE_SENSOR_CAP_GUARD_QUALIFICATION_V1"
FAIL = "FAIL_PRIMITIVE_SENSOR_CAP_GUARD_QUALIFICATION_V1"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAVERSAL_PATH = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"; files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files: stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _traversals():
    result = {}
    with TRAVERSAL_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line); result.setdefault(str(row["parent_id"]), []).append(row)
    return result


def _minimum(field, points):
    parts = []
    for start in range(0, len(points), 2048): parts.append(np.min(field.operand_signed_distances_sparse(points[start:start+2048]), axis=1))
    return np.concatenate(parts)


def _plot(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    axes[0].bar(["before", "after"], [summary["before_outside_gt_1e9"], summary["after_outside_gt_1e9"]], color=["#c65a50", "#57965c"]); axes[0].set(ylabel="variant frames outside union", title="All 757,290 sensor origins")
    axes[1].hist(summary["positive_cap_extensions_m"], bins=30, color="#4e79a7"); axes[1].set(xlabel="cap extension (m)", ylabel="endpoint caps", title="Deterministic guard lengths")
    for suffix in ("png", "pdf", "svg"): fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir/"RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("cap qualification executes exactly once")
    started=time.monotonic(); overall,error=FAIL,None
    try:
        if spec.get("gate")!=3 or spec.get("operation")!="audit" or spec.get("user_authorization",{}).get("status")!="APPROVED": raise RuntimeError("scope/authorization mismatch")
        for relative,expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT/relative)!=expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT/record["path"])!=record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        env=os.environ.copy();env["PYTHONPATH"]=str(PROJECT_ROOT/"src")
        tests=subprocess.run(["/home/zeng-workstation/anaconda3/bin/python","-m","pytest","-q","tests/v3/unit/test_primitive_construction_supervisor.py","tests/v3/unit/test_primitive_provenance_field.py","tests/v3/unit/test_swept_superellipse_field.py","tests/v3/unit/test_geometry_variant_contract.py","tests/v3/unit/test_primitive_relation_dataset.py"],cwd=PROJECT_ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (run_dir/"logs/unit_tests.log").write_text(tests.stdout,encoding="utf-8")
        if tests.returncode or "36 passed" not in tests.stdout: raise RuntimeError("expected 36 unit tests")
        card=load_json(PROJECT_ROOT/spec["data_card"]);worlds=tuple(card["worlds"]["train"]);traversals=_traversals()
        before_count=after_count=raw_after_positive=0;before_max=after_max=0.;extensions=[];conflicts=[];rows=[];frames=primitives_total=0;maximum_area_error=0.
        for world in worlds:
            primary=MESH_ROOT/world/"primary";graph=load_json(primary/"graph.json");splines=load_json(primary/"splines.json");geometry=load_json(primary/"geometry_parameters.json")
            construction=build_primitive_construction_graph(graph,splines,geometry,endpoint_attachment_mode="free_space_overlap",node_degree_source="edge_incidence")
            poses=world_unique_frame_poses(parent_id=world,traversal_manifest=traversals[world],graph=graph,spline_document=splines,geometry_parameters=geometry);frames+=len(poses.global_frame_indices)
            world_before=world_after=0
            incident={}; primitive_index={value.primitive_id:index for index,value in enumerate(construction.primitives)}
            for composition in construction.compositions:
                members={primitive_index[value.primitive_id] for value in composition.member_endpoints}
                for endpoint in composition.member_endpoints: incident[(primitive_index[endpoint.primitive_id],endpoint.endpoint_index)]=members
            for realization in GeometryRealization:
                source=realize_construction(world,construction,realization);guarded,records=guard_vertical_sensor_offset_at_caps(source,vertical_sensor_offset_m=float(geometry["fta_distance_m"])+1.,discretization_guard_m=.025);primitives_total+=len(source)
                for old,new,record in zip(source,guarded,records):
                    if old.primitive_id!=new.primitive_id or old.endpoint_half_axes_m!=new.endpoint_half_axes_m or old.endpoint_shape_exponent!=new.endpoint_shape_exponent or not np.array_equal(old.centerline_xyz_m[1:-1],new.centerline_xyz_m[1:-1]): raise RuntimeError("guard changed non-cap primitive semantics")
                    extensions.extend(value for value in (record.start_extension_m,record.end_extension_m) if value>0)
                    for axes,exponent in zip(new.endpoint_half_axes_m,new.endpoint_shape_exponent): maximum_area_error=max(maximum_area_error,abs(cross_section_area(axes,exponent)-np.pi*construction.primitives[primitive_index[new.primitive_id]].radius_m**2))
                before=_minimum(SweptSuperellipseProvenanceField(source,spacing_m=.025),poses.sensor_xyz_m); after_field=SweptSuperellipseProvenanceField(guarded,spacing_m=.025);after=_minimum(after_field,poses.sensor_xyz_m)
                world_before+=int(np.sum(before>1e-9));world_after+=int(np.sum(after>1e-9));raw_after_positive+=int(np.sum(after>0));before_max=max(before_max,float(np.max(before)));after_max=max(after_max,float(np.max(after)))
                original_field=SweptSuperellipseProvenanceField(source,spacing_m=.025)
                extension_points=[];extension_meta=[]
                for index,(new,record) in enumerate(zip(guarded,records)):
                    for endpoint_index,extension in enumerate((record.start_extension_m,record.end_extension_m)):
                        if extension<=0: continue
                        extension_points.append(new.centerline_xyz_m[0 if endpoint_index==0 else -1])
                        extension_meta.append((index,endpoint_index,new.primitive_id,realization.value))
                if extension_points:
                    extension_values=original_field.operand_signed_distances(np.asarray(extension_points))
                    for values,(index,endpoint_index,primitive_id,realization_value) in zip(extension_values,extension_meta):
                        forbidden=[candidate for candidate,value in enumerate(values) if value<=0 and candidate not in incident[(index,endpoint_index)]]
                        if forbidden: conflicts.append({"world":world,"realization":realization_value,"primitive":primitive_id,"endpoint":endpoint_index,"nonincident":forbidden})
            before_count+=world_before;after_count+=world_after;rows.append({"world":world,"frames":len(poses.global_frame_indices),"before_outside_gt_1e9":world_before,"after_outside_gt_1e9":world_after})
        checks={"exact_80_worlds":len(rows)==80,"exact_252430_source_frames":frames==252430,"exact_757290_variant_frames":3*frames==757290,"exact_24117_variant_primitives":primitives_total==24117,"reproduce_402_meaningful_outside_before":before_count==402,"zero_meaningful_outside_after":after_count==0,"after_residual_le_1e9":after_max<=1e-9,"positive_bounded_extensions":bool(extensions) and max(extensions)<=.5,"zero_nonincident_overlap":not conflicts,"area_error_le_1e10":maximum_area_error<=1e-10}
        overall=PASS if all(checks.values()) else FAIL
        summary={"schema_version":"primitive_sensor_cap_guard_qualification_v1","overall_status":overall,"scientific_pass":overall==PASS,"checks":checks,"before_outside_gt_1e9":before_count,"after_outside_gt_1e9":after_count,"raw_after_positive_le_roundoff":raw_after_positive,"before_maximum_residual_m":before_max,"after_maximum_residual_m":after_max,"positive_cap_count":len(extensions),"positive_cap_extensions_m":extensions,"maximum_cap_extension_m":max(extensions),"total_cap_extension_m":sum(extensions),"maximum_area_error_m2":maximum_area_error,"nonincident_conflicts":conflicts,"worlds":rows,"decision":"ALLOW_SLOT_CAPACITY_V1R_WITH_GUARDED_CAPS" if overall==PASS else "STOP_P1_CAP_GUARD_FAILED","c09_c10_worlds_read":0,"optimizer_steps":0,"model_inference_frames":0,"graph_replays":0,"duration_seconds":time.monotonic()-started,"error":None}
        write_json(run_dir/"metrics/summary.json",summary);write_json(run_dir/"artifacts/cap_guard_provenance.json",{"schema_version":"primitive_sensor_cap_guard_v1","vertical_offset":"fta_distance_m + 1.0m","extension_rule":"only outward axial component plus 0.025m field cell","rows":rows});write_json(run_dir/"config/environment.json",{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"numpy":np.__version__});_plot(summary,run_dir/"previews/primitive_sensor_cap_guard")
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}";(run_dir/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8");write_json(run_dir/"metrics/summary.json",{"overall_status":FAIL,"scientific_pass":False,"error":error})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error});entries=_seal(run_dir);print(json.dumps({"overall_status":overall,"error":error,"evidence_files":entries},indent=2));return 0 if error is None else 2


if __name__=="__main__":raise SystemExit(main())
