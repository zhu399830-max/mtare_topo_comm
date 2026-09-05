#!/usr/bin/env python3
"""Generate one new S10 C08 perception mesh and audit complete floor support."""
from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,open3d as o3d
from _bootstrap import PROJECT_ROOT
import execute_cano_100_parent_perception_mesh_contract_m0 as m0
import execute_cano_100_parent_perception_mesh_m1r as m1r
from mtare_topo.data.cano_perception_mesh_contract import canonical_json_hash,floor_support_contract
from mtare_topo.governance import load_json,write_json
PARENT="S10_3d_complex_C08"; OLD=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S10_3d_complex_C08/primary"; TRAJ=PROJECT_ROOT/"results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0/artifacts/S10_3d_complex_C08_trajectory.npz"; TOL=.25
def cast_down(mesh_path,origins):
 m=o3d.io.read_triangle_mesh(str(mesh_path),enable_post_processing=False);s=o3d.t.geometry.RaycastingScene();s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m));d=np.tile([[0,0,-1]],(len(origins),1));r=np.concatenate((origins.astype(np.float32),d.astype(np.float32)),1);return s.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy().astype(float)
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);run=p.parse_args().run_dir.resolve();started=time.monotonic()
 try:
  parents=load_json(m0.SOURCE_V2R/"artifacts/accepted_parent_manifest.json")["parents"];parent=next(x for x in parents if x["parent_id"]==PARENT);stratum=m0._stratum_registry()[parent["source_stratum_id"]];dest=run/"artifacts/replacement/S10_3d_complex_C08/primary"
  primary,_,graph,splines=m0._materialize(parent,stratum,dest,role="replacement_primary");primary,vertices=m1r._sanitize_materialization(dest,primary,graph,splines)
  oldmat=load_json(OLD/"materialization.json");identity={"graph_identity_equal":primary["graph_identity"]==oldmat["graph_identity"],"spline_identity_equal":primary["spline_identity"]==oldmat["spline_identity"],"parent_identity_equal":primary["parent_identity"]==oldmat["parent_identity"],"operation_trace_equal":primary["operation_trace_sha256"]==oldmat["operation_trace_sha256"],"effective_geometry_parameters_equal":primary["effective_geometry_parameter_sha256"]==oldmat["effective_geometry_parameter_sha256"],"new_mesh_sha_differs_from_old":primary["mesh_sha256"]!=oldmat["mesh_sha256"]}
  if not all(identity.values()):raise RuntimeError(f"replacement identity failed: {identity}")
  t=np.load(TRAJ);axis=t["xyz_m"].astype(float);fta=float(load_json(dest/"geometry_parameters.json")["fta_distance_m"]);query=axis.copy();query[:,2]+=2.;hit=cast_down(dest/"mesh.obj",query);observed=query[:,2]-hit;expected=axis[:,2]+fta;audit=floor_support_contract(expected,observed,TOL)
  write_json(run/"metrics/floor_support.json",{**audit,"tolerance_m":TOL,"frame338":{"expected_floor_z_m":float(expected[338]),"observed_floor_z_m":float(observed[338]) if np.isfinite(observed[338]) else None},"frame2275":{"expected_floor_z_m":float(expected[2275]),"observed_floor_z_m":float(observed[2275]) if np.isfinite(observed[2275]) else None}})
  arcs=t["route_arc_m"];err=np.abs(observed-expected);fig,ax=plt.subplots(1,2,figsize=(16,6),constrained_layout=True);limit=min(len(vertices),50000);ids=np.linspace(0,len(vertices)-1,limit,dtype=int);v=vertices[ids];ax[0].scatter(v[:,0],v[:,2],s=.1,alpha=.25);ax[0].plot(axis[:,0],axis[:,2],c="red",lw=.6);ax[0].set_title("replacement mesh XZ + full trajectory axis");ax[1].plot(arcs,np.minimum(err,5),lw=.6);ax[1].axhline(TOL,c="r",ls="--");bad=np.array(audit["unsupported_indices"]);ax[1].scatter(arcs[bad],np.minimum(err[bad],5),s=8,c="red");ax[1].set_title("floor-height error (clipped at 5m)");fig.savefig(run/"previews/S10_replacement_complete_mesh_floor_support.png",dpi=160);plt.close(fig)
  passed=bool(primary["mesh_audit"]["passed"] and primary["sanitation"]["passed"] and audit["passed"])
  summary={"schema_version":"cano_s10_perception_mesh_replacement_v1","overall_status":"PASS_CANO_S10_PERCEPTION_MESH_REPLACEMENT_V1" if passed else "FAIL_CANO_S10_PERCEPTION_MESH_REPLACEMENT_V1","asset_id":"S10_3d_complex_C08_replacement_v1","old_mesh_sha256":oldmat["mesh_sha256"],"new_mesh_sha256":primary["mesh_sha256"],"identity_checks":identity,"mesh_audit_passed":primary["mesh_audit"]["passed"],"sanitation_passed":primary["sanitation"]["passed"],"floor_support":audit,"trajectory_frames_checked":len(axis),"floor_query_rays":len(axis),"training_samples_consumed":0,"inference_frames":0,"graph_updates":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"duration_seconds":time.monotonic()-started,"claim_boundary":"Replacement perception asset and floor-support qualification only; not trajectory/LiDAR/topology/planner qualification."};write_json(run/"metrics/summary.json",summary);print(json.dumps(summary),flush=True);return 0 if passed else 2
 except Exception as e:
  write_json(run/"metrics/executor_failure.json",{"exception_type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()});raise
if __name__=="__main__":raise SystemExit(main())
