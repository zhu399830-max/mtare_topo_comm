#!/usr/bin/env python3
"""Audit footprint-scale floor support around the S10 edge_0065 mesh hole."""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,open3d as o3d
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_pose_feasibility import common_boolean_grid,feasible_grid_components
from mtare_topo.governance import load_json,write_json
WORLD="S10_3d_complex_C08"; FRAMES=(338,2275); GRID=np.round(np.arange(-1,1.0001,.05),8); N=41
MESH=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S10_3d_complex_C08/primary"
TRAJ=PROJECT_ROOT/"results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0/artifacts/S10_3d_complex_C08_trajectory.npz"
CLEAR=.8; FLOOR=1.; HEIGHT_TOL=.25; QUERY_LIFT=2.; MIN_COMPONENT=9
def scene():
 m=o3d.io.read_triangle_mesh(str(MESH/"mesh.obj"),enable_post_processing=False); s=o3d.t.geometry.RaycastingScene(); s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)); return s
def cast(s,o,d): return s.cast_rays(o3d.core.Tensor(np.concatenate((np.asarray(o,np.float32),np.asarray(d,np.float32)),1)))["t_hit"].numpy().astype(float)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); run=p.parse_args().run_dir.resolve(); started=time.monotonic()
 d=np.load(TRAJ); axis=d["xyz_m"].astype(float); fta=float(load_json(MESH/"geometry_parameters.json")["fta_distance_m"]); s=scene(); observations=[]; allrows=[]
 az=np.radians(np.arange(720)*.5); hdirs=np.stack((np.cos(az),np.sin(az),np.zeros(720)),1)
 for obs,frame in enumerate(FRAMES):
  base=axis[frame]; expected_floor_z=float(base[2]+fta); rows=[]
  for xi,dx in enumerate(GRID):
   for yi,dy in enumerate(GRID):
    q=np.array([base[0]+dx,base[1]+dy,base[2]+QUERY_LIFT]); hit=float(cast(s,q[None,:],np.array([[0,0,-1.]]))[0]); floor_z=float(q[2]-hit) if np.isfinite(hit) else float("nan"); sensor=np.array([q[0],q[1],floor_z+FLOOR])
    support=bool(np.isfinite(hit) and abs(floor_z-expected_floor_z)<=HEIGHT_TOL)
    if support:
     h=cast(s,np.broadcast_to(sensor,(720,3)),hdirs); hf=h[np.isfinite(h)&(h>=0)]; hmin=float(hf.min()) if len(hf) else float("nan"); v=cast(s,np.broadcast_to(sensor,(2,3)),np.array([[0,0,-1.],[0,0,1.]])); down,up=map(float,v)
    else: hmin=down=up=float("nan")
    feasible=bool(support and np.isfinite(hmin) and hmin>=CLEAR and np.isfinite(down) and abs(down-FLOOR)<=.05 and np.isfinite(up) and up>=CLEAR)
    r={"observation_index":obs,"frame_index":frame,"x_index":xi,"y_index":yi,"x_offset_m":float(dx),"y_offset_m":float(dy),"expected_floor_z_m":expected_floor_z,"observed_floor_z_m":floor_z,"floor_height_error_m":abs(floor_z-expected_floor_z) if np.isfinite(floor_z) else float("nan"),"support_passed":support,"horizontal_clearance_m":hmin,"downward_distance_m":down,"upward_distance_m":up,"feasible":feasible}; rows.append(r); allrows.append(r)
  observations.append(rows); print(json.dumps({"frame":frame,"supported":sum(x["support_passed"] for x in rows),"feasible":sum(x["feasible"] for x in rows)}),flush=True)
 common=common_boolean_grid(observations); comps=feasible_grid_components(common); robust=[c for c in comps if len(c)>=MIN_COMPONENT]; members={i for c in robust for i in c}; recommended=min((common[i] for i in members),key=lambda r:(math.hypot(r["x_offset_m"],r["y_offset_m"]),r["x_offset_m"],r["y_offset_m"]),default=None)
 fig,axes=plt.subplots(1,3,figsize=(19,6),constrained_layout=True)
 for j,(rows,title) in enumerate([(observations[0],"frame 338"),(observations[1],"frame 2275"),(common,"common feasible")]):
  if j<2: grid=np.array([x["floor_height_error_m"] for x in rows]).reshape(N,N); im=axes[j].imshow(grid.T,origin="lower",extent=[-1.025,1.025,-1.025,1.025],vmin=0,vmax=1,cmap="magma_r")
  else: grid=np.array([x["feasible"] for x in rows]).reshape(N,N); im=axes[j].imshow(grid.T,origin="lower",extent=[-1.025,1.025,-1.025,1.025],vmin=0,vmax=1,cmap="Greens")
  axes[j].scatter([0],[0],c="red",marker="x",s=80,label="frozen"); axes[j].set_title(title); axes[j].set_xlabel("x offset m"); axes[j].set_ylabel("y offset m"); axes[j].legend(); fig.colorbar(im,ax=axes[j])
 if recommended: axes[2].scatter([recommended["x_offset_m"]],[recommended["y_offset_m"]],c="cyan",marker="*",s=130,label="nearest robust"); axes[2].legend()
 fig.suptitle("S10 edge_0065/tunnel_3 footprint-scale floor support"); fig.savefig(run/"previews/S10_edge0065_floor_support_heatmaps.png",dpi=160); plt.close(fig)
 with (run/"artifacts/floor_support_candidates.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
 write_json(run/"artifacts/common_support_components.json",{"minimum_component_cells":MIN_COMPONENT,"components":[{"cell_count":len(c),"cells":[common[i] for i in c]} for c in comps],"recommended":recommended})
 summary={"schema_version":"cano_s10_floor_support_hole_audit_v1","overall_status":"PASS_CANO_S10_FLOOR_SUPPORT_HOLE_AUDIT_V1","frames":[338,2275],"observations":2,"grid_side":41,"parameter_candidates":len(allrows),"horizontal_rays":sum(r["support_passed"] for r in allrows)*720,"vertical_rays":len(allrows)+sum(r["support_passed"] for r in allrows)*2,"common_feasible_cells":sum(r["feasible"] for r in common),"robust_component_count":len(robust),"robust_common_support_exists":bool(robust),"recommended":recommended,"inference_frames":0,"graph_updates":0,"training_samples_consumed":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"duration_seconds":time.monotonic()-started,"interpretation":"PASS means diagnostic completeness; robust support existence is a separate result."}; write_json(run/"metrics/summary.json",summary); print(json.dumps(summary),flush=True); return 0
if __name__=="__main__": raise SystemExit(main())
