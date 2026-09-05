#!/usr/bin/env python3
"""Build and fully qualify corrected floor-following C08 trajectories."""
from __future__ import annotations
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, open3d as o3d
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_pose_feasibility import compact_support_lateral_field,nearest_common_feasible_offset
from mtare_topo.governance import load_json,write_json

WORLDS=("S01_flat_tree_small_C08","S06_3d_branch_medium_C08","S10_3d_complex_C08")
ASSETS=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAJ=PROJECT_ROOT/"results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0/artifacts"
LATERAL=PROJECT_ROOT/"results/gate4_topology/gate4_20260813_cano_s10_local_lateral_pose_feasibility_v1_seed0/artifacts/pose_xy_candidates.csv"
EXPECTED=(801,1414,2558); RAYS=720; CLEAR=.8; FLOOR=1.; TOL=.05; SUPPORT=10.; MAX_DELTA=.1
GROUPS=((6,2551),(39,2519),(105,108,2507),(600,676))

def scene(path):
 m=o3d.io.read_triangle_mesh(str(path),enable_post_processing=False); s=o3d.t.geometry.RaycastingScene(); s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)); return s
def cast(s,o,d):
 r=np.concatenate((np.asarray(o,np.float32),np.asarray(d,np.float32)),1); return s.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy().astype(float)
def geometry_metrics(xyz):
 steps=np.linalg.norm(np.diff(xyz,axis=0),axis=1); v=np.diff(xyz,axis=0); n=np.linalg.norm(v,axis=1); u=np.divide(v,n[:,None],out=np.zeros_like(v),where=n[:,None]>0)
 turns=np.degrees(np.arccos(np.clip(np.sum(u[:-1]*u[1:],axis=1),-1,1))) if len(u)>1 else np.zeros(0)
 return {"maximum_step_m":float(steps.max()),"p99_step_m":float(np.quantile(steps,.99)),"maximum_turn_deg":float(turns.max()),"p99_turn_deg":float(np.quantile(turns,.99))}
def tangents(x):
 t=np.empty_like(x); t[1:-1]=x[2:]-x[:-2]; t[0]=x[1]-x[0]; t[-1]=x[-1]-x[-2]; return t/np.linalg.norm(t,axis=1)[:,None]
def main():
 a=argparse.ArgumentParser(); a.add_argument("--run-dir",type=Path,required=True); run=a.parse_args().run_dir.resolve(); start=time.monotonic()
 by=defaultdict(dict)
 for r in csv.DictReader(LATERAL.open()): by[int(r["frame_index"])][(round(float(r["x_offset_m"]),1),round(float(r["y_offset_m"]),1))]=r["feasible"]=="True"
 anchors=[nearest_common_feasible_offset(by,g) for g in GROUPS]; anchor_frames=[g[0] for g in GROUPS]
 print(json.dumps({"shared_anchor_groups":[list(g) for g in GROUPS],"shared_offsets_xy_m":anchors}),flush=True)
 total=[]; world_metrics=[]
 for wi,(world,expected) in enumerate(zip(WORLDS,EXPECTED)):
  old=np.load(TRAJ/f"{world}_trajectory.npz"); axis=old["xyz_m"].astype(float); arcs=old["route_arc_m"].astype(float)
  if len(axis)!=expected: raise RuntimeError("frame count drift")
  fta=float(load_json(ASSETS/world/"primary/geometry_parameters.json")["fta_distance_m"]); sensor0=axis.copy(); sensor0[:,2]+=fta+1
  offset=np.zeros((len(axis),2))
  if world.startswith("S10"):
   ap=axis[np.asarray(anchor_frames)]; offset=compact_support_lateral_field(axis,ap,np.asarray(anchors),SUPPORT)
  corrected_sensor=sensor0.copy(); corrected_sensor[:,:2]+=offset
  s=scene(ASSETS/world/"primary/mesh.obj")
  down=cast(s,corrected_sensor,np.tile([[0,0,-1]],(len(axis),1)))
  if np.any(~np.isfinite(down)) or np.any(down>8): raise RuntimeError(f"floor evidence failed {world}")
  corrected_sensor[:,2]=corrected_sensor[:,2]-down+FLOOR
  corrected_axis=corrected_sensor.copy(); corrected_axis[:,2]-=fta+1
  az=np.radians(np.arange(RAYS)*.5); dirs=np.stack((np.cos(az),np.sin(az),np.zeros(RAYS)),1)
  horizontal=[]; verify_down=[]; up=[]
  for i,o in enumerate(corrected_sensor):
   h=cast(s,np.broadcast_to(o,(RAYS,3)),dirs); finite=h[np.isfinite(h)&(h>=0)]; horizontal.append(float(finite.min()) if len(finite) else float("nan"))
   v=cast(s,np.broadcast_to(o,(2,3)),np.asarray([[0,0,-1],[0,0,1]],float)); verify_down.append(float(v[0])); up.append(float(v[1]))
  horizontal=np.asarray(horizontal); verify_down=np.asarray(verify_down); up=np.asarray(up)
  passed=np.isfinite(horizontal)&(horizontal>=CLEAR)&np.isfinite(verify_down)&(np.abs(verify_down-FLOOR)<=TOL)&np.isfinite(up)&(up>=CLEAR)
  oldgm=geometry_metrics(axis); newgm=geometry_metrics(corrected_axis); delta=np.linalg.norm(np.diff(offset,axis=0),axis=1)
  continuous=bool(delta.max()<=MAX_DELTA+1e-9 and newgm["maximum_step_m"]<=oldgm["maximum_step_m"]+.1 and newgm["maximum_turn_deg"]<=oldgm["maximum_turn_deg"]+1.)
  metric={"world":world,"frames":len(axis),"passed_frames":int(passed.sum()),"failed_frames":int((~passed).sum()),"minimum_horizontal_clearance_m":float(np.nanmin(horizontal)),"minimum_upward_clearance_m":float(np.nanmin(up)),"maximum_floor_error_m":float(np.nanmax(np.abs(verify_down-FLOOR))),"maximum_lateral_offset_m":float(np.linalg.norm(offset,axis=1).max()),"maximum_adjacent_lateral_delta_m":float(delta.max()),"baseline_geometry":oldgm,"corrected_geometry":newgm,"continuity_passed":continuous,"all_frames_passed":bool(passed.all())}
  write_json(run/f"metrics/{world}.json",metric); world_metrics.append(metric)
  np.savez_compressed(run/f"artifacts/{world}_corrected_trajectory.npz",xyz_m=corrected_axis,tangent_world=tangents(corrected_axis),route_arc_m=arcs,sensor_xyz_m=corrected_sensor,lateral_offset_xy_m=offset)
  for i in range(len(axis)): total.append({"world":world,"frame_index":i,"route_arc_m":arcs[i],"horizontal_clearance_m":horizontal[i],"downward_distance_m":verify_down[i],"upward_distance_m":up[i],"passed":bool(passed[i]),"x_offset_m":offset[i,0],"y_offset_m":offset[i,1],"sensor_x_m":corrected_sensor[i,0],"sensor_y_m":corrected_sensor[i,1],"sensor_z_m":corrected_sensor[i,2]})
  fig,ax=plt.subplots(1,3,figsize=(18,5),constrained_layout=True); ax[0].plot(axis[:,0],axis[:,1],lw=.5,label="frozen"); ax[0].plot(corrected_axis[:,0],corrected_axis[:,1],lw=.5,label="corrected"); ax[0].legend(); ax[0].set_title("complete XY")
  ax[1].plot(arcs,horizontal,lw=.5); ax[1].axhline(.8,c="r",ls="--"); ax[1].set_title("horizontal clearance")
  ax[2].plot(arcs,verify_down,label="down",lw=.5); ax[2].plot(arcs,up,label="up",lw=.5); ax[2].axhline(.8,c="r",ls="--"); ax[2].axhline(1,c="g",ls=":"); ax[2].legend(); ax[2].set_title("vertical evidence")
  fig.suptitle(world); fig.savefig(run/f"previews/{world}_complete_corrective_audit.png",dpi=150); plt.close(fig)
  print(json.dumps(metric),flush=True)
 fields=list(total[0]);
 with (run/"artifacts/complete_corrected_frame_audit.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(total)
 allpass=all(m["all_frames_passed"] and m["continuity_passed"] for m in world_metrics)
 summary={"schema_version":"cano_c08_floor_following_trajectory_corrective_v1","overall_status":"PASS_CANO_C08_FLOOR_FOLLOWING_TRAJECTORY_CORRECTIVE_V1" if allpass else "FAIL_CANO_C08_FLOOR_FOLLOWING_TRAJECTORY_CORRECTIVE_V1","worlds":3,"frames":len(total),"horizontal_rays":len(total)*RAYS,"vertical_rays":len(total)*3,"all_frames_passed":allpass,"shared_anchor_groups":[list(g) for g in GROUPS],"shared_offsets_xy_m":[list(x) for x in anchors],"support_radius_m":SUPPORT,"world_metrics":world_metrics,"inference_frames":0,"graph_updates":0,"training_samples_consumed":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"duration_seconds":time.monotonic()-start}
 write_json(run/"metrics/summary.json",summary); print(json.dumps(summary),flush=True); return 0 if allpass else 2
if __name__=="__main__": raise SystemExit(main())
