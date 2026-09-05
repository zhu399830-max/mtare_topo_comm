# Phase 4 C08 Local Implicit-Union Qualification Proposal V2

Status: `DRAFT_AFTER_V1_DENSE_COST_FAILURE`  
Date: 2026-08-14  
Supersedes: the full-world dense-grid portion of `PHASE4_C08_3D_UNION_TRAJECTORY_PROPOSAL_V1.md` only.

## Fixed data and purpose

This remains a Gate-4 C08 development-only geometry qualification, not a topometric replay. It reads exactly S01/S06/S10, their 312 doubled edges / 624 directed traversals and 4,773 sealed 2 m frame identities. It reads no C09, C10, M-TARE, LiDAR, model output, checkpoint or graph state.

The V1 full-world uniform SDF ladder is invalid: a conservative 5 m radius over 9,541.643029 m has a 0.025 m dense-grid lower bound of 47.96 billion voxels (178.67 GiB for float32 alone). V2 does not use such a grid.

## Geometry and query contract

The global free-space field is analytic. For each sealed spline segment and its frozen tunnel radius, the field evaluates the distance to the finite 3-D swept tube. This global analytic field is not materialized as a voxel volume and supplies exact outside-junction boundary/clearance queries.

Only explicit graph nodes with degree at least three receive a local union window. C08 has exactly 25: S01=4, S06=7 and S10=14. For node `n`, define the local sphere radius as `1.10 * max(radius of n's incident tunnels)` and include only its incident tunnels, only their node-local spline arcs, and only points inside that 3-D sphere. The largest C08 incident radius is 5.979884 m, so the largest sphere is 6.577872 m. Non-incident tunnel volume is never unioned; intersection with the robot safety envelope is an explicit failure.

Inside each local sphere, form the incident void union as an implicit SDF and evaluate it at the three predeclared resolutions 0.10/0.05/0.025 m. Windows are processed serially and their full float fields are discarded after the machine-readable convergence summary, deterministic extracted patch mesh and source-provenance record are saved. A patch must stitch to the analytic outside boundary without an open/non-manifold seam. The final collision query is therefore hybrid and explicit: analytic outside all spheres, local implicit union inside one sphere. Overlapping local spheres are a failure unless their incident sets are identical; they must not silently be coalesced.

## Trajectory and evidence contract

Geometry is complete before any trajectory is read. Then, and only then, the solver may alter XY/z for a frame whose sealed arc belongs to one fixed local node window. It preserves world, traversal, route ordering, arc/frame identity and count. Its deterministic objective is minimum displacement, then minimum first/second difference under the unchanged horizontal `>=0.8m`, down `1.00 +/- 0.05m`, up `>=0.8m` and frozen continuity limits.

Every 4,773 frame is audited with the hybrid field. Frames outside local windows must pass the analytic query. Frames inside must pass all three resolutions and have identical pass/fail identity, with a bounded numerical distance discrepancy documented before execution. A result may restore geometry eligibility only; it cannot run M1D, topology replay or planning.

## Cost and approval boundary

The largest 0.025 m local cube has at most about 527 cells per axis, or about 146 million cells / 0.55 GiB for one float32 field. Serial processing allows a conservative 4 GiB peak-memory cap; extracted patches, trajectories and audit evidence are budgeted at 12 GiB disk and 12 CPU-hours. These are estimates to be tested once, not tunable limits.

V2 changes the geometry and safety-query method, cost and evidence from V1. It needs a new data card, run spec and explicit execution approval. No asset, trajectory, preflight, create_run or qualification action is authorized by this document.
