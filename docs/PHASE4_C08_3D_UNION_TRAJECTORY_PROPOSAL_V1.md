# Phase 4 C08 Local 3-D Union and Trajectory Qualification Proposal V1

Status: `DESIGN_APPROVED_NOT_EXECUTION_APPROVED`  
Date: 2026-08-14  
Gate: 4 / C08 development only

## Question and scope

The Gate-4 question remains whether frozen M1D local semantics can causally
construct a topometric graph.  This proposal does **not** answer that question.
It only proposes a replacement dynamic-navigation geometry/trajectory
qualification after the native perception mesh and fixed-XY floor approach
failed.

The only worlds are the existing C08 train-split development worlds:

| World | Directed traversals | Frames at 2 m | Role |
|---|---:|---:|---|
| S01_flat_tree_small_C08 | sealed existing route | 801 | development |
| S06_3d_branch_medium_C08 | sealed existing route | 1,414 | development |
| S10_3d_complex_C08 | sealed existing route | 2,558 | development |
| Total | 624 / 312 doubled edges | 4,773 | development |

The frozen order, world identity, directed traversal identity, arc coordinate,
frame index and 2 m sample count remain unchanged.  The native Cano mesh
continues to be the perception/LiDAR asset and is not regenerated or changed.
No C09/C10 or M-TARE world may be read.  No LiDAR, M1D/B0 inference, graph
update, model selection, training or planner execution is in scope.

## Method identity

The prior floor-ribbon method produced separate, overlapping inclined floor
surfaces.  The rejected fixed-XY variant then tried to select one of those
surfaces.  This proposal instead builds a *separate collision/support asset*
from TNG/splines, radii and FTA only, before it reads any route.

1. For each tunnel spline segment, define its void as the radius-swept 3-D
   tube using the frozen tunnel radius.  The collision boundary is the zero
   isosurface of the complement of this void, not a floor ribbon.
2. At an explicit graph node, union only the void fields of its explicitly
   incident tunnels within a node-local 3-D ball and a fixed local spline-arc
   window.  This removes internal walls at an intended junction.
3. A non-incident tunnel is never placed in the same local union.  Any
   non-incident overlap/near-contact within the conservative robot envelope is
   an asset failure, not a reason to merge topology or filter ray hits.
4. Extract the local boundary by a deterministic voxel-SDF/marching-cubes
   procedure, stitch it to non-junction tube boundaries, and preserve the
   contributing tunnel/node IDs on every primitive for auditing.
5. Only after the collision asset is frozen, solve a route-local sensor pose.
   Outside a frozen local node neighbourhood, XY is exactly the sealed route;
   within it, XY/z are jointly optimized with a deterministic objective:
   minimum squared displacement from the sealed route, then minimum first and
   second differences, subject to the unchanged clearance contract.  The route
   is never an input to geometry generation.

The optimizer may move only frames whose sealed arc lies in a node-local
window, may not reorder frames or traverse a different edge, and must record
every nonzero correction.  It may not delete frames, interpolate across an
invalid region, lower a threshold, or select a different mesh resolution from
the final qualification result.

## Frozen design parameters and convergence

Before a formal run, the implementation must expose the following parameters
in a proposal spec (not hard-code them): robot collision radius, vertical
envelope, incident-node 3-D radius rule, local arc rule, SDF padding, and three
predeclared voxel sizes.  The intended resolution ladder is `0.10 m`, `0.05
m`, and `0.025 m`; it is a convergence audit, not a tuning sweep.

The collision qualification resolution is the finest level only if all three
levels preserve: (a) topology-isolated union membership, (b) no open boundary
near a qualifying trajectory, and (c) 4,773-frame pass/fail identity.  Any
resolution-dependent pass/fail result, non-manifold/open local surface, or
non-incident envelope overlap is a failure requiring a new user decision.

## Baseline, fallback and evidence

The baseline is the sealed native perception mesh plus the rejected
geometry-first/fixed-XY attempt; it supports only `2234/2558` S10 poses in the
replacement audit and has no valid complete C08 replay asset.  There is no
parameter-tuned fallback.  If the 3-D union cannot pass the stated checks, the
fallback is to stop Gate 4 geometry work and retain only the existing offline
evidence.

A future one-shot formal qualification run must preserve:

- immutable input manifest and hashes for the three C08 graph/spline/radius
  documents and sealed trajectories;
- one collision asset per world, source-primitive provenance, voxel fields,
  resolution-convergence table and mesh integrity report;
- 4,773 corrected-pose records, displacement/continuity metrics and complete
  horizontal/up/down ray audit;
- explicit failures for non-incident overlaps, open/non-manifold surfaces,
  clearance failures, route/frame drift or resolution disagreement; and
- raw log, config, data card, command, environment identity and SHA-256 seal.

Pass requires every one of the 4,773 frames to meet the unchanged horizontal
clearance `>=0.8 m`, down distance `1.00 +/- 0.05 m`, up clearance `>=0.8 m`,
and the frozen route continuity limits, at all predeclared resolutions.  A
pass only restores C08 geometry eligibility; it does not execute C08 causal
replay or advance Gate 4.

## Authorization boundary

The user approved this design on 2026-08-14.  That approval authorizes
documentation and implementation preparation only.  It does not authorize
asset materialization, trajectory writing, the 4,773-frame audit,
`preflight.py`, `create_run.py`, C08 replay, or any downstream operation.
Those actions require an exact data card and run spec followed by a separate
explicit user approval.
