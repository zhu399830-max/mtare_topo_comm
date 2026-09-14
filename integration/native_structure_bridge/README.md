# Native structure route bridge V1

Current status: the patched native executable compiles, and the transport,
token extraction, retrieval, task lifecycle and route policy have synthetic
tests. A cache-only shadow sidecar records native snapshots and real regional
feedback into the native-region registry; missing model cache returns the exact
original route with `NO_MODEL_CACHE`. **It does not yet join live learned-token
extraction, qualified registration and route policy into a model worker.**
No real learned-token/native-waypoint execution has been recorded. The saved
600-second seed11 baseline contains neither organized `/velodyne_points` nor
native route snapshots, so it cannot supply that missing replay evidence.

This adds advice to the existing M-TARE global route. It does not replace its
global planner, create semantic navigation nodes, or publish `/way_point`.
Only the original planner publishes control. `local_coverage_planner`,
`localPlanner`, `pathFollower`, collision checks, candidate generation, native
connectivity and the native finish rule are unchanged.

## Contract and topics

Under the original planner namespace `/sensor_coverage_planner`:

- `native_structure/candidates`: `std_msgs/String` JSON snapshot.
- `native_structure/advice`: `std_msgs/String` JSON response for that epoch.
- `native_structure/decision`: preserved/native and final routes with reasons.
- `native_structure/feedback`: final actual waypoint, native completion state,
  local viewpoint count, original free-path feedback, native pose, explicit
  region dispatch/visit/status transitions and bridge rejections.

`src/mtare_topo/integration/native_route_advice.py` exposes `validate_snapshot`,
`make_route_advice`, `validate_route_advice` and duplicate-key-rejecting
`decode_advice`. Candidate records expose native cell IDs, native positions and
native numeric statuses in the map frame; they do not label physical branches.
Robot position is included. `registered_scan:<stamp_ns>` keys bind the latest
five accepted registered scans; an external reader must independently verify
the corresponding organized raw scan timestamps, not rename a registered cloud
as a raw observation. The graph has historical inputs beyond these five scans.

The epoch is process-session-specific and binds the current immutable native
snapshot. The response must repeat its timestamp, sources and entire candidate
ID set. Reordering is restricted to the original route's exact ID multiset and
its two depot endpoints. Every advised edge must exist in the unmodified native
cost/feasibility matrices. The declared cost is independently recalculated.
Costs include native integer offsets and are **not metres**. Missing/invalid/
expired advice preserves the exact original route. A native-snapshot defect is
reported, not repaired. Reordering by itself is not evidence of changed control;
the original local planner and lookahead can override its directional effect.

The dedicated advice callback queue waits at most 100 ms; it cannot re-enter
native sensor or execution callbacks. Advice must already use cached features;
this is not an inference budget. Invalid advice cannot carry over to a new
epoch. Disabled mode has no subscription, publication or wait. Native return-
home, multi-robot, communication/rendezvous and two-node route cases bypass
intervention. All bypasses retain the original planner's mission logic.

## Explicit parameters

Set on the original planner's private namespace, not a second planner:

```yaml
native_structure_bridge_mode: shadow  # disabled (default), shadow, active
native_structure_wait_ms: 100
native_structure_max_age_ms: 500
native_structure_max_extra_cost_ratio: 0.10
```

The last ratio is frozen to 0.10 in this version. Every snapshot saves the exact
derived integer budget `original_cost_units // 10`; zero cost permits zero
increase. This is an engineering bound, not a research safety threshold.

## Mechanical overlay and build

`prepare_native_overlay.py --source-root <exact tare_planner> --output-root
<new directory>` verifies both original C++ SHA-256s before creating any output.
It emits only two transformed C++ sources, two new headers and an original/new
hash manifest. It never patches the input tree in place or launches ROS. Apply
the generated overlay only to a disposable source/build copy of the pinned
image and build its existing `tare_planner_node` CMake target. Existing
`std_msgs`, ROS and header-only Boost property_tree are reused; no dependency or
local planner is added. The original `GetNextGlobalSubspace` pass-by-value issue
is deliberately not fixed in this overlay or its baselines.

Pure synthetic core verification:

```sh
g++ -std=c++14 -Wall -Wextra -Werror integration/native_structure_bridge/core_contract_test.cpp -o /tmp/native-route-core-test
/tmp/native-route-core-test
```

No simulation or real observations are needed for this check. Full deployment,
timing/recording checks and data use still require the approved frozen run spec.

The full frozen-image `tare_planner_node` target has also been compiled with
both changed translation units, not just a synthetic header. This is compiled
software, **not ROS callback/closed-loop validation**. Reproduce and preserve
the command, source hashes and full build stdout/stderr with:

```sh
python3 integration/native_structure_bridge/verify_native_build.py --output build/native_structure_bridge
```

The output must be a new directory. This command uses a disposable container,
never launches ROS/Gazebo and never opens a world or observation payload. It
also compiles/runs JSON transport and default-disabled/no-wait software tests
without a ROS master. Native status integers are 0 UNSEEN, 1 EXPLORING, 2
COVERED, 3 COVERED_BY_OTHERS, 4 NOGO and 5 EXPLORING_BY_OTHERS.

## Shadow capture entry point

After installing the generated overlay and building inside the disposable
capture container, the existing case runner can start this single method:

```sh
python3 tools/v3/launch_native_structure_shadow.py --planner-seed 11 --world tunnel --output <case/artifacts/planner_output>
```

The launcher starts `native_structure_advice_node.py`, then the sole native
`tare_planner_node` via `explore_native_shadow.launch`; it stops both children
on shutdown or failure. The surrounding runner still owns the simulator,
recording and frozen runtime. No second global controller or waypoint publisher
is introduced. The sidecar writes `advice_sidecar/trace.jsonl`,
`native_region_tasks.json`, `ready.json` and `summary.json`. Every raw snapshot
and feedback event has a receipt index, raw-message digest and registry verdict.
Advice is published before region bookkeeping. Invalid regional evidence is
reported explicitly; an unexpected consumer exception stops the sidecar safely.
Optional `--cache-manifest` and `--cache-manifest-sha256` load read-only pinned
advice for exact epochs only. A digest is artifact integrity, not model or
geometry qualification, and historical epochs are never rebased for a new run.

`NATIVE_REGION_DISPATCH` names the actual final waypoint's cell, not an assumed
first route cell. `NATIVE_REGION_VISIT` names actual robot cell membership and
preserves the actual pose separately from the cell centre. Both visit and
status carry the most recent per-cell dispatch epoch/evidence ID. All pose
frames and timestamps are copied from `/state_estimation_at_scan`; the pinned
`sensorScanGeneration.cpp` producer explicitly emits `map` / `sensor_at_scan`.
Consumers still check those exact recorded frame names; no alias is invented.

Only the original local residual-coverage branch is marked
`native_coverage_rule_satisfied=true`. Other COVERED heuristics (including loss
of graph connectivity or candidate viewpoints) carry explicit different rules
and false. Native region execution requires its dispatch, post-dispatch actual
membership and that qualified status transition. Candidate disappearance never
means completion. Native regional coverage does not establish channel/direction
coverage, physical exploration completion or continuous graph traversal.
# Optional native planned-path evidence (2026-09-14)

## Resident token prefetch software (not yet a measured live run)

The prefetch node now publishes data-only `/native_structure/token_ready`.
The advice sidecar can opt in with `--live-token-artifacts <relative-directory>
--live-token-inputs <relative-directory> --live-token-max-entries <bound>`.
An independent loader authenticates the recorded input/pose/feature bytes and
fixed checkpoint references; no large file reads run in the advice callback.
Lookup requires the native snapshot's exact five source keys and a cache-ready
time no later than that decision. Wrong publisher, path escape, source/hash
drift and restarts cannot reuse the cache. These options do not yet activate a
task/route policy: `live_token_status` is recorded and
`live_token_used_for_route=false` remains explicit. Only an authenticated,
verified task consumer can provide route-change evidence later.

`tools/v3/native_structure_token_process.py --stream-socket ...
--stream-input-root ... --max-requests ... --max-wall-seconds ...
--preprocessing full_relative_se3_v1 --output ... --resource-output ...`
loads the same pinned encoder/C500 once. The original manifest/batch CLI still
works. The service accepts only SHA-bound one-window manifests within the
declared input directory, and writes immutable per-window outputs. This is not
a training interface. Readiness means weights loaded, not model accuracy or a
token already available at a native decision.

`tools/v3/native_structure_live_input_node.py` subscribes to raw scans, original
odometry, registered scans and the pinned explicit-source echo. It prefetches
exact rolling five-frame windows through that socket on an independent thread.
Its required CLI bounds and producer-manifest SHA must be bound by the eventual
run specification. Queue overflow and pose/source errors are recorded; there
is no fallback to nearest timestamps, yaw-only motion or another window.
The node publishes no control/advice. A future consumer must match the actual
native five-frame keys and client receipt time before using any response.

The service and writer have synthetic socket/callback tests only. No real-time
deadline, physical pose accuracy, task correspondence or exploration benefit
has been established by these software tests.

## Planned-path evidence option

`prepare_native_overlay.py --planning-paths` adds `native_planning_paths_v1`
feedback after the native lookahead selection and before `PublishWaypoint`.
Default overlay behavior is unchanged. Original global and concatenated paths
retain XYZ, node type and native subspace index; no planner calculations change.
The original by-value `GetNextGlobalSubspace` outputs are neither repaired nor
used as task identity. Planned nodes are not confirmed traversal edges.

Over 16,384 total path nodes, counts are recorded with null paths and an explicit
incomplete flag; no path is truncated or removed from planning. Nonfinite
evidence emits a rejected record while the original waypoint call still runs.
The registry binds only the immediately preceding complete path event with the
same epoch and scan sources. Gaps and rejection never reuse older evidence.
Lookahead and the final extended waypoint are allowed to differ.

Software-only verification (new output directory required):

```sh
python3 integration/native_structure_bridge/verify_native_build.py --planning-paths --output build/native_structure_bridge/<new-name>
```

Completed build: `build/native_structure_bridge/v2_planning_paths/summary.json`.
This option has not yet produced new runtime observations and does not enable
an online model or learned control by itself.

## Local command execution provenance

The sidecar registry now also saves `native_local_execution_v1`, independently
of native regional tasks. An actual WAYPOINT followed immediately by matching
dispatch coordinates/epoch/sources can bind the received native pose, even if
that local waypoint's cell is not a remote VRP candidate. The regional dispatch
verdict remains unchanged. Only the current route epoch may attach its intent;
missing or older route intent leaves the local command explicitly unbound.

Consecutive native odometry sequence/previous-stamp records supply discrete
movement samples, commanded-direction progress and remaining target distance.
Repeated sources do not add movement; a new waypoint or feedback/pose gap ends
the previous interval. These records are NOT controller acknowledgements,
certified physical trajectories, structural direction identities or completion.
No distance threshold, source task or confirmed edge is fabricated. Existing
task completion and route advice rules are unchanged. The sidecar summary
separately counts local commands and recorded motion samples; only synthetic
callback tests have exercised this new consumer so far.
