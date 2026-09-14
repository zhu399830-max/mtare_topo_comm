# Staged execution log

## Preflight

- Project filesystem free space: approximately 641 GB.
- Docker GPU probe passed with RTX 5090 D, driver 580.173.02, 32607 MiB.
- NVIDIA Container Toolkit CLI: 1.17.8.
- Existing containers and images were not stopped, modified, or deleted.

## Generator acquisition

The paper-linked anonymous clone failed with exit code 128 and a GitHub username prompt. Public web search confirmed that the paper cites the same URL, but did not locate a credible public fork or archived repository. No partial checkout remained. Commit, LICENSE, dependencies, and entry points therefore remain unverifiable and generator reuse is prohibited.

## Isaac image

The official `nvcr.io/nvidia/isaac-sim:6.0.1` image pulled successfully. Reported repository digest:

`sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`

## Compatibility Checker

The first official headless launch request and a second reduced `--network=none` request were both rejected before process creation because the automatic approval-review stream disconnected. This is not an Isaac failure and is not a compatibility pass. No further workaround was attempted.

## Stop decision

One-world import, LiDAR smoke, dataset generation, and training remain blocked until the Compatibility Checker actually passes and a legal/executable generator path is selected.

The stop above records the first attempt and was later superseded by explicit user authorization to continue this same Gate-0 run.

## Resumed Compatibility Checker

The official Isaac Sim 6.0.1 checker was run headlessly with networking disabled and exited 0 with `System checking result: PASSED`. GPU/driver, 34.19 GB VRAM, 67.17 GB RAM, Ubuntu 24.04.3, CPU core count, and storage met the checker requirements. Recorded nonblocking warnings were CPU powersave governor, no display in headless mode, and offline OmniHub retries.

## Clean-room TNG core

Because the paper-linked generator remained unavailable and legally unverifiable, a standard-library-only independent topology core was implemented. Its scope is limited to seed separation, 3-D RGTG/CTG graph generation, graph validation, sampled clearance guards, canonical replay hash, topology statistics, and parent-level split leakage validation.

All 37 V3 unit tests passed. The first generated graph `tng_faf57ba981e3c324` was subsequently rejected when an added audit measured 1.089 m node-to-nonincident-edge distance against a 3.0 m requirement. The defect was fixed without weakening thresholds.

The corrected graph `tng_762e384fdc6b6fa0` contains 24 nodes and 26 edges in one component, with 23 RGTG edges, 3 CTG edges, cycle rank 3, 10.452 m vertical span, 3.451 m sampled nonincident-edge clearance, and 3.524 m sampled node-edge clearance. Complete X–Y and X–Z previews were retained with provenance.

## Current stop boundary

No tunnel mesh, Isaac world import, LiDAR observation, label, dataset, or model was created. The run is at `RUNNING_TNG_CORE_REVIEW`; the next stage requires user review of the corrected complete graph and a declared mesh-generation/acceptance method.

## Single-world TNG-to-mesh continuation

The user authorized one paper-inspired TNG-to-mesh-to-Isaac smoke. The prior abstract parent `tng_762e384fdc6b6fa0` was not used because its 3.451 m nonincident centerline distance cannot safely contain an approximately 5 m-wide tunnel without false geometric connections. Thresholds were not weakened.

A new parent `tng_3eca286d5e4c4cd3` was generated with a 6.5 m graph-stage clearance contract. It contains 24 nodes, 26 edges, cycle rank 3, and 6.894 m sampled nonincident edge/node-edge clearance.

The clean-room geometry method unions clipped elliptical tunnel free-space samples and extracts one surface using a fixed marching-tetrahedra decomposition. A direct block-boundary prototype failed the new 2-manifold test and was replaced without removing that test.

Geometry `g000` passed offline checks with mesh hash `274fe9e504ef5de8b9cea842c26bcaf343196cbd8d5c217973850a55cba547ed`: 111,864 vertices, 223,736 triangles, zero degenerate faces, every mesh edge incident to exactly two triangles, one mesh component, Euler characteristic -4, and genus 3 matching the graph cycle rank. A full regeneration reproduced the same mesh hash.

## Isaac USD import attempts

Attempt v1 executed in the fixed Isaac Sim 6.0.1 container and failed correctly. Isaac reported a USD parse error in `faceVertexCounts`: the exporter omitted commas between line chunks of large arrays. This is an exporter defect, not a mesh defect. The failing USD and machine result are retained.

The serializer was fixed, a regression test was added, and all 40 tests passed. `isaac_stage_v2.usda` retains the same mesh hash and has SHA-256 `b46c7c7fb81b7ea4d4b0e4fa4bb0f3c2caea3e0c8e9333b4c389c7a5859ea9c5`.

The v2 Isaac retry did not execute: the automatic approval-review stream disconnected and rejected process creation. This is neither an Isaac failure nor a pass. Safety constraints prohibit indirect workarounds; explicit user reauthorization is required before retrying the same fixed container command.

The paragraph above records the initial retry request and is superseded by the reauthorized execution below.

## Reauthorized Isaac USD v2 import

After explicit user reauthorization, the unchanged `isaac_stage_v2.usda` was opened in the official `nvcr.io/nvidia/isaac-sim:6.0.1` container. The validator completed 10 application updates and returned `PASS`: `/World/TunnelMesh` contained 111,864 vertices, 223,736 triangles and 671,208 face indices; extent was `[[-51.5, -106.5, -7.5], [29.5, 14.5, 3.5]]`; stage metadata was Z-up and one metre per unit; `PhysicsCollisionAPI` and `MeshCollisionAPI` were present with approximation `none`. The container exited 0.

## Preview attempts and retained visualization

Two Isaac headless Replicator attempts did not produce the requested PNG. The reduced second attempt used one camera and an explicit disk backend, but the renderer failed to advance to its scheduled frame and the writer did not drain. Only that temporary preview container was stopped; three pre-existing containers were not changed. This failure does not invalidate the independently completed USD import check, but no image is labelled as an Isaac RTX render.

For review, `previews/tunnel_mesh_g000_actual_surface_projections.png` was generated directly from all vertices in the accepted OBJ. It shows complete X–Y, X–Z and oblique projections with the parent TNG overlaid. Its offline provenance is recorded beside the image.

No LiDAR observation, label, dataset, model, planner rollout, or second geometry was created. The run is stopped at user review.

## g001 navigation-grade continuation

The user rejected immediate batch generation because g000 was too far from a navigation world. Work therefore remained single-world. Curved-path audit exposed a contract defect in the topology generator: RGTG edges respected `max_abs_pitch_rad`, but CTG connector candidates did not. The predecessor parent is retained only as g000 import-smoke evidence.

CTG pitch filtering was added and a single replacement parent `tng_84998d00587e03dc` was generated. It retains 24 nodes, 26 edges, cycle rank 3 and 6.894 m graph-stage clearance; maximum edge pitch is 0.180190890 rad.

g001 uses cubic-Hermite horizontal centerlines, elevation parameterized by cumulative horizontal arc length, a global-Z floor profile, 1.15x branch chambers and a 0.4 m voxel isosurface. Four attempts were rejected without weakening thresholds: two for excessive path pitch, one for a 2.395625 m turn radius below 2.5 m, and one for mesh genus 4 instead of 3.

The accepted geometry has mesh hash `1e89fd07b834b1c494d4a320c4cc9d81c67af9a6a597462d3b40d8afcaf0656c`, 159,784 vertices, 319,576 triangles, one watertight component and genus 3. Maximum path pitch is 0.180190773 rad, minimum turn radius is 2.837491 m, curved nonincident clearance is 6.856996 m against a 6.405 m requirement, and all 43,953 robot body probes pass. A full replay reproduced mesh, OBJ and USD hashes.

The official Isaac Sim 6.0.1 container imported the collision mesh and read 26 navigation curves containing 1,127 points. It completed 10 updates and exited 0. No dynamic robot, physics contact rollout, LiDAR, dataset, label or training was run. The next stop boundary is dynamic single-robot floor/contact validation on this one geometry.
