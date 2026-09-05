# C08 V7R2 Contract Review

Date: 2026-08-15

## Outcome

The sealed V7R2 run is numerically reproducible under its implemented algorithm, but it does not satisfy the run spec statement `Optimize only window frames` or the original V3 trajectory-adjustment boundary. It therefore cannot establish C08 replay eligibility.

The run directory and its `RUN_STATE=COMPLETED` seal remain immutable historical evidence. Project-level interpretation is `FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL`.

## Exact evidence

Authoritative run:

`results/gate4_topology/gate4_20260815_cano_c08_support_interval_margin_qualification_v7r2_seed0`

The frozen spec says that only window frames are optimized. The executor instead calls `solve_support_sensor(...)` over the complete trajectory before applying the window membership mask. Comparing each sealed input trajectory with the corrected trajectory and joining it to `artifacts/complete_frame_audit.csv` gives:

| World | Frames | Window frames | Outside frames | Outside frames with nonzero z change | Outside z change |
|---|---:|---:|---:|---:|---:|
| S01 | 801 | 80 | 721 | 721 | -4.599496 to -4.153639 m |
| S06 | 1,414 | 128 | 1,286 | 1,286 | -4.716372 to -3.722062 m |
| S10 | 2,558 | 258 | 2,300 | 2,300 | -3.987098 to -3.060038 m |
| Total | 4,773 | 466 | 4,307 | 4,307 | -4.716372 to -3.060038 m |

These counts supersede the earlier preliminary total of 4,343, which was based on stale remembered window counts. The CSV join above is authoritative.

The displacement is structural, not round-off. The isotropic swept-tube field places downward support near `centerline_z - tunnel_radius`, whereas the sealed Cano perception pose is defined from `centerline_z + fta_distance_m + 1m`. Tunnel radii are approximately 5 m while the three world-level `fta_distance_m` values are approximately -1.15, -1.23 and -1.98 m.

## Perception incompatibility

Using the corrected V7R2 sensor poses to raycast the frozen native Cano S10 perception mesh at frames `0,1,2,3,4,5,6,10,100,1000,2000,2557` produced objective branch LOS failure at all 12 frames. Multiple frames had no horizontal native-mesh hit and returned the 50 m maximum range. This is consistent with the corrected sensors being several metres below the native perception surface.

Finite range arrays do not make these poses semantically eligible. Removing LOS, rendering at the old pose while building the graph at the corrected pose, or treating 50 m no-hit scans as valid would change the causal data contract and is forbidden without a new approved method.

## Impact

- The 32-iteration bracketed ray-exit implementation remains validated.
- The continuous layered-field seam/isolation and patch evidence remain useful component evidence.
- The claim that the V7R2 corrected trajectories qualify the original 4,773-frame C08 causal replay is invalid.
- Frozen M1D inference and causal graph replay must not start from these corrected poses.
- No C09, C10 or M-TARE data was read during this review.

## Corrective options

### A — Split collision boundary from route-conditioned support (recommended)

Keep the typed arc incidence, 25 windows, continuous layered side/ceiling collision field and bracketed ray exits. Replace the isotropic-tube downward floor with a separate route-conditioned support surface anchored to each sealed spline at `centerline_z + fta_distance_m`. Outside every window, the original sensor pose is immutable. Only window frames may receive the declared joint XY/z correction. At junctions, incident support arcs use the same fixed window and deterministic transition/isolation rules.

This preserves native-mesh LiDAR poses and the frozen M1D sensor domain while correcting the support representation that caused the global relocation. It requires new unit/synthetic tests, a complete read-only three-world feasibility proof, a new Data Card/spec and one new formal qualification run. The previous roughly four-hour qualification cost should be expected again.

### B — Use the implicit field for LiDAR as well

This creates one coordinate-consistent geometry, but changes the perception domain from rough native Cano surfaces to smooth analytic tubes. Frozen M1D was not selected for that domain, so a C08-only domain-compatibility audit is required and failure cannot be repaired by tuning. It also supersedes the approved assumption that native Cano mesh remains the perception asset.

### C — Stop the C08 geometry route

Retain V7R2 as component/numerical evidence and make no causal replay claim.

Option A is recommended because it restores the original window-local adjustment contract and keeps collision/support qualification compatible with the already-frozen perception and model pipeline.
