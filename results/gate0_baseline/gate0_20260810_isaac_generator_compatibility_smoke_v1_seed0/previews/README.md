# Preview provenance

Every preview must record split, world, trajectory/sample ID, method, units, and the hypothesis it supports or contradicts.

## `tunnel_mesh_g001_navigation_actual_surface_projections.png` — current navigation geometry

- Split: none; Gate-0 single-world geometry development only.
- World/parent: `tng_84998d00587e03dc`; geometry `g001_navigation_grade`.
- Source: all 159,784 vertices from the accepted OBJ with the corrected TNG overlaid.
- Panels: top X–Y, elevation X–Z, oblique 3-D; units are metres.
- Method: cubic-Hermite horizontal centerlines, arc-length elevation, global-Z floor profile and enlarged branch chambers; 0.4 m voxel isosurface.
- Accepted evidence: maximum pitch 0.180 rad, minimum turn radius 2.837 m, 43,953 robot probes with zero failures, one watertight component and genus 3.
- Limitation: overview projections cannot prove wheel-floor contact or controller execution. The image is an offline actual-OBJ projection, not an Isaac RTX screenshot.

## `tunnel_mesh_g000_actual_surface_projections.png` — current complete mesh result

- Split: none; Gate-0 single-world smoke only.
- World/parent: `tng_3eca286d5e4c4cd3`; geometry `g000`.
- Trajectory/sample: none.
- Source: all 111,864 surface vertices read from the accepted `collision_render_mesh.obj`, with the parent TNG overlaid.
- Units: metres.
- Panels from left to right: top X–Y, elevation X–Z, oblique 3-D projection.
- Colours: brown is the actual mesh surface projection; blue is RGTG; red is CTG; white points are graph nodes.
- Hypothesis checked: the complete extracted tunnel surface follows the complete topology parent without an obvious missing branch in the three projections.
- Limitation: this is a deterministic offline projection, not an Isaac RTX screenshot. The USD import passed, but headless Replicator PNG capture failed at renderer frame scheduling.

## `tng_mesh_aware_top_and_elevation.svg` / `.png` — accepted mesh parent

- Split: none; Gate-0 single-world smoke only.
- World/parent: `tng_3eca286d5e4c4cd3`; geometry `g000`.
- Trajectory/sample: none.
- Method: mesh-aware RGTG/CTG topology with a 6.5 m generation guard for an approximately 5 m-wide tunnel.
- Units: metres.
- Hypothesis checked: a 24-node, three-cycle topology can preserve sufficient nonincident clearance for tunnel meshing.
- Linked mesh evidence: 111,864 vertices, 223,736 triangles, watertight edge incidence, one component, genus 3.
- Limitation: this preview shows complete centerlines, not the rendered tunnel surface. See the actual OBJ projection above.

## `tng_core_smoke_v2_top_and_elevation.svg` / `.png` — current result

- Split: none; this is a Gate-0 generator smoke test, not a dataset member.
- World/parent: `tng_762e384fdc6b6fa0`.
- Trajectory/sample: none.
- Method: deterministic clean-room TNG core; RGTG spanning tree plus CTG connectors; sampled edge–edge and node–edge clearance guards.
- Units: metres.
- Views: orthographic X–Y and X–Z projections of the complete topology parent.
- Hypothesis checked: the frozen seed can produce one connected, branching, three-dimensional graph with exactly three independent cycles.
- Limitation: graph lines are tunnel centreline candidates, not collision-certified tunnel surfaces.

## `tng_core_smoke_top_and_elevation.svg` / `.png` — rejected v1

- World/parent: `tng_faf57ba981e3c324`.
- Rejection reason: post-generation audit found a 1.089 m sampled node–nonincident-edge distance below the 3.0 m node-clearance contract. Its earlier 1.526 m edge–edge result was insufficient to establish graph-stage clearance.
- Retention reason: preserved as negative evidence; it must not be used as the accepted generator smoke result.
