# Phase 4 C08 Local Implicit Union V3

Status: `IMPLEMENTATION_CONTRACT_DRAFT`  
Date: 2026-08-14

V3 replaces V2's ambiguous “incident tunnel local arc” with the passed
edge-level arc-incidence contract.  For every graph edge endpoint, project the
node to that edge's single physical tunnel spline and retain `(edge_id,
node_id, tunnel_id, arc_m, projection_xyz, away_tangent, connector_error)`.
At a degree>=3 node, the patch contains exactly these records, including two
opposite arc records when one physical tunnel contributes two graph edges.

Each patch selects a fixed forward arc interval from its `arc_m` in the
`away_tangent` direction, clipped to the spline and to the fixed 1.10-radius
3-D node sphere.  The implicit union may include only these records.  A global
analytic tube is queried outside the sphere; it is not a substitute for a
missing arc record inside it.  The patch seam is the sphere boundary; all
arc-record endpoints crossing it must match the analytic field value and
normal within a predeclared numerical tolerance.  Duplicate physical tunnel
IDs are permitted only when their arc direction differs; duplicate
`(edge_id,node_id)` or a missing/reversed tangent is a hard failure.

The V3 synthetic seam contract will use a through-tunnel plus branch geometry:
two opposing records from one tunnel and one branch record must generate one
closed local union surface whose sphere-boundary samples agree with the
analytic field.  This is a tool contract only.  C08 materialization still
requires a new approved Data Card/spec after that synthetic test passes.
