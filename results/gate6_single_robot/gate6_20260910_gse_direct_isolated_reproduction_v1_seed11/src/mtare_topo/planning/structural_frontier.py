"""Three-dimensional opening handoff for the existing graph selector.

This module does not detect openings or certify free space. A producer must
provide observed opening positions in the graph frame and source references.
Geometry changes the waypoint, not the verified-edge routing or utility rule.
Legacy direction-only stubs remain explicitly identifiable baseline outputs.
"""
from dataclasses import dataclass, replace
import math

from .topological_frontier import RuleBasedTopologicalFrontierPlanner, _finite_xyz


@dataclass(frozen=True)
class StructuralTargetDecision:
    target: object
    geometry_used: bool
    evidence_refs: tuple[str, ...]
    reason: str
    local_planner_validation_required: bool = True


class StructuralFrontierPlanner(RuleBasedTopologicalFrontierPlanner):
    """Reuse selection, but steer toward an explicit 3-D opening when present.

    Callers attach ``opening_geometry`` to a stub only after structure tracking.
    Unknown width, support and reachability are not synthesized here. Remote
    openings are still reached via verified graph hops, never a straight jump.
    """

    def select_structural_target(self, snapshot, *, robot_xyz_m,
                                 retry_counts=None, blocked_frontiers=None):
        # Validate *all* geometry records before selection, not only the winner.
        geometry = {}
        for node in snapshot.get('nodes', []):
            for index, stub in enumerate(node.get('exit_stubs', [])):
                record = stub.get('opening_geometry')
                if record is None:
                    continue
                if not isinstance(record, dict) or record.get('schema_version') != 'observed_opening_geometry_v1':
                    raise ValueError('versioned observed opening geometry required')
                if record.get('coordinate_frame') != snapshot.get('coordinate_frame') or not record.get('coordinate_frame'):
                    raise ValueError('opening and graph coordinate frames must match')
                if record.get('position_kind') != 'observed_opening':
                    raise ValueError('primitive crop endpoints are not observed openings')
                refs = record.get('evidence_refs')
                if not isinstance(refs, (list, tuple)) or not refs or any(not isinstance(r, str) or not r for r in refs):
                    raise ValueError('nonempty observation evidence references required')
                position = _finite_xyz(record.get('position_world_m', ()), name='opening position')
                geometry[(int(node['id']), index)] = (position, tuple(refs))
        target = super().select_target(snapshot, robot_xyz_m=robot_xyz_m,
            retry_counts=retry_counts, blocked_frontiers=blocked_frontiers)
        if target.frontier is None:
            return StructuralTargetDecision(target, False, (), 'no_selected_opening')
        key = (target.frontier.node_id, target.frontier.stub_index)
        if key not in geometry:
            return StructuralTargetDecision(target, False, (), 'legacy_direction_only_baseline')
        position, refs = geometry[key]
        if target.mode != 'frontier_exit':
            return StructuralTargetDecision(target, False, refs, 'verified_graph_hop_before_opening')
        robot = _finite_xyz(robot_xyz_m, name='robot position')
        delta = tuple(p-r for p, r in zip(position, robot))
        distance = math.hypot(*delta)
        # Do not extrapolate beyond the observed opening or flatten its height.
        scale = min(1., self.config.waypoint_lookahead_m / distance) if distance else 0.
        waypoint = tuple(r + scale*d for r, d in zip(robot, delta))
        target = replace(target, waypoint_xyz_m=waypoint,
                         reason='observed_3d_opening_requires_local_planner_validation')
        return StructuralTargetDecision(target, True, refs, 'observed_3d_opening')
