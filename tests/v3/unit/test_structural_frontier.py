from copy import deepcopy
import pytest

from mtare_topo.planning.structural_frontier import StructuralFrontierPlanner
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig


def graph(position=(3., 0., 4.)):
    return dict(current_node=0, coordinate_frame='odom', edges=[], nodes=[dict(
        id=0, xyz_m=[0, 0, 0], exit_stubs=[dict(state='observed', confidence=.8,
        heading_world_deg=0., opening_geometry=dict(
            schema_version='observed_opening_geometry_v1', coordinate_frame='odom',
            position_kind='observed_opening', position_world_m=position,
            evidence_refs=['scan:4/patch:1', 'scan:4/patch:2']))])])


def planner():
    return StructuralFrontierPlanner(TopologicalPlannerConfig(4., 20., .25, .05))


def test_geometry_changes_actual_waypoint_and_keeps_input():
    g = graph(); before = deepcopy(g)
    output = planner().select_structural_target(g, robot_xyz_m=[0, 0, 0])
    assert output.target.waypoint_xyz_m == pytest.approx((2.4, 0., 3.2))
    assert output.geometry_used and output.local_planner_validation_required
    assert output.evidence_refs == ('scan:4/patch:1', 'scan:4/patch:2')
    assert g == before


def test_no_extrapolation():
    output = planner().select_structural_target(graph((1., 0., 1.)), robot_xyz_m=[0, 0, 0])
    assert output.target.waypoint_xyz_m == (1., 0., 1.)


@pytest.mark.parametrize('field,value', [('coordinate_frame', 'map'),
    ('position_kind', 'primitive_crop_endpoint'), ('evidence_refs', []),
    ('position_world_m', (0., 0., float('nan')))])
def test_invalid_geometry_fails_without_flattening(field, value):
    g = graph(); g['nodes'][0]['exit_stubs'][0]['opening_geometry'][field] = value
    with pytest.raises(ValueError):
        planner().select_structural_target(g, robot_xyz_m=[0, 0, 0])


def test_legacy_explicitly_not_geometry():
    g = graph(); del g['nodes'][0]['exit_stubs'][0]['opening_geometry']
    result = planner().select_structural_target(g, robot_xyz_m=[0, 0, 0])
    assert not result.geometry_used
    assert result.reason == 'legacy_direction_only_baseline'
    assert result.target.waypoint_xyz_m == (4., 0., 0.)


def test_remote_opening_does_not_jump_verified_path():
    g = graph(); g['nodes'][0]['id'] = 1
    g['nodes'][0]['xyz_m'] = [10, 0, 0]
    g['nodes'].append(dict(id=0, xyz_m=[0, 0, 0], exit_stubs=[]))
    g['edges'] = [dict(id=0, **{'from': 0, 'to': 1}, kind='verified_traversed',
        minimum_traversed_length_m=10., traversals=[dict(start_frame=0, end_frame=1)])]
    result = planner().select_structural_target(g, robot_xyz_m=[0, 0, 0])
    assert result.target.mode == 'graph_backtrack'
    assert result.target.waypoint_xyz_m == (4., 0., 0.)
    assert not result.geometry_used
