from copy import deepcopy
import numpy as np
import pytest
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_separated_tube_oracle import separated_tube_expectation
from mtare_topo.evaluation.gse_separated_tube_oracle import convex_main_exit


CASES=[c for c in matrix() if c['program']['type'] in ('parallel','stacked')]

@pytest.mark.parametrize('case',CASES,ids=[c['case_id'] for c in CASES])
def test_all_declared_separated_views_have_two_main_openings_no_anchors(case):
    result=separated_tube_expectation(case)
    assert result['anchors']==[] and len(result['openings'])==2
    for opening in result['openings']:
        assert np.isclose(np.linalg.norm(opening['position_m']),10.)
    assert result['other_operand_observable'] is False
    assert result['archived_scan_verified'] is False and result['training_eligible'] is False


@pytest.mark.parametrize('distance',[0.,3.,4.])
def test_touching_and_overlapping_tubes_are_not_certified(distance):
    case=deepcopy(next(c for c in CASES if c['case_id']=='stacked__circle__view2'))
    for point in case['program']['edges'][1]['points']:point[2]=distance
    with pytest.raises(ValueError,match='touch/overlap'):separated_tube_expectation(case)


def test_invalid_sensor_pose_is_not_silently_projected():
    case=deepcopy(CASES[0]);case['poses_world_m'][0][1]=3.
    with pytest.raises(ValueError,match='origin'):separated_tube_expectation(case)


def test_reference_inventory_not_used_as_visible_truth():
    case=deepcopy(CASES[0]);case['reference_inventory']=[{'node':'fake','position_m':[0,0,0]}]
    assert separated_tube_expectation(case)['anchors']==[]


def test_prism_ray_exit_known_axes_and_nonunit_parameter():
    case=next(c for c in CASES if c['case_id']=='stacked__circle__view2')
    rays=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,0,1],[0,0,2]],float)
    np.testing.assert_allclose(convex_main_exit(case,np.zeros_like(rays),rays),[30,30,2,2,1])


def test_prism_ray_origin_outside_rejected():
    with pytest.raises(ValueError,match='inside'):
        convex_main_exit(CASES[0],[[0,9,0]],[[1,0,0]])
