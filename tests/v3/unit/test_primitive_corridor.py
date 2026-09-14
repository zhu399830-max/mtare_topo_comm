import numpy as np
import pytest
from mtare_topo.planning.primitive_corridor import corridor_continuations


def run(axis,robot=(0,0,0),lookahead=4):
    return corridor_continuations(axis,robot,lookahead_m=lookahead,
                                  primitive_index=0,source_refs=['scan4/primitive0'])


def test_single_corridor_has_two_proposals_not_two_openings():
    result=run([[-8,0,0],[0,0,0],[8,0,0]])
    assert [p['axis_target_xyz_m'] for p in result['proposals']]==[[-4.,0.,0.],[4.,0.,0.]]
    assert result['opening_count'] is None
    assert all(not p['physical_opening'] and not p['creates_edge'] for p in result['proposals'])


def test_height_and_bend_affect_proposals():
    result=run([[-8,0,0],[0,0,0],[4,0,3]],lookahead=2.5)
    assert result['proposals'][-1]['axis_target_xyz_m']==pytest.approx([2.,0.,1.5])


def test_reversal_does_not_change_targets():
    axis=[[-8,1,0],[0,0,1],[8,0,2]]
    assert run(axis)==run(axis[::-1])


def test_short_support_is_not_extrapolated_or_terminal():
    result=run([[-1,0,0],[0,0,0],[1,0,0]])
    assert all(p['reaches_observation_boundary'] and p['axis_travel_m']==1 for p in result['proposals'])
    assert result['opening_count'] is None


def test_ambiguous_attachment_rejected():
    result=run([[-1,0,0],[0,2,0],[1,0,0]],robot=(0,0,0))
    assert result['status']=='AMBIGUOUS_AXIS_ATTACHMENT'


@pytest.mark.parametrize('axis', [[[0,0,0],[0,0,0],[1,0,0]],
    [[0,0,0],[1,0,0],[float('nan'),0,0]]])
def test_bad_geometry_rejected(axis):
    with pytest.raises(ValueError):run(axis)
