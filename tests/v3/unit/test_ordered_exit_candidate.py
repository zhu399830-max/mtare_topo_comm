import numpy as np
import pytest
from mtare_topo.teacher.ordered_exit_candidate import ordered_exit_candidate as candidate


@pytest.mark.parametrize('gap',[1e-9,.0001,.006755640983918898,.02])
def test_positive_gap_not_bridged(gap):
    result=candidate([2.,2.+gap,30.],[0,1,1],[1,-1,1],[1,0])
    assert result.status=='candidate' and result.distance_m==2. and result.operands==(0,)


def test_overlap_and_permutation():
    ts=np.array([1.5,2.,30.]);ids=np.array([1,0,1]);dots=np.array([-1,1,1])
    expected=candidate(ts,ids,dots,[1,0])
    assert expected.distance_m==30. and expected.operands==(1,)
    for order in ([2,0,1],[1,2,0]):
        assert candidate(ts[order],ids[order],dots[order],[1,0])==expected


def test_coincident_exits_keep_sources_but_contact_needs_reference():
    assert candidate([2.,2.],[0,1],[1,1],[1,1]).operands==(0,1)
    assert candidate([2.,2.,30.],[0,1,1],[1,-1,1],[1,0]).reason=='coincident_entry_exit'


@pytest.mark.parametrize('ts,ids,dots,reason',[
    ([2.,2.],[0,0],[1,1],'duplicate_or_multishell_crossing'),
    ([2.],[0],[0],'tangent_crossing'),
    ([2.],[0],[-1],'nonalternating_or_missing_crossing'),
    ([2.,3.],[0,0],[1,-1],'unclosed_crossing_stream'),
    ([],[],[],'missing_intersections')])
def test_ambiguous_stream_never_returns_early_candidate(ts,ids,dots,reason):
    result=candidate(ts,ids,dots,[1])
    assert result.status=='needs_reference' and result.reason==reason


def test_range_and_invalid_source():
    assert candidate([30.],[0],[1],[1],maximum_m=10.).status=='out_of_range'
    with pytest.raises(ValueError):candidate([2.],[1],[1],[1])
