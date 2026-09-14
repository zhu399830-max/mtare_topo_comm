"""Analytic counterexample to ID-only negative membership, not ray simulation.

Two representations of the same straight tube have exactly the same union:
[-2, 10] x [-1, 1]^2. An internal degree-two split is not a structural event.
Source non-incidence changes under that split, so cannot alone label a visible
opening as unrelated to the terminal at x=-2. No real targets are produced.
"""
import pytest


def merged_axial_union(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],hi))
        else:
            result.append((lo,hi))
    return result


@pytest.mark.parametrize('split', [-1,0,1,5,9])
def test_nonincident_source_is_not_geometric_nonmembership(split):
    unsplit={'tube':(-2,10)}
    split_tube={'near':(-2,split),'far':(split,10)}
    # Same continuous solid interior and outer boundary, not sampled overlap.
    assert merged_axial_union(unsplit.values())==merged_axial_union(split_tube.values())==[(-2,10)]
    # Same observed cap and window opening; no internal structural node added.
    def id_only_negative(primitives):
        terminal_sources={name for name,(lo,hi) in primitives.items() if lo==-2}
        opening_source=next(name for name,(lo,hi) in primitives.items() if hi==10)
        return opening_source not in terminal_sources
    assert id_only_negative(unsplit) is False
    assert id_only_negative(split_tube) is True


def test_a_real_gap_is_not_erased_by_the_counterexample_union():
    assert merged_axial_union([(-2,0),(1,10)])==[(-2,0),(1,10)]
    # A reference gap alone still says nothing about whether it was observed.
