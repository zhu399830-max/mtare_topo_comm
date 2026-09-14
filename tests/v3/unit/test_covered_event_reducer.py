from fractions import Fraction as F
from itertools import permutations
from mtare_topo.teacher.covered_event_reducer import reduce_covered_events as reduce
from mtare_topo.teacher.exact_event_diagnostic import reduce_exact_events as old


def covered():
    return [(F(1), 1, 'b_entry', -1), (F(2), 1, 'b_exit', 1),
            (F(2), 2, 'c_entry', -1), (F(3), 0, 'a_exit', 1),
            (F(3), 2, 'c_exit', 1)]


def test_persistent_third_operand_resolves_mixed_event():
    result = reduce(covered(), [1, 0, 0])
    assert old(covered(), [1, 0, 0])['reason'] == 'coincident_entry_exit'
    assert result['exact_t'] == 3 and result['operands'] == (0, 2)
    assert result['covered_mixed_events'] == [(F(2), (0,))]


def test_order_duplicates_and_operand_renaming():
    expected = reduce(covered(), [1, 0, 0])
    for events in permutations(covered()):
        assert reduce(list(events) + list(events), [1, 0, 0]) == expected
    mapping = {0: 2, 1: 0, 2: 1}
    result = reduce([(t, mapping[i], p, s) for t, i, p, s in covered()], [0, 0, 1])
    assert result['operands'] == (1, 2)
    assert result['covered_mixed_events'] == [(F(2), (2,))]


def test_touch_without_persistent_cover_still_rejected():
    events = [(F(2), 0, 'a', 1), (F(2), 1, 'b', -1), (F(3), 1, 'c', 1)]
    assert reduce(events, [1, 0])['reason'] == 'coincident_entry_exit'


def test_cover_that_exits_at_same_root_is_not_persistent():
    events = covered()[:-2] + [(F(2), 0, 'a_exit', 1), (F(3), 2, 'c_exit', 1)]
    assert reduce(events, [1, 0, 0])['reason'] == 'coincident_entry_exit'


def test_sub_float_positive_gap_stops_at_near_exit():
    a = F(2); b = a + F(1, 10**30)
    assert float(a) == float(b)
    events = [(a, 0, 'a', 1), (b, 1, 'b', -1), (F(3), 1, 'c', 1)]
    result = reduce(events, [1, 0])
    assert result['exact_t'] == a and result['operands'] == (0,)


def test_full_stream_validation_is_not_bypassed():
    assert reduce(covered()[:-2] + [(F(3), 2, 'c_exit', 1)], [1, 0, 0])['reason'] == 'unclosed_stream'
    assert reduce(covered() + [(F(4), 0, 'bad', 1)], [1, 0, 0])['reason'] == 'nonalternating_stream'


def test_same_operand_multiplane_is_still_unknown():
    assert reduce(covered() + [(F(2), 1, 'another_plane', 1)], [1, 0, 0])['reason'] == 'multiple_planes_same_operand_root'
