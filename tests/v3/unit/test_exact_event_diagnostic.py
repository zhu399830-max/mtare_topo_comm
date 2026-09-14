from fractions import Fraction as F
from mtare_topo.teacher.exact_event_diagnostic import reduce_exact_events


def test_sub_float_gap_remains_a_gap():
    a=F(2);b=a+F(1,10**30)
    assert float(a)==float(b)
    result=reduce_exact_events([(a,0,'a',1),(b,1,'b',-1),(F(3),1,'c',1)],[1,0])
    assert result['exact_t']==a and result['operands']==(0,)


def test_sub_float_overlapping_exits_preserve_last_source():
    a=F(2);b=a+F(1,10**30)
    assert reduce_exact_events([(a,0,'a',1),(b,1,'b',1)],[1,1])['operands']==(1,)


def test_exact_duplicates_and_order_do_not_change_result():
    events=[(F(1),1,'a',-1),(F(2),0,'b',1),(F(3),1,'c',1)]
    assert reduce_exact_events(events,[1,0])==reduce_exact_events(events[::-1]+events,[1,0])


def test_contact_and_unclosed_are_not_candidates():
    assert reduce_exact_events([(F(2),0,'a',1),(F(2),1,'b',-1),(F(3),1,'c',1)],[1,0])['status']=='needs_reference'
    assert reduce_exact_events([(F(2),0,'a',1),(F(3),0,'b',-1)],[1])['status']=='needs_reference'
