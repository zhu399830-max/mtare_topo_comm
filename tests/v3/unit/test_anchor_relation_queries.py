import numpy as np
import pytest
from mtare_topo.representation.anchor_relation_queries import complete_anchor_pairs

def test_complete_unique_coverage():
    ids=np.array([9,4,8,2]);a=[4,8];p=complete_anchor_pairs(ids,a)
    assert len(p)==2*4-2*3//2
    assert {tuple(sorted(ids[x])) for x in p}=={(4,9),(4,8),(2,4),(8,9),(2,8)}

def test_order_preserves_physical_queries():
    ids=np.array([9,4,8,2]);rev=ids[::-1]
    assert {tuple(sorted(ids[p])) for p in complete_anchor_pairs(ids,[4,8])}=={tuple(sorted(rev[p])) for p in complete_anchor_pairs(rev,[8,4])}

def test_empty_and_invalid():
    assert complete_anchor_pairs(np.array([1,2]),[]).shape==(0,2)
    with pytest.raises(ValueError):complete_anchor_pairs(np.array([1,1]),[1])
    with pytest.raises(ValueError):complete_anchor_pairs(np.array([1,2]),[3])
