import numpy as np
import pytest
from mtare_topo.data.gse_surface_affinity_targets import bind_pure_patch_targets as bind


def neighbors(n):
    a=np.full((n,8),-1,dtype=np.int64)
    for i in range(n):a[i,0]=(i+1)%n
    return a


def test_same_and_different_pure_sources_only():
    t=bind([['a'],['a'],['b']],np.ones(3,dtype=bool),[[0],[1],[2]],neighbors(3))
    assert t.values[:,0].tolist()==[1,0,0] and t.known[:,0].all()
    assert not t.structural_membership and not t.values.flags.writeable


def test_mixed_patch_never_majority_assigned():
    t=bind([['a'],['a'],['b'],['a']],np.ones(4,dtype=bool),[[0,1,2],[3]],neighbors(2))
    assert t.patch_reason[0]=='MIXED_SOURCES' and not t.known.any()


def test_ambiguous_return_not_arbitrarily_selected():
    t=bind([['a','b'],['a']],np.ones(2,dtype=bool),[[0],[1]],neighbors(2))
    assert not t.known.any()


def test_source_code_without_qualified_surface_evidence_is_unknown():
    t=bind([['a'],['a']],np.asarray([False,True]),[[0],[1]],neighbors(2))
    assert t.patch_reason[0]=='UNQUALIFIED_RETURN' and not t.known.any()


def test_source_name_permutation_cannot_change_labels():
    a=bind([['a'],['b'],['a']],np.ones(3,dtype=bool),[[0],[1],[2]],neighbors(3))
    b=bind([['z'],['q'],['z']],np.ones(3,dtype=bool),[[0],[1],[2]],neighbors(3))
    assert np.array_equal(a.values,b.values) and np.array_equal(a.known,b.known)


def test_invalid_point_identity_rejected():
    with pytest.raises(ValueError,match='indices'):
        bind([['a']],np.ones(1,dtype=bool),[[1]],neighbors(1))
