import pytest
from mtare_topo.representation.gse_supervision_claims import Predicate as P,EvidenceClaim as C,SupervisionLedger as L


def test_real_terminal_nonmembership_is_not_absence():
    x=L([C(P.EXISTENCE,'terminal',True,'visible-cap'),C(P.MEMBERSHIP,'opening',False,'separate-channel','terminal')])
    assert x.lookup(P.EXISTENCE,'terminal') is True
    assert x.lookup(P.MEMBERSHIP,'opening','terminal') is False
    assert x.lookup(P.EXISTENCE,'opening') is None


def test_localization_failure_does_not_mean_structure_absence():
    x=L([C(P.LOCALIZATION,'candidate',False,'independent-position-exclusion')])
    assert x.lookup(P.LOCALIZATION,'candidate') is False
    assert x.lookup(P.EXISTENCE,'candidate') is None


def test_hidden_world_information_not_implicitly_available():
    # The ledger accepts observed claims only by contract; these two identical
    # ledgers deliberately omit different hidden structures. This is an API
    # counterexample, NOT a ray visibility test or proof of producer validity.
    a=L([]);b=L([])
    assert a.lookup(P.EXISTENCE,'behind-occluder') is None
    assert b.lookup(P.EXISTENCE,'behind-occluder') is None
    assert not a.qualifies_real_labels


def test_conflicts_fail_and_permutation_preserves_answers():
    a=C(P.EXISTENCE,'x',True,'source-a');b=C(P.EXISTENCE,'x',True,'source-b')
    assert L([a,b]).sources(P.EXISTENCE,'x')==L([b,a]).sources(P.EXISTENCE,'x')
    with pytest.raises(ValueError,match='conflicting'):L([a,C(P.EXISTENCE,'x',False,'source-c')])


def test_no_unbound_or_wrong_arity_claim():
    with pytest.raises(ValueError):C(P.EXISTENCE,'x',True,'')
    with pytest.raises(ValueError):C(P.MEMBERSHIP,'x',True,'source')
    with pytest.raises(ValueError):C(P.EXISTENCE,'x',True,'source','y')
    with pytest.raises(ValueError):C(P.EXISTENCE,'x',1,'source')
