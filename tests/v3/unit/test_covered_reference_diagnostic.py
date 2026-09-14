from types import SimpleNamespace as N
import pytest
from mtare_topo.teacher.covered_event_diagnostic import CoveredEventDiagnostic
from mtare_topo.teacher.covered_reference_diagnostic import CoveredReferenceDiagnostic


def make(monkeypatch, result, checked='origin_checked', hit=None):
    monkeypatch.setattr(CoveredEventDiagnostic, 'query', lambda *a, **k: result)
    d = CoveredReferenceDiagnostic.__new__(CoveredReferenceDiagnostic)
    d._origin = N(check=lambda p: N(status=checked, inside=(True,)))
    d.compare = lambda *a, **k: N(reference=hit)
    return d


def forbidden(*a, **k):
    raise AssertionError('fallback must not run')


@pytest.mark.parametrize('status', ['candidate', 'out_of_range'])
def test_accepted_path_unchanged(monkeypatch, status):
    result = dict(status=status, covered_mixed_events=[{'persistent_operands': [1]}])
    d = make(monkeypatch, result); d.compare = forbidden
    assert d.query([0, 0, 0], [1, 0, 0]) is result


def test_contact_never_promoted_by_reference(monkeypatch):
    d = make(monkeypatch, dict(status='needs_reference', reason='coincident_entry_exit'))
    d.compare = forbidden; d._origin.check = forbidden
    r = d.query([0, 0, 0], [1, 0, 0])
    assert r['status'] == 'needs_reference'
    assert r['reference_status'] == 'blocked_uncovered_contact'
    assert not r['reference_attempted']


def test_invalid_origin_not_promoted(monkeypatch):
    original = dict(status='needs_reference', reason='origin:surface')
    d = make(monkeypatch, original, checked='needs_reference'); d.compare = forbidden
    assert d.query([0, 0, 0], [1, 0, 0]) is original


def test_reference_preserves_evidence_and_sources(monkeypatch):
    original = dict(status='needs_reference', reason='unclosed_stream', records=[{'native_t': 2.}])
    d = make(monkeypatch, original, hit=N(distance_m=2., source_primitive_ids=('a', 'b')))
    r = d.query([0, 0, 0], [1, 0, 0])
    assert r['status'] == 'reference_return' and r['distance_m'] == 2.
    assert r['sources'] == ['a', 'b'] and r['records'] == original['records']
    assert r['reason'] == 'unclosed_stream'


def test_reference_none_stays_unknown(monkeypatch):
    d = make(monkeypatch, dict(status='needs_reference', reason='no_exact_attribution'))
    r = d.query([0, 0, 0], [1, 0, 0])
    assert r['status'] == 'needs_reference' and 'distance_m' not in r
    assert r['reference_status'] == 'unknown'
