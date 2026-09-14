from types import SimpleNamespace as N
from mtare_topo.teacher.exact_event_diagnostic import ExactEventDiagnostic
from mtare_topo.teacher.exact_reference_diagnostic import ExactReferenceDiagnostic


def make(monkeypatch,result,checked='origin_checked',hit=None):
    monkeypatch.setattr(ExactEventDiagnostic,'query',lambda *a,**k:result)
    diagnostic=ExactReferenceDiagnostic.__new__(ExactReferenceDiagnostic)
    diagnostic._origin=N(check=lambda p:N(status=checked,inside=(True,)))
    diagnostic.compare=lambda *a,**k:N(reference=hit)
    return diagnostic


def test_candidate_unchanged(monkeypatch):
    result={'status':'candidate','distance_m':2.}
    d=make(monkeypatch,result)
    d.compare=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected fallback'))
    assert d.query([0,0,0],[1,0,0]) is result


def test_unknown_never_filled(monkeypatch):
    d=make(monkeypatch,{'status':'needs_reference','reason':'unclosed_stream'})
    result=d.query([0,0,0],[1,0,0])
    assert result['status']=='needs_reference' and result['reference_status']=='unknown'


def test_reference_retains_original_reason_and_sources(monkeypatch):
    d=make(monkeypatch,{'status':'needs_reference','reason':'multiple_planes_same_operand_root'},
           hit=N(distance_m=2.,source_primitive_ids=('a','b')))
    result=d.query([0,0,0],[1,0,0])
    assert result['status']=='reference_return' and result['sources']==['a','b'] and result['reason']=='multiple_planes_same_operand_root'


def test_invalid_origin_cannot_bypass_precondition(monkeypatch):
    result={'status':'needs_reference','reason':'origin:surface'}
    d=make(monkeypatch,result,checked='needs_reference')
    d.compare=lambda *a,**k:(_ for _ in ()).throw(AssertionError('origin bypass'))
    assert d.query([0,0,0],[1,0,0]) is result
