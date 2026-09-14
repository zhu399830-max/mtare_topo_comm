from types import SimpleNamespace
import pytest
from mtare_topo.evaluation import gse_reference_chain_audit as audit
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def fixture(monkeypatch,shift=0.):
    definitions=[('t','T','J',(-1.,0,0),(0.,0,0)),('a','J','D',(0.,0,0),(1.,0,0)),
                 ('b','D','Z',(1.+shift,0,0),(2.,0,0)),('other','J','W',(0.,0,0),(0.,1.,0))]
    primitives=[];groups={}
    for sid,n1,n2,p1,p2 in definitions:
        primitives.append(SimpleNamespace(primitive_id=sid,centerline_xyz_m=[p1,p2]))
        for side,node in enumerate((n1,n2)):
            groups.setdefault(node,dict(node_id_teacher_only=node,paths=[]))['paths'].append({'endpoint_key':[sid,side]})
    monkeypatch.setattr(audit,'load_p1a_realized_construction',lambda d:(None,primitives))
    monkeypatch.setattr(audit,'construction_incident_paths',lambda d:list(groups.values()))
    document={'synthetic':True}
    target=dict(source_binding={'construction_sha256':canonical_sha(document)},teacher_provenance=dict(
        terminal_anchor_start=1,terminals=[dict(endpoint_key_teacher_only=['t',0],node_id_teacher_only='T')],
        openings=[{'primitive_id_teacher_only':'b'}],terminal_nonmembership=[dict(anchor_index=1,opening_index=0,
            terminal_continuation_sources=['t'],opening_continuation_sources=['a','b'],junction_node_teacher_only='J')]))
    return target,document


def test_complete_split_chain_is_recovered_without_producer(monkeypatch):
    result=audit.audit_reference_chains(*fixture(monkeypatch))
    assert result[0]['opening_sources']==2 and result[0]['terminal_boundaries']==['J','T']
    assert result[0]['reference_only_not_physical_separation']


def test_nonzero_gap_not_snapped(monkeypatch):
    with pytest.raises(ValueError,match='unresolved'):
        audit.audit_reference_chains(*fixture(monkeypatch,shift=1e-14))


def test_incomplete_saved_chain_fails(monkeypatch):
    target,doc=fixture(monkeypatch)
    target['teacher_provenance']['terminal_nonmembership'][0]['opening_continuation_sources']=['b']
    with pytest.raises(ValueError,match='contradicts'):audit.audit_reference_chains(target,doc)
