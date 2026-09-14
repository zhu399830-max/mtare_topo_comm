from copy import deepcopy
from mtare_topo import governance_native_graph_v2 as g
from mtare_topo.governance_surface_selection import digest


def test_exact_card_and_drift(monkeypatch):
    scope={'synthetic_scope':307}
    monkeypatch.setattr(g,'compile_scope',lambda root:scope)
    h=digest(scope)
    card=dict(schema_version=g.SCHEMA,card_id=g.SLUG,operation='data_export',
        scope=scope,scope_sha256=h,approval=dict(status='APPROVED',scope_sha256=h,
        authorized_operations=['data_export'],authorized_gates=[3],confirmation_reference='test only'))
    assert g.validate_card(card).passed
    for key,value in [('operation','training'),('scope',{}),('scope_sha256','wrong')]:
        bad=deepcopy(card);bad[key]=value
        assert not g.validate_card(bad).passed
    bad=deepcopy(card);bad['approval']['authorized_gates']=[4]
    assert not g.validate_card(bad).passed

