from copy import deepcopy
from mtare_topo.governance_synthetic_matrix import SCHEMA,SLUG,POLICY,scope,validate_card
from mtare_topo.governance_surface_selection import digest


def test_closed_synthetic_population_and_standing_scope_binding():
    s=scope();card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=s,scope_sha256=digest(s),policy=POLICY,
        approval=dict(status='APPROVED',scope_sha256=digest(s),authorized_operations=['data_export'],authorized_gates=[3],scope='unit test only'))
    assert validate_card(card).passed
    assert len(s['cases'])==144 and s['control_conditions']==12 and s['rendered_frame_occurrences']==720
    bad=deepcopy(card);bad['scope']['cases'].pop()
    assert not validate_card(bad).passed
    bad=deepcopy(card);bad['approval']['authorized_operations']=['train']
    assert not validate_card(bad).passed
