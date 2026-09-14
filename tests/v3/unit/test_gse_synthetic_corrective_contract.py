from copy import deepcopy
from mtare_topo.governance_synthetic_corrective import SCHEMA,SLUG,POLICY,scope,validate_card
from mtare_topo.governance_surface_selection import digest


def test_exact_scope_and_no_training_authority():
    s=scope();assert len(s['input_sha256'])==8 and sum(s['case_invalid_counts'].values())==20
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=s,scope_sha256=digest(s),policy=POLICY,
        approval=dict(status='APPROVED',scope_sha256=digest(s),authorized_operations=['data_export'],authorized_gates=[3],scope='fixed synthetic corrective'))
    assert validate_card(card).passed
    bad=deepcopy(card);bad['scope']['unchanged_valid_positions']=0
    assert not validate_card(bad).passed
    bad=deepcopy(card);bad['approval']['authorized_operations']=['training']
    assert not validate_card(bad).passed
