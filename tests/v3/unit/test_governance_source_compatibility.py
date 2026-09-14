from copy import deepcopy
from mtare_topo.governance_source_compatibility import SCHEMA,SLUG,scope,digest,validate_card


def card():
    s=scope()
    return dict(schema_version=SCHEMA,card_id=SLUG,operation='audit',scope=s,scope_sha256=digest(s),
        approval=dict(status='APPROVED',scope_sha256=digest(s),authorized_operations=['audit'],authorized_gates=[3],
        approved_by='unit test',approved_at='test',scope='test fixture only',confirmation_reference='unit test'))


def test_exact_metadata_scope():
    assert validate_card(card()).passed
    assert len(scope()['input_sha256'])==39


def test_rehashed_expansion_or_training_rejected():
    for key,value in [('rays',691201),('real_worlds_read',['C10']),('training_eligible',True)]:
        c=deepcopy(card());c['scope'][key]=value
        c['scope_sha256']=c['approval']['scope_sha256']=digest(c['scope'])
        assert not validate_card(c).passed
