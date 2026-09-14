from copy import deepcopy
import pytest
from mtare_topo.governance_covered_sensor_export import SCHEMA, SLUG, scope, digest, validate_card


def card():
    s = scope()
    return dict(schema_version=SCHEMA, card_id=SLUG, operation='data_export', scope=s,
        scope_sha256=digest(s), approval=dict(status='APPROVED',authorized_operations=['data_export'],
        authorized_gates=[3],scope_sha256=digest(s),approved_by='test only',approved_at='test',
        scope='test fixture not actual permission',confirmation_reference='unit test'))


def test_exact_card():
    assert validate_card(card()).passed


@pytest.mark.parametrize('key,value', [('frames',61),('rays',1),('training_eligible',True),
    ('teacher_labels_generated',1),('real_worlds_read',['C10'])])
def test_modified_scope_rejected_even_with_rehashed_approval(key,value):
    c = deepcopy(card()); c['scope'][key] = value
    c['scope_sha256'] = c['approval']['scope_sha256'] = digest(c['scope'])
    assert not validate_card(c).passed


def test_approval_cannot_authorize_training():
    c = card(); c['approval']['authorized_operations'] = ['training']
    assert not validate_card(c).passed
