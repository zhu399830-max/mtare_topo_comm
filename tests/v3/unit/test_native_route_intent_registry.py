from copy import deepcopy
import pytest
from test_native_region_task_registry import registry,native_snapshot,event,feed


def decision():
    s=native_snapshot()
    return dict(schema_version='native_route_decision_v1',epoch=s['epoch'],mode='shadow',changed=False,
        original_route=s['original_route'],route=s['original_route'],source_frame_keys=s['source_frame_keys'],
        native_candidates_preserved=True,native_edges_changed=False)


def test_final_native_route_links_local_waypoint_but_never_completes_task():
    r=registry()
    d=r.consume('decision',decision(),order=1,record_ref='decision:1')
    assert d['accepted'] and not d['task_state_changed']
    result=feed(r,event('WAYPOINT',2,xyz_m=[0.5,0.,0.]))
    assert result['route_intent_bound']
    state=r.snapshot()
    assert len(state['route_intents'])==1
    assert state['waypoint_route_links'][0]['route_intent_record_ref']=='decision:1'
    assert not state['waypoint_route_links'][0]['task_completed']
    assert not state['attempts'] and not state['confirmed_traversals']


def test_late_decision_does_not_rewrite_earlier_waypoint_or_prefix():
    r=registry();feed(r,event('WAYPOINT',1,xyz_m=[0.,0.,0.]))
    before=deepcopy(r.snapshot()['waypoint_route_links'])
    r.consume('decision',decision(),order=2,record_ref='decision:1')
    assert r.snapshot()['waypoint_route_links']==before
    assert before[0]['route_intent_record_ref'] is None


@pytest.mark.parametrize('changes',[dict(source_frame_keys=['wrong']),dict(epoch='synthetic:2'),
    dict(route=[0,1,0]),dict(route=[0,2,1,0],changed=True),dict(native_edges_changed=True)])
def test_invalid_decision_does_not_change_state(changes):
    r=registry();d=decision();d.update(changes)
    result=r.consume('decision',d,order=1,record_ref='bad')
    assert not result['accepted'] and r.snapshot()['route_intents']=={}
