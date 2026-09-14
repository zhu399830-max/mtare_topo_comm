from copy import deepcopy
from dataclasses import replace
import pytest
from mtare_topo.integration.structural_route_policy import suggest_from_verified_task_history
from mtare_topo.topology.structural_task_lifecycle import (
    StructuralTaskLifecycle, NativeTaskCandidate, DirectionFrameKind,
    TaskEvidence, EvidenceKind, AssociationEvidence, NativeSnapshotBinding,
)


def setup(candidate_overrides=None):
    s = dict(schema_version='native_route_snapshot_v1', epoch='synthetic:0', stamp_ns=1000,
             max_age_ns=500, max_extra_cost_units=3, source_frame_keys=['registered_scan:1000'],
             candidate_ids=[1, 2], original_route=[0, 1, 2, 0], original_cost_units=30,
             native_edges=[[a, b, 10] for a in range(3) for b in range(3) if a != b])
    l = StructuralTaskLifecycle()
    for name in ('current1', 'current2', 'past'):
        historical = name == 'past'
        sources = ('registered_scan:500',) if historical else tuple(s['source_frame_keys'])
        binding = NativeSnapshotBinding('synthetic:previous' if historical else s['epoch'],
            500 if historical else s['stamp_ns'], sources, 0)
        candidate = NativeTaskCandidate(name, '1' if name == 'current1' else '2' if name == 'current2' else 'past-native-id',
            's', 'a', 'east', (1., 0., 0.), (10., 0., 0.), 'observed_floor', 0, sources,
            direction_frame_kind=DirectionFrameKind.COMMON_METRIC, direction_frame_id='map',
            native_snapshot_binding=binding)
        l.add_native_candidate(replace(candidate, **(candidate_overrides or {}).get(name, {})))
    l.apply_task_evidence(TaskEvidence('dispatch', 'past', 's', 'east', 'observed_floor', 1,
                                      EvidenceKind.DISPATCH, ('source',), 'synthetic'))
    l.apply_task_evidence(TaskEvidence('done', 'past', 's', 'east', 'observed_floor', 2,
        EvidenceKind.EXPLORATION_COMPLETE, ('source',), 'synthetic', 'trajectory', 'execution'))
    return s, l


def propose(l):
    l.propose_association(association_id='match', source_task_id='current1', target_task_id='past',
                         order=3, descriptor_ref='synthetic_retrieval', model_score=1.)


def verify(l):
    ev = AssociationEvidence('checks', 'current1', 'past', 's', 4,
        'registration', 'direction', 'trajectory', 'floor', True, True, True, True)
    l.resolve_associations(resolution_id='resolution', source_task_id='current1',
        association_ids=('match',), evidence=(ev,), order=4, retrieval_set_ref='synthetic_retrieval')


def run(s, l):
    return suggest_from_verified_task_history(s, l, {1: 'current1', 2: 'current2'},
        source_frame_keys=s['source_frame_keys'], now_ns=1100)


def test_proposal_score_alone_cannot_change_route_but_verified_execution_can():
    s, l = setup(); propose(l)
    assert run(s, l)['decision']['route'] == s['original_route']
    verify(l); before = l.snapshot()
    r = run(s, l)
    assert r['decision']['changed'] and r['decision']['route'] == [0, 2, 1, 0]
    assert l.snapshot() == before and not r['task_state_changed'] and not r['candidate_removed']
    assert not r['actual_waypoint_effect_verified']


def test_revocation_and_reopening_restore_original_order():
    s, l = setup(); propose(l); verify(l)
    l.revoke_association('match', evidence_id='revocation', order=5, source_refs=('new',), reason='new contradiction')
    assert run(s, l)['decision']['route'] == s['original_route']
    s, l = setup(); propose(l); verify(l)
    l.apply_task_evidence(TaskEvidence('new', 'past', 's', 'east', 'observed_floor', 5,
                                     EvidenceKind.NEW_OBSERVATION, ('new',), 'reopened'))
    assert run(s, l)['decision']['route'] == s['original_route']


def test_unreachable_reordering_and_freshness_preserve_native_route():
    s, l = setup(); propose(l); verify(l)
    s['native_edges'] = [e for e in s['native_edges'] if e[:2] != [0, 2]]
    r = run(s, l)
    assert r['decision']['route'] == s['original_route'] and not r['decision']['accepted']
    with pytest.raises(ValueError, match='snapshot'):
        suggest_from_verified_task_history(s, l, {1: None, 2: None}, source_frame_keys=['other'], now_ns=1100)


@pytest.mark.parametrize('overrides,reason', [
    ({'native_candidate_id': '99'}, 'NATIVE_TASK_CANDIDATE_ID_MISMATCH'),
    ({'native_candidate_id': '01'}, 'NATIVE_TASK_CANDIDATE_ID_MISMATCH'),
    ({'native_snapshot_binding': None}, 'NATIVE_TASK_SNAPSHOT_UNBOUND'),
    ({'native_snapshot_binding': NativeSnapshotBinding('synthetic:old', 1000,
        ('registered_scan:1000',), 0)}, 'NATIVE_TASK_EPOCH_MISMATCH'),
    ({'native_snapshot_binding': NativeSnapshotBinding('synthetic:0', 500,
        ('registered_scan:1000',), 0)}, 'NATIVE_TASK_STAMP_MISMATCH'),
    ({'source_refs': ('registered_scan:500',), 'native_snapshot_binding': NativeSnapshotBinding(
        'synthetic:0', 1000, ('registered_scan:500',), 0)}, 'NATIVE_TASK_SOURCE_MISMATCH'),
    ({'source_refs': ('velodyne_points:1000',), 'native_snapshot_binding': NativeSnapshotBinding(
        'synthetic:0', 1000, ('velodyne_points:1000',), 0)}, 'NATIVE_TASK_SOURCE_MISMATCH'),
])
def test_current_task_own_binding_is_required_even_when_caller_supplies_fresh_sources(overrides, reason):
    s, l = setup({'current1': overrides})
    propose(l); verify(l)
    snapshot_before, ledger_before = deepcopy(s), l.snapshot()
    result = run(s, l)
    assert result['advice'] is None
    assert not result['decision']['accepted'] and not result['decision']['changed']
    assert result['decision']['route'] == s['original_route']
    assert result['decision']['reason'] == reason
    assert not result['current_task_bindings_verified']
    assert result['rejected_candidate_id'] == 1
    assert result['deferred_candidate_ids'] == []
    assert l.snapshot() == ledger_before and s == snapshot_before


def test_wrong_current_candidate_task_mapping_cannot_use_another_verified_direction():
    s, l = setup(); propose(l); verify(l)
    result = suggest_from_verified_task_history(s, l, {1: 'current2', 2: 'current1'},
        source_frame_keys=s['source_frame_keys'], now_ns=1100)
    assert result['decision']['reason'] == 'NATIVE_TASK_CANDIDATE_ID_MISMATCH'
    assert result['decision']['route'] == s['original_route']
    assert not result['decision']['accepted']


def test_current_epoch_is_not_required_for_historical_completed_target():
    s, l = setup(); propose(l); verify(l)
    target_binding = l.snapshot()['tasks']['past']['candidate']['native_snapshot_binding']
    assert target_binding['epoch'] != s['epoch']
    assert target_binding['source_frame_keys'] != tuple(s['source_frame_keys'])
    result = run(s, l)
    assert result['current_task_bindings_verified']
    assert result['decision']['route'] == [0, 2, 1, 0]
    assert result['decision']['changed']


def test_unbound_historical_target_does_not_disable_valid_current_bindings():
    s, l = setup({'past': {'native_snapshot_binding': None}}); propose(l); verify(l)
    assert run(s, l)['decision']['route'] == [0, 2, 1, 0]


def test_explicit_unknown_current_bindings_preserve_original_without_borrowing_old_tasks():
    s, l = setup(); propose(l); verify(l)
    result = suggest_from_verified_task_history(s, l, {1: None, 2: None},
        source_frame_keys=s['source_frame_keys'], now_ns=1100)
    assert result['decision']['route'] == s['original_route']
    assert result['deferred_candidate_ids'] == []
