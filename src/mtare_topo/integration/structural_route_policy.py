"""Reversible task-history ordering, followed by the native route validator.

Only complete, uniquely resolved candidate sets may suggest deferring a task
matching an already executed direction. No candidate is deleted or completed.
Registration producers remain responsible for evidence correctness; this is
not learned utility prediction, a planner replacement, or an identity oracle.
"""
from mtare_topo.integration.native_route_advice import make_route_advice, validate_route_advice, validate_snapshot
from mtare_topo.topology.structural_task_lifecycle import StructuralTaskLifecycle, TaskState


def _reject_current_binding(snapshot, *, now_ns, reason, candidate_id):
    decision = validate_route_advice(snapshot, None, now_ns=now_ns)
    decision['reason'] = reason
    return dict(advice=None, decision=decision, deferred_candidate_ids=[],
                task_state_changed=False, candidate_removed=False,
                learned_probability_used=False, actual_waypoint_effect_verified=False,
                current_task_bindings_verified=False, rejected_candidate_id=candidate_id)


def suggest_from_verified_task_history(snapshot, ledger, candidate_tasks, *, source_frame_keys, now_ns):
    validate_snapshot(snapshot)
    if not isinstance(ledger, StructuralTaskLifecycle):
        raise ValueError('typed structural task lifecycle required')
    if list(source_frame_keys) != snapshot['source_frame_keys']:
        raise ValueError('candidate/task binding must refer to this exact native scan snapshot')
    if (not isinstance(candidate_tasks, dict) or set(candidate_tasks) != set(snapshot['candidate_ids'])
            or any(type(k) is not int for k in candidate_tasks)):
        raise ValueError('every native candidate requires a task or explicit unknown binding')
    state = ledger.snapshot()
    if any(t is not None and (not isinstance(t, str) or t not in state['tasks']) for t in candidate_tasks.values()):
        raise ValueError('unknown task binding')
    # Caller-supplied frame keys do not establish that a task belongs to this
    # native snapshot. Check each task's immutable ingest provenance itself.
    # Historical association targets deliberately need not have this epoch.
    for cid, task_id in candidate_tasks.items():
        if task_id is None:
            continue
        candidate = state['tasks'][task_id]['candidate']
        binding = candidate['native_snapshot_binding']
        if candidate['native_candidate_id'] != str(cid):
            reason = 'NATIVE_TASK_CANDIDATE_ID_MISMATCH'
        elif binding is None:
            reason = 'NATIVE_TASK_SNAPSHOT_UNBOUND'
        elif binding['epoch'] != snapshot['epoch']:
            reason = 'NATIVE_TASK_EPOCH_MISMATCH'
        elif binding['snapshot_stamp_ns'] != snapshot['stamp_ns']:
            reason = 'NATIVE_TASK_STAMP_MISMATCH'
        elif (tuple(candidate['source_refs']) != tuple(snapshot['source_frame_keys'])
              or tuple(binding['source_frame_keys']) != tuple(candidate['source_refs'])):
            reason = 'NATIVE_TASK_SOURCE_MISMATCH'
        elif binding['task_order'] != candidate['order']:
            reason = 'NATIVE_TASK_ORDER_MISMATCH'
        else:
            continue
        return _reject_current_binding(snapshot, now_ns=now_ns, reason=reason, candidate_id=cid)
    deferred, refs = set(), []
    for cid, task_id in candidate_tasks.items():
        if task_id is None:
            continue
        # Do not combine old resolutions: latest source resolution is the only
        # current candidate-set decision; later ambiguity supersedes old wins.
        resolutions = [r for r in state['association_resolutions'].values() if r['source_task_id'] == task_id]
        if not resolutions:
            continue
        resolution = resolutions[-1]
        selected = resolution['selected_association_id']
        if selected is None:
            continue
        association = state['associations'][selected]
        if association['state'] != 'unique_in_declared_retrieval_set':
            continue
        past = state['tasks'][association['target_task_id']]
        if past['state'] != TaskState.COMPLETED or not past['completion_evidence_id']:
            continue
        # Same task is not marked done: only its native route position changes.
        deferred.add(cid)
        refs.extend((resolution['resolution_id'], resolution['retrieval_set_ref'],
                     association['verification_evidence_id'], past['completion_evidence_id']))
    original = snapshot['original_route']
    route = [original[0], *sorted(original[1:-1], key=lambda cid: cid in deferred), original[-1]]
    refs = list(dict.fromkeys(ref for ref in refs if ref))
    try:
        advice = make_route_advice(snapshot, route, evidence_refs=refs)
    except ValueError:
        # Reordering cannot invent a finite edge absent from the native matrix.
        advice = None
    decision = validate_route_advice(snapshot, advice, now_ns=now_ns)
    if advice is None:
        decision['reason'] = 'TASK_ORDER_HAS_NO_NATIVE_PATH'
    return dict(advice=advice, decision=decision, deferred_candidate_ids=sorted(deferred),
                task_state_changed=False, candidate_removed=False,
                learned_probability_used=False, actual_waypoint_effect_verified=False,
                current_task_bindings_verified=True, rejected_candidate_id=None)
