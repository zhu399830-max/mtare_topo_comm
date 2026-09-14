"""Synthetic contract tests; no map, dataset, model or ROS access."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from mtare_topo.topology.structural_task_lifecycle import (
    AnchorVisit,
    AssociationEvidence,
    DirectionFrameKind,
    EvidenceKind,
    NativeTaskCandidate,
    NativeSnapshotBinding,
    StructuralTaskLifecycle,
    TaskEvidence,
    TaskState,
)


def candidate(task_id="east", **changes):
    return replace(NativeTaskCandidate(
        task_id=task_id, native_candidate_id=f"native-{task_id}", segment="segment-1",
        anchor_id="anchor-A", direction_id=f"direction-{task_id}",
        direction_xyz=(1.0, 0.0, 0.0), target_xyz_m=(10.0, 0.0, 0.0),
        level_id="lower", order=0, source_refs=("synthetic-native-candidate",),
        direction_frame_id="synthetic-common-map", direction_frame_kind=DirectionFrameKind.COMMON_METRIC,
    ), **changes)


def task_evidence(kind, order, task_id="east", **changes):
    kwargs = dict(evidence_id=f"{task_id}-{kind.value}-{order}", task_id=task_id,
                  segment="segment-1", direction_id=f"direction-{task_id}",
                  level_id="lower", order=order, kind=kind,
                  source_refs=("synthetic-observation",), reason="synthetic task feedback")
    if kind is EvidenceKind.EXPLORATION_COMPLETE:
        kwargs.update(trajectory_ref="synthetic-execution-trace", exploration_feedback_ref="synthetic-exploration-result")
    kwargs.update(changes)
    return TaskEvidence(**kwargs)


def association_evidence(**changes):
    return replace(AssociationEvidence(
        evidence_id="verified-pair", source_task_id="east", target_task_id="east-again",
        segment="segment-1", order=1, registration_ref="synthetic-registration",
        direction_ref="synthetic-3d-direction", trajectory_ref="synthetic-trajectory",
        level_ref="synthetic-level", registration_verified=True, direction_verified=True,
        trajectory_verified=True, level_verified=True,
    ), **changes)


def pair_ledger(**second_changes):
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.add_native_candidate(candidate("east-again", **second_changes))
    ledger.propose_association(association_id="pair", source_task_id="east", target_task_id="east-again", order=0, descriptor_ref="synthetic-frozen-token", model_score=1.0)
    return ledger


def resolve_pair(ledger, evidence=None):
    return ledger.resolve_associations(resolution_id="synthetic-batch", source_task_id="east",
        association_ids=("pair",), evidence=(association_evidence() if evidence is None else evidence,),
        order=1, retrieval_set_ref="synthetic-complete-single-candidate-retrieval")


def visit(visit_id="v0", anchor_id="A", order=0, **changes):
    return replace(AnchorVisit(
        visit_id=visit_id, anchor_id=anchor_id, segment="segment-1", trajectory_id="traj-1",
        order=order, position_xyz_m=(float(order), 0.0, 0.0),
        source_refs=(f"synthetic-pose-{order}",), actual_passage_verified=True,
    ), **changes)


@pytest.mark.parametrize("score", [None, -1.0, 0.0, 1.0])
def test_native_candidates_are_never_dropped_by_model_score(score):
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate(model_score=score))
    snapshot = ledger.snapshot()
    assert len(snapshot["tasks"]) == 1
    assert snapshot["tasks"]["east"]["candidate"]["model_score"] == score
    assert snapshot["tasks"]["east"]["state"] is TaskState.PENDING
    assert snapshot["confirmed_edges"] == []


def test_unknown_native_candidate_remains_outstanding_even_after_native_finish():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate(), initial_state=TaskState.UNKNOWN)
    result = ledger.completion_report(native_finish=True, at_home=True)
    assert result["task_counts"][TaskState.UNKNOWN.value] == 1
    assert result["outstanding_tasks"] == 1
    assert result["status"] == "native_finished_with_outstanding_tasks"
    assert not result["recorded_completion_conditions_met"]


def test_arrival_does_not_complete_a_dispatched_task():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 1)) is TaskState.IN_PROGRESS
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.POSITION_REACHED, 2)) is TaskState.IN_PROGRESS
    assert ledger.snapshot()["tasks"]["east"]["completion_evidence_id"] is None


def test_completion_requires_execution_and_exploration_feedback_not_a_score():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="invalid task transition"):
        ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 1))
    assert ledger.snapshot() == before
    with pytest.raises(ValueError, match="feedback and trajectory"):
        task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 1, exploration_feedback_ref=None)
    with pytest.raises(ValueError, match="EvidenceKind"):
        replace(task_evidence(EvidenceKind.POSITION_REACHED, 1), kind="high_model_score")


def test_completion_is_direction_specific_and_can_reopen_without_rewriting_history():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.add_native_candidate(candidate("west", direction_xyz=(-1.0, 0.0, 0.0)))
    ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 1))
    ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 2))
    prefix = ledger.snapshot()["history"]
    assert ledger.snapshot()["tasks"]["west"]["state"] is TaskState.PENDING
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.NEW_OBSERVATION, 3)) is TaskState.PENDING
    snapshot = ledger.snapshot()
    assert snapshot["history"][:len(prefix)] == prefix
    assert snapshot["tasks"]["east"]["completion_evidence_id"] is None
    assert prefix[-1]["state"] == TaskState.COMPLETED.value


@pytest.mark.parametrize("change", [dict(direction_id="direction-west"), dict(segment="segment-2"), dict(level_id="upper")])
def test_completion_evidence_cannot_cross_direction_level_or_segment(change):
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 1))
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="exact segment"):
        ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 2, **change))
    assert ledger.snapshot() == before


def test_blocking_is_retryable_and_never_completion():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 1))
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.BLOCKED, 2)) is TaskState.BLOCKED_RETRY
    with pytest.raises(ValueError, match="invalid task transition"):
        ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 3))
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.RETRY, 3)) is TaskState.PENDING
    assert ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 4)) is TaskState.IN_PROGRESS


def test_score_only_association_is_provisional_and_creates_no_edge():
    ledger = pair_ledger()
    result = ledger.snapshot()
    assert result["associations"]["pair"]["state"] == "proposed"
    assert len(result["tasks"]) == 2
    assert result["confirmed_edges"] == []


@pytest.mark.parametrize("field", ["registration_verified", "direction_verified", "trajectory_verified", "level_verified"])
def test_every_independent_association_check_is_required(field):
    ledger = pair_ledger()
    result = resolve_pair(ledger, association_evidence(**{field: False}))
    assert result["state"] == "unknown"
    assert result["selected_association_id"] is None


@pytest.mark.parametrize("changes", [dict(level_id="upper"), dict(level_id=None), dict(direction_xyz=(-1.0, 0.0, 0.0)), dict(direction_xyz=(0.0, 0.0, 1.0))])
def test_vertical_layers_and_different_orientations_do_not_merge(changes):
    ledger = pair_ledger(**changes)
    result = resolve_pair(ledger)
    assert result["state"] == "unknown"
    assert result["selected_association_id"] is None
    assert len(ledger.snapshot()["tasks"]) == 2


def test_verified_association_is_reversible_and_does_not_share_completion():
    ledger = pair_ledger()
    result = resolve_pair(ledger)
    assert result["state"] == "unique_in_declared_retrieval_set"
    assert not result["global_identity_verified"]
    assert not result["exhaustive_history_search"]
    ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 2))
    ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 3))
    prefix = ledger.snapshot()["history"]
    ledger.revoke_association("pair", evidence_id="new-conflicting-observation", order=4, source_refs=("synthetic-conflict",), reason="new evidence contradicts the provisional identity")
    result = ledger.snapshot()
    assert result["associations"]["pair"]["state"] == "revoked"
    assert result["tasks"]["east-again"]["state"] is TaskState.PENDING
    assert result["history"][:len(prefix)] == prefix
    assert result["confirmed_edges"] == []


def test_pairwise_verification_cannot_accept_before_competitors_are_checked():
    ledger = pair_ledger()
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="resolve_associations"):
        ledger.verify_association("pair", association_evidence())
    assert ledger.snapshot() == before


def competing_ledger():
    ledger = StructuralTaskLifecycle()
    for name in ("east", "east-again", "east-third"):
        ledger.add_native_candidate(candidate(name))
    for name, target, score in (("pair", "east-again", 1.0), ("pair-third", "east-third", -0.3)):
        ledger.propose_association(association_id=name, source_task_id="east", target_task_id=target,
            order=0, descriptor_ref="synthetic-complete-retrieval", model_score=score)
    return ledger


def competing_checks(**changes):
    return (association_evidence(), association_evidence(evidence_id="verified-third",
        target_task_id="east-third", **changes))


def resolve_competitors(ledger, **changes):
    arguments = dict(resolution_id="competition", source_task_id="east",
        association_ids=("pair", "pair-third"), evidence=competing_checks(), order=1,
        retrieval_set_ref="synthetic-complete-two-target-retrieval")
    arguments.update(changes)
    return ledger.resolve_associations(**arguments)


def test_multiple_passing_candidates_all_remain_unknown_regardless_of_score():
    ledger = competing_ledger()
    result = resolve_competitors(ledger)
    assert result["state"] == "unknown"
    assert result["unknown_reason"] == "multiple_supported_candidates"
    assert result["selected_association_id"] is None
    assert result["supported_association_ids"] == ["pair", "pair-third"]
    assert {row["state"] for row in ledger.snapshot()["associations"].values()} == {"unknown"}
    assert len(ledger.snapshot()["tasks"]) == 3
    assert ledger.snapshot()["confirmed_edges"] == []


@pytest.mark.parametrize("changes", [
    dict(association_ids=("pair",), evidence=(association_evidence(),)),
    dict(evidence=(association_evidence(),)),
    dict(evidence=competing_checks(order=2)),
    dict(evidence=competing_checks(source_task_id="east-again")),
])
def test_missing_competitors_missing_checks_future_and_wrong_binding_fail_atomically(changes):
    ledger = competing_ledger()
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        resolve_competitors(ledger, **changes)
    assert ledger.snapshot() == before


def test_unique_result_is_scoped_to_complete_declared_set_and_ignores_scores():
    ledger = competing_ledger()
    checks = (association_evidence(registration_verified=False), competing_checks()[1])
    result = resolve_competitors(ledger, evidence=checks)
    assert result["selected_association_id"] == "pair-third"  # Lower model score, independent checks.
    assert result["state"] == "unique_in_declared_retrieval_set"
    assert result["complete_declared_set_evaluated"]
    assert result["search_scope"] == "declared_retrieval_set_only"
    assert not result["global_identity_verified"]
    assert not result["exhaustive_history_search"]
    before = ledger.snapshot()
    result["candidate_checks"][0]["unknown_reasons"].clear()
    assert ledger.snapshot() == before


@pytest.mark.parametrize("changes,reason", [
    (dict(direction_frame_id="another-map"), "different_direction_frames"),
    (dict(direction_frame_kind=DirectionFrameKind.SENSOR_LOCAL), "direction_not_in_common_metric_frame"),
    (dict(direction_frame_id=None, direction_frame_kind=DirectionFrameKind.UNKNOWN), "direction_not_in_common_metric_frame"),
    (dict(level_id=None), "unknown_or_different_levels"),
])
def test_direction_frame_and_unknown_level_must_not_be_inferred(changes, reason):
    ledger = pair_ledger(**changes)
    result = resolve_pair(ledger)
    assert result["state"] == "unknown"
    assert reason in result["candidate_checks"][0]["unknown_reasons"]
    assert ledger.snapshot()["associations"]["pair"]["verification_evidence_id"] is None


def test_equal_sensor_local_forward_vectors_do_not_become_same_world_direction():
    ledger = StructuralTaskLifecycle()
    for name in ("east", "east-again"):
        ledger.add_native_candidate(candidate(name, direction_frame_id="velodyne",
            direction_frame_kind=DirectionFrameKind.SENSOR_LOCAL))
    ledger.propose_association(association_id="pair", source_task_id="east", target_task_id="east-again",
        order=0, descriptor_ref="synthetic", model_score=1.0)
    result = resolve_pair(ledger)
    assert result["state"] == "unknown"
    assert "direction_not_in_common_metric_frame" in result["candidate_checks"][0]["unknown_reasons"]


def test_cross_segment_proposal_is_rejected_without_dropping_candidates():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.add_native_candidate(candidate("east-again", segment="segment-2"))
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="one segment"):
        ledger.propose_association(association_id="cross", source_task_id="east", target_task_id="east-again", order=0, descriptor_ref="synthetic", model_score=1.0)
    assert ledger.snapshot() == before


def test_edges_only_follow_adjacent_actual_passages_with_continuous_evidence():
    ledger = StructuralTaskLifecycle()
    assert ledger.record_anchor_visit(visit()) is None
    first = ledger.record_anchor_visit(visit("v1", "B", 1, previous_visit_id="v0", continuous_trajectory_ref="synthetic-A-to-B"))
    assert (first["from_anchor"], first["to_anchor"]) == ("A", "B")
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="consecutive"):
        ledger.record_anchor_visit(visit("v2", "C", 2, previous_visit_id="v0", continuous_trajectory_ref="synthetic-forbidden-shortcut"))
    assert ledger.snapshot() == before
    second = ledger.record_anchor_visit(visit("v2", "C", 2, previous_visit_id="v1", continuous_trajectory_ref="synthetic-B-to-C"))
    assert (second["from_anchor"], second["to_anchor"]) == ("B", "C")
    assert len(ledger.snapshot()["confirmed_edges"]) == 2


def test_unverified_arrival_and_missing_trajectory_never_create_edges():
    ledger = StructuralTaskLifecycle()
    with pytest.raises(ValueError, match="actual anchor passage"):
        ledger.record_anchor_visit(visit(actual_passage_verified=False))
    ledger.record_anchor_visit(visit())
    assert ledger.record_anchor_visit(visit("v1", "B", 1)) is None
    assert ledger.snapshot()["confirmed_edges"] == []


@pytest.mark.parametrize("change", [dict(segment="segment-2"), dict(trajectory_id="traj-2")])
def test_no_edge_across_segments_or_trajectories(change):
    ledger = StructuralTaskLifecycle()
    ledger.record_anchor_visit(visit())
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="boundary"):
        ledger.record_anchor_visit(visit("v1", "B", 1, previous_visit_id="v0", continuous_trajectory_ref="synthetic-discontinuous", **change))
    assert ledger.snapshot() == before
    assert ledger.record_anchor_visit(visit("v1", "B", 1, **change)) is None
    assert ledger.snapshot()["confirmed_edges"] == []


def test_returning_to_previous_segment_does_not_bridge_unobserved_interval():
    ledger = StructuralTaskLifecycle()
    ledger.record_anchor_visit(visit())
    ledger.record_anchor_visit(visit("v1", "X", 1, segment="segment-2"))
    with pytest.raises(ValueError, match="consecutive"):
        ledger.record_anchor_visit(visit("v2", "B", 2, previous_visit_id="v0", continuous_trajectory_ref="synthetic-hidden-gap"))


def test_actual_edges_do_not_complete_tasks_and_revisiting_anchor_is_not_self_loop():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.record_anchor_visit(visit())
    assert ledger.record_anchor_visit(visit("v1", "A", 1, previous_visit_id="v0", continuous_trajectory_ref="synthetic-same-anchor")) is None
    ledger.record_anchor_visit(visit("v2", "B", 2, previous_visit_id="v1", continuous_trajectory_ref="synthetic-passage"))
    assert ledger.snapshot()["tasks"]["east"]["state"] is TaskState.PENDING


def completed_ledger():
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(candidate())
    ledger.apply_task_evidence(task_evidence(EvidenceKind.DISPATCH, 1))
    ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 2))
    return ledger


def test_completion_report_separates_native_finish_home_and_recorded_task_completion():
    ledger = completed_ledger()
    for native, home in [(False, True), (None, True), (True, False), (True, None)]:
        report = ledger.completion_report(native_finish=native, at_home=home)
        assert report["all_recorded_tasks_completed"]
        assert not report["recorded_completion_conditions_met"]
    report = ledger.completion_report(native_finish=True, at_home=True)
    assert report["recorded_completion_conditions_met"]
    assert report["outstanding_tasks"] == 0
    assert not report["full_exploration_proven"]


@pytest.mark.parametrize("flags, status", [(dict(budget_exhausted=True), "budget_exhausted_incomplete"), (dict(timed_out=True), "timeout_incomplete"), (dict(budget_exhausted=True, timed_out=True), "timeout_incomplete")])
def test_budget_or_timeout_never_reports_exploration_success(flags, status):
    report = completed_ledger().completion_report(native_finish=True, at_home=True, **flags)
    assert report["status"] == status
    assert report["all_recorded_tasks_completed"]
    assert not report["recorded_completion_conditions_met"]


def test_empty_ledger_is_not_complete_by_vacuous_truth():
    report = StructuralTaskLifecycle().completion_report(native_finish=True, at_home=True)
    assert report["status"] == "no_task_evidence"
    assert not report["all_recorded_tasks_completed"]


def test_invalid_inputs_and_out_of_order_evidence_are_atomic():
    ledger = completed_ledger()
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="earlier causal prefix"):
        ledger.apply_task_evidence(task_evidence(EvidenceKind.NEW_OBSERVATION, 1))
    with pytest.raises(ValueError, match="already consumed"):
        ledger.apply_task_evidence(task_evidence(EvidenceKind.EXPLORATION_COMPLETE, 2))
    with pytest.raises(ValueError, match="already exists"):
        ledger.add_native_candidate(candidate(order=2))
    with pytest.raises(ValueError):
        candidate(model_score=float("nan"))
    with pytest.raises(ValueError):
        candidate(direction_xyz=(0.0, 0.0, 0.0))
    assert ledger.snapshot() == before


def test_snapshots_are_json_serializable_and_cannot_mutate_history():
    ledger = pair_ledger()
    snapshot = ledger.snapshot()
    json.dumps(snapshot, allow_nan=False)
    snapshot["associations"]["pair"]["state"] = "verified"
    snapshot["history"][0]["candidate"]["task_id"] = "corrupted"
    assert ledger.snapshot()["associations"]["pair"]["state"] == "proposed"
    assert ledger.snapshot()["history"][0]["candidate"]["task_id"] == "east"


def test_native_snapshot_binding_is_immutable_and_matches_own_sources_and_causal_order():
    binding = NativeSnapshotBinding("synthetic:current", 1000, ("synthetic-native-candidate",), 0)
    bound = candidate(native_snapshot_binding=binding)
    with pytest.raises(FrozenInstanceError):
        binding.epoch = "synthetic:new"
    with pytest.raises(ValueError, match="own causal order"):
        replace(bound, order=1)
    with pytest.raises(ValueError, match="own source references"):
        replace(bound, source_refs=("another-source",))
    with pytest.raises(ValueError, match="immutable native snapshot"):
        replace(bound, native_snapshot_binding={"epoch": "synthetic:current"})
    with pytest.raises(ValueError, match="immutable"):
        NativeSnapshotBinding("synthetic:current", 1000, ["synthetic-native-candidate"], 0)
    ledger = StructuralTaskLifecycle()
    ledger.add_native_candidate(bound)
    snapshot = ledger.snapshot()
    snapshot["tasks"]["east"]["candidate"]["native_snapshot_binding"]["epoch"] = "changed"
    assert ledger.snapshot()["tasks"]["east"]["candidate"]["native_snapshot_binding"]["epoch"] == "synthetic:current"
