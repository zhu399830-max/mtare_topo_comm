"""Production-shaped synthetic JSON replay; no map, ROS or real payload reads."""
from copy import deepcopy
import json

import pytest

from mtare_topo.integration.native_region_task_registry import (
    NativeRegionTaskRegistry, RegionExecutionState, replay_native_region_records,
)


def native_snapshot(epoch="synthetic:1", candidate_ids=(1, 2), stamp=1000):
    positions = {str(cid): [float(cid * 10), 0.0, 0.0] for cid in (0, *candidate_ids)}
    route = [0, *candidate_ids, 0]
    return dict(schema_version="native_route_snapshot_v1", epoch=epoch, stamp_ns=stamp,
        max_age_ns=500, max_extra_cost_units=3, source_frame_keys=[f"registered_scan:{stamp}"],
        candidate_ids=list(candidate_ids), original_route=route, original_cost_units=10 * (len(route) - 1),
        native_edges=[[a, b, 10] for a in (0, *candidate_ids) for b in (0, *candidate_ids)],
        candidate_positions_m=positions, candidate_records=[dict(id=cid, position_xyz_m=positions[str(cid)],
            status=1, status_name="EXPLORING") for cid in candidate_ids],
        robot_position_xyz_m=[0.0, 0.0, 0.0], coordinate_frame="map")


def event(kind, index, candidate_id=1, **changes):
    stamp = 1000 + index * 10
    value = dict(schema_version="native_route_feedback_v1", session_id="synthetic", event_index=index,
        epoch="synthetic:1", stamp_ns=stamp, source_frame_keys=["registered_scan:1000"],
        evidence_id=f"synthetic:event:{index}", event=kind, candidate_id=candidate_id,
        position_xyz_m=[float(candidate_id * 10), 0.0, 0.0])
    if kind == "NATIVE_REGION_DISPATCH":
        value.update(waypoint_xyz_m=[candidate_id * 10 - 0.5, 0.0, 0.0],
            robot_position_xyz_m=[0.0, 0.0, 0.0], status=1,
            dispatch_scope="published_native_waypoint_region_not_route_first_cell")
    if kind in ("NATIVE_REGION_VISIT", "NATIVE_POSE"):
        value.update(robot_position_xyz_m=[candidate_id * 10 - 0.5, 0.0, 0.0],
            robot_orientation_xyzw=[0.0, 0.0, 0.0, 1.0], pose_stamp_ns=stamp, pose_sequence=index,
            pose_frame_id="map", pose_child_frame_id="sensor",
            pose_source_key=f"state_estimation:{stamp}", pose_binding_exact=True,
            previous_region_id=0, pose_evidence_kind="native_state_estimation_cell_membership",
            continuous_trajectory_verified=False, dispatch_epoch="synthetic:1",
            dispatch_evidence_id="synthetic:event:1")
    if kind == "NATIVE_REGION_STATUS":
        value.update(status=2, status_name="COVERED", previous_status=1, actor_robot_id=0,
            native_rule="local_residual_coverage_below_threshold", native_coverage_rule_satisfied=True,
            physical_exploration_verified=False, direction_complete=False,
            dispatch_epoch="synthetic:1", dispatch_evidence_id="synthetic:event:1")
    value.update(changes)
    return value


def registry():
    result = NativeRegionTaskRegistry(segment="segment-1", trajectory_id="trajectory-1")
    assert result.consume("snapshot", native_snapshot(), order=0, record_ref="synthetic:snapshot:1")["accepted"]
    return result


def feed(registry, payload, order=None):
    return registry.consume("feedback", json.dumps(payload), order=payload["event_index"] if order is None else order,
        record_ref=f"synthetic:feedback:{payload['event_index']}")


def task(registry, candidate_id=1):
    return next(row for row in registry.snapshot()["tasks"] if row["candidate_id"] == candidate_id)


def test_native_cells_are_preserved_as_regions_without_fabricated_direction_tasks():
    result = registry().snapshot()
    assert {row["candidate_id"] for row in result["tasks"]} == {1, 2}
    assert {row["task_kind"] for row in result["tasks"]} == {"native_grid_region_not_direction"}
    assert all(row["direction_scope"] == "unknown_unenumerated" for row in result["tasks"])
    assert result["direction_task_count"] == 0 and result["regions_with_unknown_direction_scope"] == 2
    assert result["confirmed_traversals"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("covered_first", [False, True])
def test_explicit_dispatch_visit_local_coverage_records_only_regional_execution(covered_first):
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    kinds = ["NATIVE_REGION_STATUS", "NATIVE_REGION_VISIT"] if covered_first else ["NATIVE_REGION_VISIT", "NATIVE_REGION_STATUS"]
    first = feed(ledger, event(kinds[0], 2))
    assert not first["native_region_executed"]
    assert task(ledger)["execution_state"] is RegionExecutionState.DISPATCHED
    final = feed(ledger, event(kinds[1], 3))
    assert final["native_region_executed"]
    current = task(ledger)
    assert current["execution_state"] is RegionExecutionState.EXECUTED
    assert current["executed_attempt_ids"] == ("synthetic:event:1",)
    assert not current["all_region_directions_completed"]
    assert current["direction_task_ids"] == []
    assert not ledger.snapshot()["full_exploration_proven"]


@pytest.mark.parametrize("changes", [
    dict(native_coverage_rule_satisfied=False, native_rule="no_candidate_viewpoints"),
    dict(native_coverage_rule_satisfied=False, native_rule="lost_communications"),
    dict(previous_status=2), dict(actor_robot_id=1),
])
def test_heuristic_or_nonlocal_covered_status_is_not_completion(changes):
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    feed(ledger, event("NATIVE_REGION_VISIT", 2))
    result = feed(ledger, event("NATIVE_REGION_STATUS", 3, **changes))
    assert result["reason"] == "NATIVE_STATUS_NOT_QUALIFIED_LOCAL_COVERAGE_TRANSITION"
    assert task(ledger)["execution_state"] is RegionExecutionState.DISPATCHED
    assert task(ledger)["executed_attempt_ids"] == ()


def test_visit_and_candidate_disappearance_never_imply_completed_exploration():
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    feed(ledger, event("NATIVE_REGION_VISIT", 2))
    ledger.consume("snapshot", native_snapshot("synthetic:2", (2,), 1050), order=3, record_ref="synthetic:snapshot:2")
    assert task(ledger)["execution_state"] is RegionExecutionState.DISPATCHED
    assert len(ledger.snapshot()["tasks"]) == 2
    assert task(ledger)["executed_attempt_ids"] == ()


@pytest.mark.parametrize("changes,reason", [
    (dict(epoch="synthetic:old"), "DISPATCH_NOT_BOUND_TO_CURRENT_SNAPSHOT"),
    (dict(source_frame_keys=["velodyne_points:1000"]), "DISPATCH_SOURCE_BINDING_MISMATCH"),
    (dict(candidate_id=99, position_xyz_m=[990.0, 0.0, 0.0]), "WAYPOINT_REGION_NOT_A_NATIVE_CANDIDATE"),
])
def test_unbound_waypoint_dispatch_does_not_create_or_mutate_tasks(changes, reason):
    ledger = registry()
    before = ledger.snapshot()["tasks"]
    result = feed(ledger, event("NATIVE_REGION_DISPATCH", 1, **changes))
    assert not result["accepted"] and result["reason"] == reason
    assert ledger.snapshot()["tasks"] == before
    assert ledger.snapshot()["attempts"] == {}


@pytest.mark.parametrize("changes", [dict(dispatch_epoch="synthetic:wrong"), dict(dispatch_evidence_id="unknown"), dict(candidate_id=2, position_xyz_m=[20.0, 0.0, 0.0])])
def test_completion_must_bind_exact_dispatched_region_and_epoch(changes):
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    feed(ledger, event("NATIVE_REGION_VISIT", 2))
    result = feed(ledger, event("NATIVE_REGION_STATUS", 3, **changes))
    assert not result["accepted"]
    assert task(ledger)["execution_state"] is RegionExecutionState.DISPATCHED


def test_later_feedback_can_bind_an_older_explicit_dispatch_not_current_epoch():
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    ledger.consume("snapshot", native_snapshot("synthetic:2", (2,), 1015), order=2, record_ref="synthetic:snapshot:2")
    feed(ledger, event("NATIVE_REGION_VISIT", 2, epoch="synthetic:2"), order=3)
    result = feed(ledger, event("NATIVE_REGION_STATUS", 3, epoch="synthetic:2"), order=4)
    assert result["native_region_executed"]
    assert task(ledger)["execution_state"] is RegionExecutionState.EXECUTED


def test_missing_dispatch_still_records_actual_pose_without_fabricating_a_task():
    ledger = registry()
    result = feed(ledger, event("NATIVE_REGION_VISIT", 1, candidate_id=99))
    assert result["accepted"]
    assert result["region_execution_rejection_reason"] == "NO_MATCHING_EXPLICIT_DISPATCH"
    assert len(ledger.snapshot()["tasks"]) == 2
    assert len(ledger.snapshot()["anchors"]) == 1


def test_actual_pose_anchors_are_not_cell_centers_and_no_uncertified_edges_are_created():
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_VISIT", 1))
    feed(ledger, event("NATIVE_REGION_VISIT", 2, candidate_id=2))
    result = ledger.snapshot()
    assert len(result["anchors"]) == 2
    assert result["anchors"][0]["position_xyz_m"] == (9.5, 0.0, 0.0)
    assert result["confirmed_traversals"] == []
    assert result["traversal_trace"][0]["reason"] == "CONTINUOUS_TRAJECTORY_NOT_VERIFIED"
    assert not result["traversal_trace"][0]["confirmed"]


@pytest.mark.parametrize("changes", [dict(pose_binding_exact=False), dict(pose_stamp_ns=999999),
    dict(continuous_trajectory_verified=True), dict(pose_frame_id="vehicle")])
def test_unbound_future_or_uncertified_pose_claims_are_rejected(changes):
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    result = feed(ledger, event("NATIVE_REGION_VISIT", 2, **changes))
    assert not result["accepted"]
    assert ledger.snapshot()["anchors"] == []
    assert not ledger.snapshot()["attempts"]["synthetic:event:1"]["visit_evidence_id"]


def test_region_reopening_preserves_prior_execution_and_unknown_directions():
    ledger = registry()
    feed(ledger, event("NATIVE_REGION_DISPATCH", 1))
    feed(ledger, event("NATIVE_REGION_VISIT", 2))
    feed(ledger, event("NATIVE_REGION_STATUS", 3))
    before = deepcopy(ledger.snapshot()["decision_trace"])
    feed(ledger, event("NATIVE_REGION_STATUS", 4, previous_status=2, status=1, status_name="EXPLORING",
        native_rule="native_reopening", native_coverage_rule_satisfied=False))
    assert task(ledger)["execution_state"] is RegionExecutionState.PENDING
    assert task(ledger)["executed_attempt_ids"] == ("synthetic:event:1",)
    assert task(ledger)["direction_scope"] == "unknown_unenumerated"
    assert ledger.snapshot()["decision_trace"][:len(before)] == before


def test_native_finish_does_not_complete_regions_or_unknown_directions():
    ledger = registry()
    feed(ledger, event("NATIVE_EXECUTION_STATE", 1, native_finish=True, at_home=True))
    result = ledger.snapshot()
    assert result["native_finish"] and result["at_home"]
    assert all(row["execution_state"] is RegionExecutionState.PENDING for row in result["tasks"])
    assert not result["full_exploration_proven"] and result["regions_with_unknown_direction_scope"] == 2


def test_rewritten_pose_source_duplicate_feedback_and_session_change_are_rejected():
    ledger = registry()
    feed(ledger, event("NATIVE_POSE", 1))
    result = feed(ledger, event("NATIVE_REGION_VISIT", 2, pose_stamp_ns=1010,
        pose_source_key="state_estimation:1010", robot_position_xyz_m=[8.0, 0.0, 0.0]))
    assert result["reason"] == "NATIVE_POSE_SOURCE_REWRITTEN"
    result = feed(ledger, event("NATIVE_POSE", 1), order=3)
    assert result["reason"] == "NONCAUSAL_OR_DUPLICATE_FEEDBACK"
    result = feed(ledger, event("NATIVE_POSE", 4, session_id="other", evidence_id="other:event:4"), order=4)
    assert result["reason"] == "SESSION_CHANGED_NEW_REGISTRY_REQUIRED"


def test_native_pose_then_region_callback_can_bind_the_same_actual_pose_once():
    ledger = registry()
    pose = event("NATIVE_POSE", 1)
    assert feed(ledger, pose)["accepted"]
    visit = event("NATIVE_REGION_VISIT", 2, pose_stamp_ns=pose["pose_stamp_ns"],
        pose_sequence=pose["pose_sequence"], pose_source_key=pose["pose_source_key"])
    assert feed(ledger, visit)["accepted"]
    visit.update(event_index=3, evidence_id="synthetic:event:3", stamp_ns=1030)
    assert feed(ledger, visit)["accepted"]
    assert len(ledger.snapshot()["anchors"]) == 1
    assert len(ledger.snapshot()["trajectory"]) == 3


def test_startup_pose_without_candidate_snapshot_does_not_create_region_tasks():
    ledger = NativeRegionTaskRegistry(segment="segment-1", trajectory_id="trajectory-1")
    result = feed(ledger, event("NATIVE_POSE", 1, epoch="", source_frame_keys=[]))
    assert result["accepted"]
    assert ledger.snapshot()["tasks"] == []
    assert ledger.consume("snapshot", native_snapshot(), order=2, record_ref="synthetic:later-snapshot")["accepted"]
    assert len(ledger.snapshot()["tasks"]) == 2


def test_snapshot_binding_and_json_replay_are_real_callable_and_immutable():
    messages = [("snapshot", native_snapshot()), ("feedback", event("NATIVE_REGION_DISPATCH", 1)),
                ("feedback", event("NATIVE_REGION_VISIT", 2)), ("feedback", event("NATIVE_REGION_STATUS", 3))]
    records = [json.dumps(dict(kind=kind, payload=payload, order=i, record_ref=f"synthetic:record:{i}"))
               for i, (kind, payload) in enumerate(messages)]
    result = replay_native_region_records(records, segment="segment-1", trajectory_id="trajectory-1")
    assert result["tasks"][0]["execution_state"] is RegionExecutionState.EXECUTED
    assert result["tasks"][0]["observations"][0]["binding"]["epoch"] == "synthetic:1"
    assert all(row["accepted"] for row in result["decision_trace"])
    assert len(result["feedback_records"]) == 3
    ledger = registry()
    view = ledger.snapshot()
    view["tasks"][0]["candidate_id"] = 999
    assert task(ledger)["candidate_id"] == 1
    bad = native_snapshot("synthetic:2")
    bad["candidate_records"][0]["id"] = 99
    before = ledger.snapshot()["tasks"]
    assert not ledger.consume("snapshot", bad, order=1, record_ref="synthetic:bad")["accepted"]
    assert ledger.snapshot()["tasks"] == before
