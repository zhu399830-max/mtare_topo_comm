"""Replay actual native candidate/feedback JSON without inventing directions.

A M-TARE grid cell is a regional bookkeeping unit, not a channel direction.
Native dispatch + the explicit local coverage transition + subsequently
observed membership record a *native regional execution*. They never certify
all directions in that cell, physical exploration completion, or graph edges.
The real producer currently reports no continuous-trajectory certificate, so
actual pose anchors are retained while confirmed traversals remain empty.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import math

from mtare_topo.integration.native_route_advice import validate_snapshot, route_cost
from mtare_topo.integration.native_local_execution import NativeLocalExecution
from mtare_topo.topology.structural_task_lifecycle import (
    AnchorVisit, NativeSnapshotBinding, StructuralTaskLifecycle,
)


NATIVE_STATUS = {0: "UNSEEN", 1: "EXPLORING", 2: "COVERED", 3: "COVERED_BY_OTHERS",
                 4: "NOGO", 5: "EXPLORING_BY_OTHERS"}


class RegionExecutionState(str, Enum):
    PENDING = "pending_native_region"
    DISPATCHED = "native_region_dispatched"
    EXECUTED = "native_region_executed_with_evidence"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name)
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name)
    return value


def _xyz(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError(name)
    return tuple(float(x) for x in value)


def _sources(value):
    if not isinstance(value, list) or not value or len(set(value)) != len(value):
        raise ValueError("SOURCE_FRAME_KEYS")
    for key in value:
        _text(key, "SOURCE_FRAME_KEY")
    return tuple(value)


def _decode(value):
    if isinstance(value, str):
        def unique(pairs):
            out = {}
            for key, item in pairs:
                if key in out:
                    raise ValueError("DUPLICATE_JSON_KEY")
                out[key] = item
            return out
        value = json.loads(value, object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return deepcopy(value)


@dataclass(frozen=True)
class RegionObservation:
    native_candidate_id: int
    position_xyz_m: tuple[float, float, float]
    status: int
    status_name: str
    binding: NativeSnapshotBinding
    record_ref: str


@dataclass(frozen=True)
class _Region:
    task_id: str
    candidate_id: int
    position_xyz_m: tuple[float, float, float]
    observations: tuple[RegionObservation, ...]
    execution_state: RegionExecutionState = RegionExecutionState.PENDING
    active_dispatch_id: str | None = None
    executed_attempt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Attempt:
    dispatch_id: str
    candidate_id: int
    dispatch_epoch: str
    stamp_ns: int
    event_index: int
    dispatch_record_ref: str
    visit_evidence_id: str | None = None
    covered_evidence_id: str | None = None
    executed: bool = False


class NativeRegionTaskRegistry:
    """Production JSON consumer; callers preserve the referenced raw records.

    ``consume(kind, payload, order=..., record_ref=...)`` accepts the unmodified
    native snapshot/feedback schemas emitted by the native bridge. ``order`` is
    replay receipt order, not a fabricated mapping between scan namespaces.
    Missing/unsupported execution evidence is reported and never filled in.
    """

    def __init__(self, *, segment: str, trajectory_id: str, actor_robot_id: int = 0):
        self.segment = _text(segment, "SEGMENT")
        self.trajectory_id = _text(trajectory_id, "TRAJECTORY_ID")
        self.actor_robot_id = _integer(actor_robot_id, "ACTOR_ROBOT_ID")
        self._session = None
        self._last_order = -1
        self._last_event_index = -1
        self._last_feedback_stamp = -1
        self._last_snapshot_sequence = -1
        self._last_snapshot_stamp = -1
        self._last_pose_sequence = -1
        self._last_pose_stamp = -1
        self._pose_sources = {}
        self._pose_memberships = {}
        self._snapshots = {}
        self._latest_epoch = None
        self._regions: dict[int, _Region] = {}
        self._attempts: dict[str, _Attempt] = {}
        self._trace = []
        self._feedback_records = {}
        self._route_intents = {}
        self._waypoint_route_links = []
        self._trajectory = []
        self._anchors = StructuralTaskLifecycle()
        self._last_anchor_region = None
        self._last_anchor_id = None
        self._traversal_trace = []
        self._native_finish = None
        self._at_home = None
        self._local_execution = NativeLocalExecution(segment=segment, trajectory_id=trajectory_id)

    def consume(self, kind, payload, *, order: int, record_ref: str):
        """Return a trace decision; invalid input does not alter task states."""
        _integer(order, "REPLAY_ORDER")
        _text(record_ref, "RECORD_REF")
        if order <= self._last_order:
            raise ValueError("STRICT_REPLAY_ORDER_REQUIRED")
        self._last_order = order
        try:
            message = _decode(payload)
            if kind == "snapshot":
                outcome = self._consume_snapshot(message, order, record_ref)
            elif kind == "feedback":
                outcome = self._consume_feedback(message, order, record_ref)
            elif kind == "decision":
                outcome = self._consume_decision(message, order, record_ref)
            else:
                raise ValueError("UNSUPPORTED_RECORD_KIND")
        except (ValueError, KeyError, TypeError) as error:
            outcome = dict(accepted=False, reason=str(error), task_state_changed=False)
        decision = dict(order=order, record_ref=record_ref, kind=kind, **outcome)
        self._trace.append(deepcopy(decision))
        return deepcopy(decision)

    def _consume_decision(self, message, order, record_ref):
        """Retain the native final route as INTENT, never region completion.

        This fills the dropped decision-message interface. It does not replace
        an actual waypoint's cell with the first/global route cell.
        """
        if message.get('schema_version') != 'native_route_decision_v1':
            raise ValueError('NATIVE_DECISION_SCHEMA')
        epoch = message.get('epoch')
        if epoch != self._latest_epoch or epoch not in self._snapshots:
            raise ValueError('DECISION_NOT_BOUND_TO_CURRENT_SNAPSHOT')
        if epoch in self._route_intents:
            raise ValueError('DUPLICATE_NATIVE_ROUTE_INTENT')
        snapshot = self._snapshots[epoch]
        if (message.get('source_frame_keys') != snapshot['source_frame_keys']
                or message.get('original_route') != snapshot['original_route']):
            raise ValueError('NATIVE_DECISION_SOURCE_OR_ORIGINAL_ROUTE_MISMATCH')
        route = message.get('route')
        original = snapshot['original_route']
        if (not isinstance(route,list) or any(type(n) is not int for n in route)
                or Counter(route) != Counter(original) or route[0] != original[0] or route[-1] != original[-1]):
            raise ValueError('NATIVE_DECISION_CHANGED_CANDIDATES_OR_DEPOT')
        if (message.get('native_candidates_preserved') is not True or message.get('native_edges_changed') is not False
                or type(message.get('changed')) is not bool or message['changed'] != (route != original)):
            raise ValueError('NATIVE_DECISION_CLAIMS_INCONSISTENT')
        mode = message.get('mode')
        if mode not in ('disabled','shadow','active') or (mode != 'active' and route != original):
            raise ValueError('NONACTIVE_DECISION_CHANGED_NATIVE_ROUTE')
        cost = route_cost(route,validate_snapshot(snapshot))
        if cost > snapshot['original_cost_units'] + snapshot['max_extra_cost_units']:
            raise ValueError('NATIVE_DECISION_EXCEEDS_FIXED_COST_LIMIT')
        self._route_intents[epoch] = dict(epoch=epoch,order=order,record_ref=record_ref,
            route=list(route),source_frame_keys=list(snapshot['source_frame_keys']),
            mode=mode,native_route_changed=message['changed'],task_execution_verified=False,
            task_completion_verified=False,raw_decision=message)
        return dict(accepted=True,reason='NATIVE_ROUTE_INTENT_NOT_TASK_EXECUTION',
                    task_state_changed=False,epoch=epoch)

    def _link_waypoint(self, message, record_ref):
        position = _xyz(message['xyz_m'],'ACTUAL_PUBLISHED_WAYPOINT_REQUIRED')
        intent = self._route_intents.get(message.get('epoch'))
        bound = (intent is not None and message.get('epoch') == self._latest_epoch
                 and intent['source_frame_keys'] == message['source_frame_keys'])
        # Native producer emits path evidence immediately before PublishWaypoint.
        # A gap/rejection/other feedback must never reuse an older path, even in
        # the same epoch. The lookahead and extended waypoint need not coincide.
        previous_id = f"{message['session_id']}:event:{message['event_index'] - 1}"
        previous = self._feedback_records.get(previous_id)
        path = previous['payload'] if previous else {}
        path_bound = (path.get('event') == 'NATIVE_PLANNING_PATHS'
            and path.get('paths_complete') is True and path.get('epoch') == message.get('epoch')
            and path.get('source_frame_keys') == message['source_frame_keys'])
        self._waypoint_route_links.append(dict(waypoint_evidence_id=message['evidence_id'],
            waypoint_record_ref=record_ref,waypoint_xyz_m=position,
            route_intent_record_ref=intent['record_ref'] if bound else None,
            epoch=message.get('epoch'),bound=bound,scope='native_route_and_local_waypoint_same_epoch_not_target_identity',
            planning_path_record_ref=previous['record_ref'] if path_bound else None,
            planning_path_evidence_id=previous_id if path_bound else None,
            planning_path_bound=path_bound,
            global_task_reached=False,task_completed=False))
        return dict(accepted=True,reason='LOCAL_WAYPOINT_LINKED_TO_ROUTE_INTENT' if bound else 'LOCAL_WAYPOINT_WITHOUT_PRIOR_ROUTE_INTENT',
                    task_state_changed=False,route_intent_bound=bound)

    def _validate_planning_paths(self, message):
        if (message.get('path_schema') != 'native_planning_paths_v1'
                or message.get('coordinate_frame') != 'map'
                or message.get('evidence_scope') != 'native_planned_paths_not_executed_traversal'):
            raise ValueError('NATIVE_PLANNING_PATH_SCHEMA_FRAME_SCOPE')
        for key in ('paths_complete', 'lookahead_updated', 'follow_local_path_from_start'):
            if type(message.get(key)) is not bool:
                raise ValueError('EXPLICIT_PLANNING_PATH_FLAGS')
        _xyz(message['lookahead_xyz_m'], 'NATIVE_LOOKAHEAD_XYZ')
        total = sum(_integer(message[key], 'NATIVE_PATH_COUNT')
                    for key in ('global_node_count', 'exploration_node_count'))
        if message['paths_complete'] != (total <= 16384):
            raise ValueError('NATIVE_PATH_RECORDING_BOUND')
        for key, count in (('global_path', 'global_node_count'), ('exploration_path', 'exploration_node_count')):
            nodes = message[key]
            if not message['paths_complete']:
                if nodes is not None:
                    raise ValueError('NO_TRUNCATED_NATIVE_PATH')
                continue
            if not isinstance(nodes, list) or len(nodes) != message[count]:
                raise ValueError('NATIVE_PATH_COUNT_MISMATCH')
            for node in nodes:
                if not isinstance(node, dict):
                    raise ValueError('NATIVE_PATH_NODE_OBJECT')
                _xyz(node['xyz_m'], 'NATIVE_PATH_NODE_XYZ')
                if type(node.get('native_node_type')) is not int or node['native_node_type'] not in (0,1,2,3,4,5,6,7,8,10):
                    raise ValueError('NATIVE_PATH_NODE_TYPE')
                if type(node.get('native_subspace_index')) is not int or node['native_subspace_index'] < -1:
                    raise ValueError('NATIVE_PATH_SUBSPACE_INDEX')

    def _consume_snapshot(self, message, order, record_ref):
        validate_snapshot(message)
        epoch = message["epoch"]
        session, separator, sequence = epoch.rpartition(":")
        if not separator or not session or not sequence.isdecimal():
            raise ValueError("NATIVE_SESSION_EPOCH_REQUIRED")
        if self._session is not None and session != self._session:
            raise ValueError("SESSION_CHANGED_NEW_REGISTRY_REQUIRED")
        if epoch in self._snapshots:
            raise ValueError("SNAPSHOT_EPOCH_ALREADY_RECORDED")
        if int(sequence) <= self._last_snapshot_sequence or message["stamp_ns"] < self._last_snapshot_stamp:
            raise ValueError("NONCAUSAL_NATIVE_SNAPSHOT")
        if message.get("coordinate_frame") != "map":
            raise ValueError("NATIVE_MAP_FRAME_REQUIRED")
        _xyz(message["robot_position_xyz_m"], "SNAPSHOT_ROBOT_POSITION")
        records = message["candidate_records"]
        if not isinstance(records, list) or len(records) != len(message["candidate_ids"]):
            raise ValueError("EXACT_NATIVE_CANDIDATE_RECORDS_REQUIRED")
        rows = {}
        binding = NativeSnapshotBinding(epoch, message["stamp_ns"], tuple(message["source_frame_keys"]), order)
        for record in records:
            cid = _integer(record["id"], "NATIVE_CANDIDATE_ID")
            if cid in rows or cid not in message["candidate_ids"]:
                raise ValueError("EXACT_NATIVE_CANDIDATE_ID_BINDING_REQUIRED")
            position = _xyz(record["position_xyz_m"], "NATIVE_CELL_POSITION")
            if position != _xyz(message["candidate_positions_m"][str(cid)], "NATIVE_CELL_POSITION_LOOKUP"):
                raise ValueError("NATIVE_CELL_POSITION_MISMATCH")
            status = _integer(record["status"], "NATIVE_CELL_STATUS")
            if NATIVE_STATUS.get(status) != record["status_name"]:
                raise ValueError("NATIVE_CELL_STATUS_NAME_MISMATCH")
            if cid in self._regions and self._regions[cid].position_xyz_m != position:
                raise ValueError("NATIVE_CELL_ID_GEOMETRY_CHANGED")
            rows[cid] = RegionObservation(cid, position, status, record["status_name"], binding, record_ref)
        self._session = session
        self._snapshots[epoch] = message
        self._latest_epoch = epoch
        self._last_snapshot_sequence, self._last_snapshot_stamp = int(sequence), message["stamp_ns"]
        for cid, observation in rows.items():
            if cid in self._regions:
                old = self._regions[cid]
                self._regions[cid] = replace(old, observations=(*old.observations, observation))
            else:
                self._regions[cid] = _Region(f"{self.segment}:native_region:{cid}", cid,
                    observation.position_xyz_m, (observation,))
        return dict(accepted=True, reason="NATIVE_REGIONS_PRESERVED_DIRECTION_SCOPE_UNKNOWN",
                    task_state_changed=bool(rows), epoch=epoch,
                    native_candidate_ids=list(message["candidate_ids"]),
                    region_task_bindings={str(cid): self._regions[cid].task_id for cid in rows},
                    direction_task_bindings={str(cid): None for cid in rows})

    def _feedback_envelope(self, message):
        if message.get("schema_version") != "native_route_feedback_v1":
            raise ValueError("FEEDBACK_SCHEMA")
        session = _text(message["session_id"], "FEEDBACK_SESSION")
        if self._session is not None and session != self._session:
            raise ValueError("SESSION_CHANGED_NEW_REGISTRY_REQUIRED")
        index = _integer(message["event_index"], "FEEDBACK_EVENT_INDEX")
        stamp = _integer(message["stamp_ns"], "FEEDBACK_STAMP")
        if index <= self._last_event_index or stamp < self._last_feedback_stamp:
            raise ValueError("NONCAUSAL_OR_DUPLICATE_FEEDBACK")
        evidence_id = _text(message["evidence_id"], "FEEDBACK_EVIDENCE_ID")
        if evidence_id != f"{session}:event:{index}":
            raise ValueError("FEEDBACK_EVIDENCE_ID_BINDING")
        _text(message["event"], "FEEDBACK_EVENT")
        if not isinstance(message["epoch"], str):
            raise ValueError("FEEDBACK_EPOCH")
        # Early native events can precede five available scans. Such events
        # remain logged but cannot establish an epoch-bound dispatch.
        if message["source_frame_keys"]:
            _sources(message["source_frame_keys"])
        elif not isinstance(message["source_frame_keys"], list):
            raise ValueError("SOURCE_FRAME_KEYS")
        return session, index, stamp, evidence_id

    def _consume_feedback(self, message, order, record_ref):
        session, index, stamp, evidence_id = self._feedback_envelope(message)
        event = message["event"]
        # Validate event-local payloads before mutating task or anchor state.
        if event in ("NATIVE_REGION_DISPATCH", "NATIVE_REGION_VISIT", "NATIVE_REGION_STATUS"):
            cid = _integer(message["candidate_id"], "FEEDBACK_NATIVE_CELL_ID")
            _xyz(message["position_xyz_m"], "FEEDBACK_NATIVE_CELL_CENTER")
            if cid in self._regions and tuple(message["position_xyz_m"]) != self._regions[cid].position_xyz_m:
                raise ValueError("FEEDBACK_NATIVE_CELL_CENTER_MISMATCH")
        if event == "WAYPOINT":
            result = self._link_waypoint(message, record_ref)
        elif event == "NATIVE_PLANNING_PATHS":
            self._validate_planning_paths(message)
            result = dict(accepted=True, reason="PLANNED_PATH_RECORDED_NOT_EXECUTED_EDGE", task_state_changed=False)
        elif event == "NATIVE_REGION_DISPATCH":
            result = self._dispatch(message, record_ref)
        elif event == "NATIVE_REGION_VISIT":
            result = self._visit(message, order, record_ref)
        elif event == "NATIVE_REGION_STATUS":
            result = self._covered(message)
        elif event == "NATIVE_POSE":
            self._validate_pose(message)
            self._trajectory.append(dict(record_ref=record_ref, evidence=message))
            self._commit_pose(message)
            result = dict(accepted=True, reason="ACTUAL_NATIVE_POSE_RECORDED", task_state_changed=False)
        elif event == "NATIVE_EXECUTION_STATE":
            if type(message["native_finish"]) is not bool or type(message["at_home"]) is not bool:
                raise ValueError("EXPLICIT_NATIVE_FINISH_HOME_REQUIRED")
            self._native_finish, self._at_home = message["native_finish"], message["at_home"]
            result = dict(accepted=True, reason="NATIVE_FINISH_IS_NOT_STRUCTURAL_COMPLETION", task_state_changed=False)
        else:
            result = dict(accepted=True, reason="NATIVE_EVENT_RETAINED_WITHOUT_TASK_INFERENCE", task_state_changed=False)
        # This separate scope accepts a *local* command even when its waypoint
        # cell is not a distant global candidate. It never upgrades a rejected
        # regional dispatch, completes a direction, or creates a graph edge.
        local_execution = self._local_execution.consume(message, record_ref=record_ref,
            waypoint_link=self._waypoint_route_links[-1] if event == 'WAYPOINT' else None,
            route_intent=self._route_intents.get(message.get('epoch')))
        self._session = session
        self._last_event_index, self._last_feedback_stamp = index, stamp
        self._feedback_records[evidence_id] = dict(record_ref=record_ref, payload=message)
        return dict(event=event, evidence_id=evidence_id, local_execution=local_execution, **result)

    def _dispatch(self, message, record_ref):
        cid, epoch = message["candidate_id"], message["epoch"]
        if epoch != self._latest_epoch or epoch not in self._snapshots:
            return dict(accepted=False, reason="DISPATCH_NOT_BOUND_TO_CURRENT_SNAPSHOT", task_state_changed=False)
        snapshot = self._snapshots[epoch]
        if cid not in snapshot["candidate_ids"]:
            return dict(accepted=False, reason="WAYPOINT_REGION_NOT_A_NATIVE_CANDIDATE", task_state_changed=False)
        if tuple(message["source_frame_keys"]) != tuple(snapshot["source_frame_keys"]):
            return dict(accepted=False, reason="DISPATCH_SOURCE_BINDING_MISMATCH", task_state_changed=False)
        if message["stamp_ns"] < snapshot["stamp_ns"]:
            raise ValueError("DISPATCH_BEFORE_SNAPSHOT")
        if message.get("dispatch_scope") != "published_native_waypoint_region_not_route_first_cell":
            raise ValueError("EXPLICIT_PUBLISHED_WAYPOINT_DISPATCH_REQUIRED")
        _xyz(message["waypoint_xyz_m"], "DISPATCH_WAYPOINT")
        _xyz(message["robot_position_xyz_m"], "DISPATCH_ROBOT_POSITION")
        attempt = _Attempt(message["evidence_id"], cid, epoch, message["stamp_ns"],
            message["event_index"], record_ref)
        self._attempts[attempt.dispatch_id] = attempt
        self._regions[cid] = replace(self._regions[cid], execution_state=RegionExecutionState.DISPATCHED,
            active_dispatch_id=attempt.dispatch_id)
        return dict(accepted=True, reason="EXPLICIT_NATIVE_REGION_DISPATCH", task_state_changed=True,
                    region_task_id=self._regions[cid].task_id)

    def _bound_attempt(self, message):
        dispatch_id = message.get("dispatch_evidence_id")
        attempt = self._attempts.get(dispatch_id)
        if attempt is None:
            return None, "NO_MATCHING_EXPLICIT_DISPATCH"
        if (attempt.candidate_id != message["candidate_id"]
                or attempt.dispatch_epoch != message.get("dispatch_epoch")):
            return None, "DISPATCH_REGION_OR_EPOCH_MISMATCH"
        if message["event_index"] <= attempt.event_index or message["stamp_ns"] < attempt.stamp_ns:
            return None, "EXECUTION_EVIDENCE_PRECEDES_DISPATCH"
        return attempt, None

    def _validate_pose(self, message):
        _xyz(message["robot_position_xyz_m"], "ACTUAL_ROBOT_POSITION")
        if message.get("pose_frame_id") != "map":
            raise ValueError("POSE_NOT_IN_DECLARED_NATIVE_MAP_FRAME")
        pose_stamp = _integer(message["pose_stamp_ns"], "POSE_STAMP")
        if pose_stamp > message["stamp_ns"]:
            raise ValueError("FUTURE_POSE")
        _integer(message["pose_sequence"], "POSE_SEQUENCE")
        _text(message["pose_source_key"], "POSE_SOURCE_KEY")
        quat = message["robot_orientation_xyzw"]
        if (not isinstance(quat, list) or len(quat) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in quat)
                or not math.isclose(math.hypot(*quat), 1.0, rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError("VALID_ACTUAL_ROBOT_ORIENTATION_REQUIRED")
        if message["pose_sequence"] < self._last_pose_sequence or pose_stamp < self._last_pose_stamp:
            raise ValueError("NONCAUSAL_NATIVE_POSE")
        signature = (pose_stamp, tuple(message["robot_position_xyz_m"]), tuple(quat))
        old = self._pose_sources.get(message["pose_source_key"])
        if old is not None and old != signature:
            raise ValueError("NATIVE_POSE_SOURCE_REWRITTEN")

    def _commit_pose(self, message):
        self._last_pose_sequence = message["pose_sequence"]
        self._last_pose_stamp = message["pose_stamp_ns"]
        self._pose_sources[message["pose_source_key"]] = (message["pose_stamp_ns"],
            tuple(message["robot_position_xyz_m"]), tuple(message["robot_orientation_xyzw"]))

    def _visit(self, message, order, record_ref):
        if message.get("pose_evidence_kind") != "native_state_estimation_cell_membership":
            raise ValueError("EXPLICIT_NATIVE_POSE_MEMBERSHIP_REQUIRED")
        if message.get("pose_binding_exact") is not True:
            return dict(accepted=False, reason="ACTUAL_POSE_BINDING_UNKNOWN", task_state_changed=False)
        self._validate_pose(message)
        # No current producer supplies the independent continuity certificate.
        # Do not silently reinterpret a future producer's boolean as one.
        if message.get("continuous_trajectory_verified") is not False:
            raise ValueError("UNSUPPORTED_CONTINUOUS_TRAJECTORY_CLAIM")
        attempt, reason = self._bound_attempt(message)
        if attempt is not None and message["pose_stamp_ns"] < attempt.stamp_ns:
            attempt, reason = None, "VISIT_POSE_PRECEDES_DISPATCH"
        cid = message["candidate_id"]
        old_membership = self._pose_memberships.get(message["pose_source_key"])
        if old_membership is not None and old_membership != cid:
            raise ValueError("NATIVE_POSE_REGION_MEMBERSHIP_CONFLICT")
        anchor_id = None
        if cid != self._last_anchor_region:
            anchor_id = f"{self.segment}:pose_anchor:{message['evidence_id']}"
            refs = tuple(dict.fromkeys((record_ref, message["evidence_id"], message["pose_source_key"])))
            self._anchors.record_anchor_visit(AnchorVisit(message["evidence_id"], anchor_id,
                self.segment, self.trajectory_id, order, tuple(message["robot_position_xyz_m"]), refs, True))
            if self._last_anchor_id is not None:
                self._traversal_trace.append(dict(from_anchor=self._last_anchor_id, to_anchor=anchor_id,
                    confirmed=False, reason="CONTINUOUS_TRAJECTORY_NOT_VERIFIED", evidence_id=message["evidence_id"]))
            self._last_anchor_region, self._last_anchor_id = cid, anchor_id
        self._trajectory.append(dict(record_ref=record_ref, evidence=message))
        self._commit_pose(message)
        self._pose_memberships[message["pose_source_key"]] = cid
        completed = False
        if attempt is not None:
            attempt = replace(attempt, visit_evidence_id=message["evidence_id"])
            completed = self._save_attempt(attempt)
        return dict(accepted=True, reason="ACTUAL_REGION_MEMBERSHIP_NOT_DIRECTION_COMPLETION",
                    task_state_changed=completed, region_execution_rejection_reason=reason,
                    native_region_executed=completed, anchor_id=anchor_id)

    def _covered(self, message):
        current = _integer(message["status"], "CURRENT_NATIVE_STATUS")
        previous = _integer(message["previous_status"], "PREVIOUS_NATIVE_STATUS")
        _integer(message["actor_robot_id"], "ACTOR_ROBOT_ID")
        if NATIVE_STATUS.get(current) != message["status_name"] or previous not in NATIVE_STATUS:
            raise ValueError("NATIVE_STATUS_TRANSITION_SCHEMA")
        if message.get("physical_exploration_verified") is not False or message.get("direction_complete") is not False:
            raise ValueError("NATIVE_CELL_STATUS_CANNOT_CERTIFY_DIRECTIONS_OR_PHYSICAL_COMPLETION")
        if type(message.get("native_coverage_rule_satisfied")) is not bool:
            raise ValueError("EXPLICIT_NATIVE_COVERAGE_RULE_VERDICT_REQUIRED")
        if (current == 1 and previous in (2, 3) and message["actor_robot_id"] == self.actor_robot_id
                and message["candidate_id"] in self._regions):
            cid = message["candidate_id"]
            self._regions[cid] = replace(self._regions[cid], execution_state=RegionExecutionState.PENDING,
                                        active_dispatch_id=None)
            return dict(accepted=True, reason="EXPLICIT_NATIVE_REGION_REOPENED_DIRECTIONS_STILL_UNKNOWN", task_state_changed=True)
        if (current != 2 or previous == 2 or message.get("native_coverage_rule_satisfied") is not True
                or message.get("native_rule") != "local_residual_coverage_below_threshold"
                or message.get("actor_robot_id") != self.actor_robot_id):
            return dict(accepted=True, reason="NATIVE_STATUS_NOT_QUALIFIED_LOCAL_COVERAGE_TRANSITION", task_state_changed=False)
        attempt, reason = self._bound_attempt(message)
        if attempt is None:
            return dict(accepted=False, reason=reason, task_state_changed=False)
        completed = self._save_attempt(replace(attempt, covered_evidence_id=message["evidence_id"]))
        return dict(accepted=True, reason="NATIVE_LOCAL_COVERAGE_EXECUTION_RECORDED" if completed else "WAITING_FOR_ACTUAL_POST_DISPATCH_REGION_VISIT",
                    task_state_changed=completed, native_region_executed=completed)

    def _save_attempt(self, attempt):
        just_executed = not attempt.executed and bool(attempt.visit_evidence_id and attempt.covered_evidence_id)
        if just_executed:
            attempt = replace(attempt, executed=True)
            region = self._regions[attempt.candidate_id]
            self._regions[attempt.candidate_id] = replace(region,
                execution_state=RegionExecutionState.EXECUTED if region.active_dispatch_id == attempt.dispatch_id else region.execution_state,
                executed_attempt_ids=(*region.executed_attempt_ids, attempt.dispatch_id))
        self._attempts[attempt.dispatch_id] = attempt
        return just_executed

    def snapshot(self):
        regions = []
        for region in self._regions.values():
            row = asdict(region)
            row.update(task_kind="native_grid_region_not_direction", direction_scope="unknown_unenumerated",
                       direction_task_ids=[], all_region_directions_completed=False,
                       execution_evidence_scope="native_local_coverage_rule_and_recorded_membership_only")
            regions.append(row)
        anchors = self._anchors.snapshot()
        return deepcopy(dict(schema_version="native_region_task_registry_v1", segment=self.segment,
            trajectory_id=self.trajectory_id, session_id=self._session, latest_epoch=self._latest_epoch,
            tasks=regions, native_candidate_snapshots=self._snapshots, attempts={key: asdict(value) for key, value in self._attempts.items()},
            anchors=anchors["visits"], confirmed_traversals=anchors["confirmed_edges"],
            traversal_trace=self._traversal_trace, trajectory=self._trajectory, decision_trace=self._trace,
            feedback_records=self._feedback_records, native_finish=self._native_finish, at_home=self._at_home,
            route_intents=self._route_intents,waypoint_route_links=self._waypoint_route_links,
            local_execution=self._local_execution.snapshot(),
            direction_task_count=0, regions_with_unknown_direction_scope=len(regions),
            candidate_removed=False, full_exploration_proven=False))


def replay_native_region_records(records, *, segment, trajectory_id, actor_robot_id=0):
    """Consume JSON objects/JSONL strings with kind, payload, order, record_ref."""
    registry = NativeRegionTaskRegistry(segment=segment, trajectory_id=trajectory_id, actor_robot_id=actor_robot_id)
    for raw in records:
        record = _decode(raw)
        registry.consume(record["kind"], record["payload"], order=record["order"], record_ref=record["record_ref"])
    return registry.snapshot()
