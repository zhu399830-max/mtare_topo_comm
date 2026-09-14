"""Bind native published local commands to subsequent recorded odometry.

This is execution provenance, NOT a direction detector, arrival classifier,
controller acknowledgement or exploration-completion producer. In particular,
the waypoint's cell need not be a remote global-VRP candidate. No GT, distance
threshold, planner change or new completion rule is involved.
"""
from __future__ import annotations

from copy import deepcopy
import math


class NativeLocalExecution:
    """Consume envelopes already validated by NativeRegionTaskRegistry.

    Feedback gaps and each new publication terminate the preceding command's
    observation interval. Odometry is linked only by the native producer's
    consecutive sequence AND previous-stamp chain. Recorded displacements do
    not prove continuous physical motion between samples or causal benefit.
    """

    def __init__(self, *, segment, trajectory_id):
        self.segment, self.trajectory_id = segment, trajectory_id
        self._session = None
        self._index = None
        self._pose = None
        self._waypoint = None
        self._active = None
        self._commands = {}
        self._events = []

    def _close(self, reason, ref):
        if self._active is not None:
            self._events.append(dict(kind="interval_closed", command_id=self._active,
                                     reason=reason, record_ref=ref, task_completed=False))
            self._active = None

    @staticmethod
    def _pose_record(message, ref):
        return dict(source_key=message['pose_source_key'], stamp_ns=message['pose_stamp_ns'],
                    sequence=message['pose_sequence'], xyz_m=tuple(message['robot_position_xyz_m']),
                    orientation_xyzw=tuple(message['robot_orientation_xyzw']),
                    frame_id=message['pose_frame_id'], record_ref=ref,
                    evidence_id=message['evidence_id'])

    def consume(self, message, *, record_ref, waypoint_link=None, route_intent=None):
        session, index = message['session_id'], message['event_index']
        if self._session is not None and session != self._session:
            raise ValueError('LOCAL_EXECUTION_SESSION_CHANGED')
        if self._index is not None and index <= self._index:
            raise ValueError('LOCAL_EXECUTION_NONCAUSAL_FEEDBACK')
        if self._index is not None and index != self._index + 1:
            self._close('FEEDBACK_GAP', record_ref)
            self._waypoint = None
            self._pose = None
        self._session, self._index = session, index
        event = message['event']
        if event == 'WAYPOINT':
            self._close('SUPERSEDED_BY_PUBLISHED_WAYPOINT', record_ref)
            self._waypoint = dict(message=deepcopy(message), record_ref=record_ref,
                                  link=deepcopy(waypoint_link), intent=deepcopy(route_intent))
            return dict(bound=False, reason='PUBLISHED_COMMAND_AWAITS_EXACT_POSE_BINDING')
        if event == 'NATIVE_POSE':
            return self._observe_pose(message, record_ref)
        if event == 'NATIVE_REGION_DISPATCH':
            return self._dispatch(message, record_ref)
        return dict(bound=False, reason='NO_LOCAL_COMMAND_EVENT')

    def _dispatch(self, message, ref):
        pending = self._waypoint
        waypoint = pending['message'] if pending else {}
        pose = self._pose
        if (not pending or waypoint['event_index'] + 1 != message['event_index']
                or waypoint['epoch'] != message['epoch']
                or waypoint['source_frame_keys'] != message['source_frame_keys']
                or tuple(waypoint['xyz_m']) != tuple(message.get('waypoint_xyz_m', ()))
                or message.get('dispatch_scope') != 'published_native_waypoint_region_not_route_first_cell'):
            return dict(bound=False, reason='NO_EXACT_PRECEDING_PUBLISHED_WAYPOINT')
        # The source must have actually been seen in this consumer, not just
        # copied into a dispatch payload. Region visits are not pose callbacks.
        if (pose is None or message.get('pose_binding_exact') is not True
                or message.get('pose_source_key') != pose['source_key']
                or message.get('pose_stamp_ns') != pose['stamp_ns']
                or message.get('pose_sequence') != pose['sequence']
                or message.get('pose_frame_id') != 'map'
                or pose['frame_id'] != 'map'
                or tuple(message.get('robot_position_xyz_m', ())) != pose['xyz_m']
                or tuple(message.get('robot_orientation_xyzw', ())) != pose['orientation_xyzw']
                or pose['stamp_ns'] > waypoint['stamp_ns']):
            return dict(bound=False, reason='COMMAND_POSE_NOT_BOUND_TO_RECEIVED_NATIVE_POSE')
        delta = tuple(float(a) - float(b) for a, b in zip(waypoint['xyz_m'], pose['xyz_m']))
        length = math.hypot(*delta)
        if not math.isfinite(length):
            return dict(bound=False, reason='LOCAL_COMMAND_GEOMETRY_OUT_OF_NUMERIC_RANGE')
        direction = None if length == 0 else tuple(x / length for x in delta)
        link, intent = pending['link'] or {}, pending['intent'] or {}
        route_bound = link.get('bound') is True and intent.get('epoch') == message['epoch']
        command_id = message['evidence_id']
        self._commands[command_id] = dict(command_id=command_id, segment=self.segment,
            trajectory_id=self.trajectory_id, epoch=message['epoch'],
            dispatch_record_ref=ref, waypoint_evidence_id=waypoint['evidence_id'],
            waypoint_record_ref=pending['record_ref'], command_stamp_ns=waypoint['stamp_ns'],
            source_frame_keys=list(message['source_frame_keys']),
            waypoint_xyz_m=tuple(waypoint['xyz_m']), start_pose=deepcopy(pose),
            local_command_direction_xyz=direction, native_waypoint_region_id=message['candidate_id'],
            route_intent_record_ref=intent.get('record_ref') if route_bound else None,
            native_route_changed=bool(intent.get('native_route_changed')) if route_bound else None,
            planning_path_record_ref=link.get('planning_path_record_ref'),
            scope='published_local_command_and_recorded_odometry_not_channel_completion',
            remote_task_id=None, semantic_direction_id=None, level_id=None,
            controller_acknowledged=False, task_completed=False,
            physical_continuity_verified=False, learned_motion_effect_verified=False)
        self._active = command_id
        self._events.append(dict(kind='command_bound', command_id=command_id, record_ref=ref))
        return dict(bound=True, command_id=command_id, reason='LOCAL_COMMAND_BOUND_NOT_REMOTE_TASK_DISPATCH')

    def _observe_pose(self, message, ref):
        current = self._pose_record(message, ref)
        previous = self._pose
        if previous is not None:
            same_source = all(current[key] == previous[key] for key in
                ('source_key', 'stamp_ns', 'sequence', 'xyz_m', 'orientation_xyzw', 'frame_id'))
            if same_source:
                # A repeated source remains the original evidence, not a new
                # odometry callback or an additional movement sample.
                return dict(bound=False, reason='DUPLICATE_POSE_SOURCE_NOT_NEW_MOTION')
            if current['sequence'] <= previous['sequence'] or current['stamp_ns'] <= previous['stamp_ns']:
                self._close('ODOMETRY_SOURCE_NOT_ADVANCING', ref)
                self._pose = None
                return dict(bound=False, reason='ODOMETRY_SOURCE_NOT_ADVANCING')
        if self._active is not None:
            command = self._commands[self._active]
            if (previous is None or current['sequence'] != previous['sequence'] + 1
                    or message.get('previous_pose_stamp_ns') != previous['stamp_ns']
                    or current['stamp_ns'] <= previous['stamp_ns']
                    or current['source_key'] == previous['source_key']
                    or current['stamp_ns'] < command['command_stamp_ns']):
                self._close('ODOMETRY_SOURCE_CHAIN_NOT_CONSECUTIVE', ref)
            else:
                step = math.dist(current['xyz_m'], previous['xyz_m'])
                offset = tuple(a-b for a,b in zip(current['xyz_m'], command['start_pose']['xyz_m']))
                direction = command['local_command_direction_xyz']
                progress = None if direction is None else sum(a*b for a,b in zip(offset,direction))
                distance = math.dist(current['xyz_m'], command['waypoint_xyz_m'])
                if not all(math.isfinite(v) for v in (step, distance, progress if progress is not None else 0.)):
                    self._close('ODOMETRY_GEOMETRY_OUT_OF_NUMERIC_RANGE', ref)
                    self._pose = current
                    return dict(bound=False, reason='ODOMETRY_GEOMETRY_OUT_OF_NUMERIC_RANGE')
                self._events.append(dict(kind='odometry_observed', command_id=self._active,
                    record_ref=ref, pose=deepcopy(current), previous_pose_ref=previous['record_ref'],
                    observed_step_chord_m=step,
                    displacement_along_command_m=progress,
                    distance_to_published_waypoint_m=distance,
                    source_chain_verified=True, physical_continuity_verified=False,
                    position_reached=None, task_completed=False))
        self._pose = current
        return dict(bound=False, reason='NATIVE_ODOMETRY_RECORDED_FOR_LOCAL_COMMANDS')

    def snapshot(self):
        return deepcopy(dict(schema_version='native_local_execution_v1',
            segment=self.segment, trajectory_id=self.trajectory_id, session_id=self._session,
            commands=self._commands, events=self._events, active_command_id=self._active,
            structural_task_completions=0, confirmed_graph_edges=0,
            scope='recorded_command_execution_provenance_only'))
