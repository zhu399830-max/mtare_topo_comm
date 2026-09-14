"""Bind tokens' native epochs to actual metric visits, not semantic identity.

A contiguous visit to one native grid region is a provenance grouping only.
Two returns to the same region get distinct visit IDs. No direction, level,
task completion, global place union or traversed edge is inferred.
"""
from copy import deepcopy

from mtare_topo.integration.native_route_advice import validate_snapshot


class NativeEpochVisitBinding:
    def __init__(self):
        self.session = None
        self.last_event = -1
        self.last_epoch_sequence = -1
        self.latest_visit = None
        self.visit_start = None
        self.region = None

    def consume(self, row):
        message = row['payload']
        if row['topic'].endswith('/feedback'):
            session, event = message['session_id'], message['event_index']
            if self.session not in (None,session) or type(event) is not int or event <= self.last_event:
                raise ValueError('native session/event sequence discontinuity')
            if self.last_event >= 0 and event != self.last_event + 1:
                # A missing feedback could have contained an intermediate visit.
                self.latest_visit = None
                self.region = None
                self.visit_start = None
            self.session, self.last_event = session,event
            if message.get('event') != 'NATIVE_REGION_VISIT':
                return None
            if (message.get('pose_binding_exact') is not True or message.get('pose_frame_id') != 'map'
                    or message.get('pose_evidence_kind') != 'native_state_estimation_cell_membership'):
                # Do not let later snapshots inherit an earlier valid visit.
                self.latest_visit = None
                self.region = None
                self.visit_start = None
                return None
            cid = message['candidate_id']
            if type(cid) is not int or cid < 0:
                raise ValueError('native region id required')
            if cid != self.region:
                self.region = cid
                self.visit_start = message['evidence_id']
            self.latest_visit = deepcopy(row)
            return None
        if not row['topic'].endswith('/candidates'):
            return None
        validate_snapshot(message)
        session, sequence = message['epoch'].rsplit(':',1)
        if not sequence.isdecimal() or int(sequence) <= self.last_epoch_sequence or self.session not in (None,session):
            raise ValueError('native epoch sequence/session mismatch')
        self.last_epoch_sequence = int(sequence)
        self.session = session
        result = dict(epoch=message['epoch'], native_source_frame_keys=list(message['source_frame_keys']),
            snapshot_ref=row['record_ref'], binding_status='UNKNOWN', metric_visit_id=None,
            native_region_id=None, visit_record_ref=None, direction_identity_known=False,
            level_identity_known=False, task_completed=False, graph_mutated=False)
        if self.latest_visit is None:
            return dict(result, reason='NO_PRIOR_NATIVE_VISIT')
        visit = self.latest_visit['payload']
        checks = [(visit['source_frame_keys']==message['source_frame_keys'],'VISIT_SOURCE_WINDOW_MISMATCH'),
            (visit['robot_position_xyz_m']==message['robot_position_xyz_m'],'VISIT_POSITION_MISMATCH'),
            (visit['pose_stamp_ns']==message['stamp_ns'],'VISIT_POSE_STAMP_MISMATCH'),
            (visit['candidate_id']==message['original_route'][0],'VISIT_DEPOT_REGION_MISMATCH')]
        for valid, reason in checks:
            if not valid:
                return dict(result,reason=reason)
        result.update(binding_status='EXACT_NATIVE_METRIC_VISIT',metric_visit_id=self.visit_start,
            native_region_id=self.region,visit_record_ref=self.latest_visit['record_ref'],
            reason='NATIVE_METRIC_VISIT_ONLY_NOT_DIRECTION_OR_PLACE_IDENTITY')
        return result


def classify_local_matches(current, candidates, prior_bindings):
    """Keep every local fit; group metadata only by recorded visit provenance.

    Registration is supplied from a saved diagnostic, not assumed to have
    finished by a live planner deadline. One visit group is not unique place
    identity and cannot transfer completion or create a direction task.
    """
    if current['epoch'] in prior_bindings:
        raise ValueError('current epoch already in history')
    rows=[];groups={};seen=set()
    for candidate in candidates:
        past_epoch=candidate['historical_epoch']
        if candidate['current_epoch']!=current['epoch'] or past_epoch in seen or past_epoch not in prior_bindings:
            raise ValueError('distinct strictly prior candidate bindings required')
        seen.add(past_epoch)
        past=prior_bindings[past_epoch]
        row=dict(historical_epoch=past_epoch,local_fit_accepted=candidate['accepted'],
            kind='REGISTRATION_REJECTED',metric_visit_id=None,task_identity_verified=False,
            source_registration_ref=candidate['record_ref'])
        if candidate['accepted']:
            if current['binding_status']!='EXACT_NATIVE_METRIC_VISIT' or past['binding_status']!='EXACT_NATIVE_METRIC_VISIT':
                row['kind']='LOCAL_FIT_WITHOUT_EXACT_VISIT_BINDING'
            else:
                row['metric_visit_id']=past['metric_visit_id']
                row['kind']='SAME_RECORDED_VISIT' if past['metric_visit_id']==current['metric_visit_id'] else 'OTHER_RECORDED_VISIT'
                groups.setdefault(past['metric_visit_id'],[]).append(past_epoch)
        rows.append(row)
    return dict(epoch=current['epoch'],scan_candidates=rows,local_fit_visit_groups=groups,
        accepted_scan_count=sum(r['local_fit_accepted'] for r in rows),
        accepted_visit_group_count=len(groups),unresolved_identity=True,
        live_registration_available=False,direction_tasks_created=0,task_completed=False,graph_mutated=False)
