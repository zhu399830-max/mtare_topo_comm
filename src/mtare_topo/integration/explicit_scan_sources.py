"""Source identity from authenticated same-callback evidence, not clock guesses.

Caller authenticates the pinned instrumented publisher, source/bag hashes and
topic headers. Fields are checked against actual recorded raw/registered frames.
Source identity does not certify scan registration or actual link pose accuracy.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ObservedScan:
    seq: int
    stamp_ns: int
    receipt_ns: int

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.seq,self.stamp_ns,self.receipt_ns)):
            raise ValueError('NONNEGATIVE_SCAN_HEADERS_REQUIRED')


class ExplicitScanSources:
    def __init__(self, *, segment_id, source_artifact_ref):
        if not segment_id or not source_artifact_ref:
            raise ValueError('EXPLICIT_SEGMENT_AND_AUTHENTICATED_PRODUCER_REQUIRED')
        self.segment_id=segment_id; self.source_artifact_ref=source_artifact_ref
        self.session=None; self.by_registered={}; self.used_raw=set(); self.last_output_seq=-1

    def add(self, evidence, *, raw: ObservedScan, registered: ObservedScan,
            evidence_receipt_ns: int, evidence_ref: str):
        if not isinstance(evidence,Mapping) or evidence.get('schema_version')!='native_scan_source_v1':
            raise ValueError('EXPLICIT_SOURCE_SCHEMA_REQUIRED')
        for key in ('source_seq','source_stamp_ns','registered_seq','registered_stamp_ns'):
            if type(evidence.get(key)) is not int or evidence[key]<0:
                raise ValueError('EXPLICIT_SOURCE_HEADER_REQUIRED')
        session=evidence.get('session_id')
        if not isinstance(session,str) or not session or (self.session is not None and self.session!=session):
            raise ValueError('SOURCE_SESSION_CHANGED_NEW_SEGMENT_REQUIRED')
        if (evidence.get('raw_topic')!='/velodyne_points' or evidence.get('registered_topic')!='/registered_scan'
                or evidence.get('source_frame')!='velodyne' or evidence.get('registered_frame')!='map'
                or evidence.get('raw_frame_matches') is not True
                or evidence.get('physical_pose_verified') is not False):
            raise ValueError('SOURCE_TOPIC_FRAME_OR_CLAIM_MISMATCH')
        if (raw.seq,raw.stamp_ns,registered.seq,registered.stamp_ns)!=(
            evidence['source_seq'],evidence['source_stamp_ns'],evidence['registered_seq'],evidence['registered_stamp_ns']):
            raise ValueError('RECORDED_HEADERS_DO_NOT_MATCH_CALLBACK_EVIDENCE')
        if (registered.seq<=self.last_output_seq or registered.stamp_ns in self.by_registered
                or raw.stamp_ns in self.used_raw):
            raise ValueError('DUPLICATE_OR_REORDERED_SOURCE_EVIDENCE')
        if type(evidence_receipt_ns) is not int or evidence_receipt_ns<0 or not evidence_ref:
            raise ValueError('SOURCE_RECEIPT_AND_REFERENCE_REQUIRED')
        # Receipt ordering between separate ROS subscribers is NOT a premise.
        self.by_registered[registered.stamp_ns]=(raw,registered,evidence_receipt_ns,evidence_ref)
        self.used_raw.add(raw.stamp_ns);self.last_output_seq=registered.seq;self.session=session

    def window(self, registered_keys, *, segment_id, snapshot_stamp_ns, snapshot_receipt_ns):
        if segment_id!=self.segment_id:
            raise ValueError('SEGMENT_MISMATCH')
        keys=tuple(registered_keys)
        if len(keys)!=5 or len(set(keys))!=5:
            raise ValueError('FIVE_DISTINCT_REGISTERED_KEYS_REQUIRED')
        rows=[]
        for key in keys:
            if not isinstance(key,str) or not key.startswith('registered_scan:') or not key.split(':',1)[1].isdecimal():
                raise ValueError('REGISTERED_KEY_REQUIRED')
            stamp=int(key.split(':',1)[1])
            if stamp not in self.by_registered:
                return dict(status='UNBOUND',reason='MISSING_EXPLICIT_SOURCE',usable_for_native_advice=False)
            rows.append(self.by_registered[stamp])
        if (any(a[1].seq>=b[1].seq or a[0].stamp_ns>=b[0].stamp_ns for a,b in zip(rows,rows[1:]))
                or rows[-1][1].stamp_ns!=snapshot_stamp_ns):
            raise ValueError('NONCAUSAL_NATIVE_SOURCE_ORDER')
        # Skipped raw inputs are explicit; never invent the skipped scan/window.
        ready=all(max(r.receipt_ns,g.receipt_ns,e)<=snapshot_receipt_ns for r,g,e,_ in rows)
        return dict(status='EXPLICIT_SOURCE_BOUND',raw_source_keys=[f'/velodyne_points:{r.stamp_ns}' for r,_,_,_ in rows],
            registered_source_keys=list(keys),source_evidence_refs=[e for _,_,_,e in rows],
            source_transport_bound=True,available_at_native_snapshot=ready,
            physical_pose_verified=False,registration_verified=False,
            token_ready=False,task_identity_verified=False,usable_for_native_advice=False)
