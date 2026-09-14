"""Authenticate live token notifications, then perform nonblocking exact lookup.

Heavy file reads occur at ingest, outside the native advice callback. A cache
hit certifies bytes and availability only: it does not confirm a place, task,
physical traversal or permission to reorder. No old epoch is rebased.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import zipfile
import numpy as np

from mtare_topo.integration.native_route_advice import validate_snapshot
from mtare_topo.integration.structural_token_retrieval import FrameBinding,LocalTokenRecord

CHECKPOINTS={
    'encoder':'8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb',
    'relation':'ac1badf396804debb2b7daa505375bb0885a9d79bd918832a9f66e16697f6a76'}


def _ns(value):
    if type(value) is not int or value<0:raise ValueError('EXACT_NONNEGATIVE_MONOTONIC_NS')
    return value


class LiveStructuralTokenCache:
    def __init__(self,root,*,artifact_prefix,input_prefix,segment,max_entries,clock=time.monotonic_ns):
        self.root=Path(root).resolve();self.prefixes={}
        for key,value in [('artifact',artifact_prefix),('input',input_prefix)]:
            path=Path(value)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('EXACT_NONROOT_PROJECT_RELATIVE_CACHE_SCOPE')
            self.prefixes[key]=path
        if not isinstance(segment,str) or not segment or type(max_entries) is not int or not 1<=max_entries<=20000:
            raise ValueError('EXPLICIT_CACHE_SEGMENT_AND_BOUND')
        self.segment,self.max_entries,self.clock=segment,max_entries,clock
        self._lock=threading.RLock();self._entries={};self._requests=set();self._session=None

    def _read(self,reference,scope,limit=16*1024**2):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
            raise ValueError('EXACT_FILE_SHA_REFERENCE')
        path=Path(reference['path'])
        if path.is_absolute() or '..' in path.parts or self.prefixes[scope] not in path.parents:
            raise ValueError('CACHE_SOURCE_OUTSIDE_ALLOWED_SCOPE')
        current=self.root
        for part in path.parts:
            current=current/part
            if current.is_symlink():raise ValueError('CACHE_SOURCE_SYMLINK')
        if current.stat().st_size>limit:raise ValueError('CACHE_FILE_BYTE_CAP')
        raw=current.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('CACHE_PAYLOAD_DRIFT')
        return raw

    @staticmethod
    def _arrays(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(i.file_size for i in archive.infolist())>16*1024**2:raise ValueError('CACHE_ARRAY_EXPANSION_CAP')
        with np.load(io.BytesIO(raw),allow_pickle=False) as data:
            return {k:data[k] for k in data.files}

    def ingest(self,event,*,received_monotonic_ns):
        """Caller authenticates the live producer; no task inference here."""
        event=deepcopy(event)
        if event.get('schema_version')!='live_structural_token_ready_v1' or event.get('segment')!=self.segment:
            raise ValueError('LIVE_CACHE_SCHEMA_OR_SEGMENT')
        client=event['client_result'];response=client['response'];binding=event['binding']
        if (response.get('schema_version')!='streaming_structural_token_response_v1'
                or response.get('status')!='PROCESSED' or response.get('sequence_id')!=self.segment
                or response.get('preprocessing_version')!='full_relative_se3_v1'
                or response.get('checkpoint_sha256')!=CHECKPOINTS
                or response.get('training_steps')!=0 or response.get('model_forward') is not True
                or response.get('control_published') is not False
                or response.get('native_task_identity_verified') is not False):
            raise ValueError('LIVE_CACHE_MODEL_SCOPE')
        times=[client['client_started_monotonic_ns'],response['server_received_monotonic_ns'],
               response['server_completed_monotonic_ns'],client['client_received_monotonic_ns'],received_monotonic_ns]
        if any(_ns(a)>_ns(b) for a,b in zip(times,times[1:])):raise ValueError('LIVE_CACHE_TIMELINE')
        ref=dict(path=response['project_relative_output_record'],sha256=response['output_record_sha256'])
        row=json.loads(self._read(ref,'artifact',65536))
        if (row.get('status')!='PROCESSED' or row.get('coordinate_frame')!='current_sensor_m'
                or row.get('model_forward') is not True or row.get('preprocessing_version')!='full_relative_se3_v1'
                or row.get('source')!=response.get('source') or row['window_id']!=response['window_id']):
            raise ValueError('LIVE_CACHE_RECORD_BINDING')
        source=row['source'];frames=source['frames'];keys=tuple(binding['registered_source_keys'])
        raw_keys=tuple(frame['source_key'] for frame in frames)
        if (len(frames)!=5 or len(keys)!=5 or len(set(keys))!=5
                or any(not isinstance(k,str) or not k.startswith('registered_scan:') or not k.split(':')[1].isdecimal() for k in keys)
                or binding.get('status')!='EXPLICIT_SOURCE_BOUND' or binding.get('source_transport_bound') is not True
                or binding.get('physical_pose_verified') is not False or tuple(binding['raw_source_keys'])!=raw_keys
                or source['sequence_id']!=self.segment or row['stamp_ns']!=frames[-1]['stamp_ns']):
            raise ValueError('LIVE_CACHE_FIVE_FRAME_BINDING')
        if any(int(a.split(':')[1])>=int(b.split(':')[1]) for a,b in zip(keys,keys[1:])):
            raise ValueError('LIVE_CACHE_REGISTERED_ORDER')
        token_ref=response['project_relative_token_ref']
        if (Path(token_ref['path']).parent!=Path(ref['path']).parent
                or Path(token_ref['path']).name!=row['token_ref']['path']
                or token_ref['sha256']!=row['token_ref']['sha256']):
            raise ValueError('LIVE_CACHE_TOKEN_REFERENCE')
        inputs=self._arrays(self._read(source['input'],'input'))
        tokens=self._arrays(self._read(token_ref,'artifact'))
        raw=inputs['raw_pointcloud_bytes'];poses=inputs['world_from_sensor']
        if (raw.dtype!=np.uint8 or raw.shape!=(5,123200) or poses.dtype!=np.float64 or poses.shape!=(5,4,4)
                or not np.isfinite(poses).all() or not np.array_equal(tokens['world_from_sensor'],poses)
                or not np.array_equal(tokens['source_stamp_ns'],[f['stamp_ns'] for f in frames])):
            raise ValueError('LIVE_CACHE_INPUT_TOKEN_POSE_BINDING')
        frame_bindings=tuple(FrameBinding(frame['source_key'],frame['stamp_ns'],
            hashlib.sha256(raw[i].tobytes()).hexdigest()) for i,frame in enumerate(frames))
        record=LocalTokenRecord(response['request_id'],self.segment,frames[-1]['frame_order'],raw_keys,
            tokens['endpoints_current_sensor_m'],tokens['ray_tokens'],frame_bindings,
            CHECKPOINTS['encoder'],CHECKPOINTS['relation'])
        available=_ns(self.clock())
        if available<received_monotonic_ns:raise ValueError('LIVE_CACHE_CLOCK_REGRESSION')
        details=dict(record_ref=ref,token_ref=token_ref,request_id=response['request_id'],
            raw_source_frame_keys=list(raw_keys),registered_source_frame_keys=list(keys),
            client_received_monotonic_ns=times[-2],cache_ready_monotonic_ns=available,
            physical_pose_verified=False,task_identity_verified=False)
        with self._lock:
            if keys in self._entries or response['request_id'] in self._requests:
                raise ValueError('DUPLICATE_LIVE_CACHE_WINDOW')
            if len(self._entries)>=self.max_entries:raise ValueError('LIVE_CACHE_CAP_EXCEEDED')
            self._entries[keys]=(record,details);self._requests.add(response['request_id'])
        return deepcopy(details)

    def lookup(self,snapshot,*,decision_monotonic_ns):
        validate_snapshot(snapshot);_ns(decision_monotonic_ns)
        session,sep,seq=snapshot['epoch'].rpartition(':')
        if not sep or not session or not seq.isdecimal():raise ValueError('NATIVE_SESSION_REQUIRED')
        keys=tuple(snapshot['source_frame_keys'])
        with self._lock:
            if self._session is not None and self._session!=session:raise ValueError('NATIVE_RESTART_NEW_CACHE_REQUIRED')
            self._session=session
            entry=self._entries.get(keys)
            if entry is None:return dict(ready=False,reason='EXACT_FIVE_FRAME_TOKEN_MISSING')
            record,details=entry
            if snapshot['stamp_ns']!=int(keys[-1].split(':')[1]):raise ValueError('NATIVE_STAMP_SOURCE_MISMATCH')
            if details['cache_ready_monotonic_ns']>decision_monotonic_ns:
                return dict(ready=False,reason='TOKEN_NOT_READY_AT_THIS_DECISION')
            return dict(ready=True,reason='AUTHENTICATED_EXACT_WINDOW_READY',epoch=snapshot['epoch'],
                **deepcopy(details),task_association_confirmed=False,route_change_authorized=False)

    def record(self,registered_source_keys):
        """Immutable token record for downstream retrieval worker, not advice."""
        with self._lock:return self._entries[tuple(registered_source_keys)][0]
