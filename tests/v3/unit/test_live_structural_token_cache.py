from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pytest

from mtare_topo.integration.live_structural_token_cache import LiveStructuralTokenCache
from mtare_topo.integration.streaming_structural_tokens import StreamingStructuralTokens
from tests.v3.unit.test_native_live_window_writer import parts,write_window
from tests.v3.unit.test_native_region_task_registry import native_snapshot


@dataclass
class Output:
    ray_ids: np.ndarray
    ray_tokens: np.ndarray
    endpoints_current_sensor_m: np.ndarray
    coordinate_frame: str='current_sensor_m'


def fixture(root):
    folder=root/'inputs';folder.mkdir()
    rows,binding,raw,poses=parts()
    binding.update(registered_source_keys=[f'registered_scan:{n}' for n in (600,700,800,900,1000)],
        source_transport_bound=True)
    request=write_window(root,folder,0,rows,binding,segment='live')
    class Extractor:
        def extract(self,*a,**kw):return Output(np.arange(2),np.ones((2,128),np.float32),
                                               np.array([[2,0,0],[4,0,0]],np.float32))
    ticks=iter([10,20])
    worker=StreamingStructuralTokens(root,root/'tokens',device='cpu',max_requests=1,input_root=folder,
        loader=lambda *a,**kw:Extractor(),clock=lambda:next(ticks))
    response=worker.process(request)
    event=dict(schema_version='live_structural_token_ready_v1',segment='live',binding=binding,
        client_result=dict(response=response,client_started_monotonic_ns=5,client_received_monotonic_ns=25,
            usable_for_native_advice=False))
    snapshot=native_snapshot();snapshot['source_frame_keys']=binding['registered_source_keys']
    cache=LiveStructuralTokenCache(root,artifact_prefix='tokens',input_prefix='inputs',segment='live',max_entries=2,
        clock=lambda:40)
    return cache,event,snapshot


def test_authenticate_actual_worker_files_before_nonblocking_exact_lookup(tmp_path):
    cache,event,snapshot=fixture(tmp_path)
    assert not cache.lookup(snapshot,decision_monotonic_ns=35)['ready']
    detail=cache.ingest(event,received_monotonic_ns=30)
    assert detail['cache_ready_monotonic_ns']==40
    before=cache.lookup(snapshot,decision_monotonic_ns=39)
    assert not before['ready'] and before['reason']=='TOKEN_NOT_READY_AT_THIS_DECISION'
    after=cache.lookup(snapshot,decision_monotonic_ns=40)
    assert after['ready'] and not after['route_change_authorized'] and not after['task_association_confirmed']
    record=cache.record(snapshot['source_frame_keys'])
    assert record.tokens.shape==(2,128) and record.tokens.flags.writeable is False
    assert record.source_refs==tuple(event['binding']['raw_source_keys'])


@pytest.mark.parametrize('fault',['segment','source','weights','timeline','scope','digest','pose','status'])
def test_corrupt_or_unrelated_notification_cannot_become_ready(tmp_path,fault):
    cache,event,snapshot=fixture(tmp_path)
    response=event['client_result']['response']
    if fault=='segment':event['segment']='another'
    if fault=='source':event['binding']['raw_source_keys'][0]='not-the-original'
    if fault=='weights':response['checkpoint_sha256']['relation']='0'*64
    if fault=='timeline':event['client_result']['client_received_monotonic_ns']=19
    if fault=='scope':response['project_relative_output_record']='inputs/../record.json'
    if fault=='digest':response['output_record_sha256']='0'*64
    if fault=='pose':event['binding']['physical_pose_verified']=True
    if fault=='status':response['status']='REJECTED_RECORD'
    with pytest.raises(ValueError):cache.ingest(event,received_monotonic_ns=30)
    assert not cache.lookup(snapshot,decision_monotonic_ns=100)['ready']


def test_no_nearest_source_fallback_and_restart_cannot_reuse_cache(tmp_path):
    cache,event,snapshot=fixture(tmp_path);cache.ingest(event,received_monotonic_ns=30)
    other=deepcopy(snapshot);other['source_frame_keys'][0]='registered_scan:599'
    assert not cache.lookup(other,decision_monotonic_ns=100)['ready']
    other['epoch']='new-session:1'
    with pytest.raises(ValueError,match='RESTART'):cache.lookup(other,decision_monotonic_ns=100)


def test_duplicate_ready_event_does_not_overwrite_availability(tmp_path):
    cache,event,snapshot=fixture(tmp_path);cache.ingest(event,received_monotonic_ns=30)
    with pytest.raises(ValueError,match='DUPLICATE'):cache.ingest(event,received_monotonic_ns=30)
    assert cache.lookup(snapshot,decision_monotonic_ns=100)['cache_ready_monotonic_ns']==40
