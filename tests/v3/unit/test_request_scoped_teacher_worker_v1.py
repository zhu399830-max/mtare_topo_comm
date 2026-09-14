import gzip
import hashlib
import io
import json
import weakref
import pytest
from test_teacher_bundle_wire_v1 import bundle,same
from teacher_bundle_wire_v1 import encode_bundle,read_frame,write_frame
from request_scoped_teacher_worker_v1 import handle_payload,serve
from request_scoped_teacher_client_v1 import RequestScopedTeacherClient


class WeakDict(dict):
    pass


@pytest.mark.parametrize('fail',[False,True])
def test_request_objects_released_on_success_and_handled_failure(fail):
    refs=[]
    def diagnose(b):
        assert all(r() is None for r in refs), 'previous request survived into next calculation'
        refs.append(weakref.ref(b['student']['ranges_m']))
        raw=WeakDict(a=[1,2,3]);refs.append(weakref.ref(raw));return raw
    def produce(b,raw):
        target=WeakDict(b='same');refs.append(weakref.ref(target))
        if fail:raise ValueError('synthetic failure')
        return target
    payload=encode_bundle(bundle())
    for _ in range(2):
        response,ok=handle_payload(payload,diagnose,produce,{},'archive',lambda:None)
        assert ok is not fail
        assert all(r() is None for r in refs), 'live references must not await gc'
        decoded=json.loads(gzip.decompress(response))
        if not fail:
            expected=dict(status='OK',input_sha256=hashlib.sha256(payload).hexdigest(),archive_sha256='archive',
                raw_interfaces={'a':[1,2,3]},produced_targets={'b':'same'},initial_modules={})
            assert response==gzip.compress(json.dumps(expected,allow_nan=False).encode(),compresslevel=6,mtime=0)
        else:assert 'synthetic failure' in decoded['error']


def test_failure_response_stops_before_second_request():
    incoming=io.BytesIO();write_frame(incoming,b'first');write_frame(incoming,b'second');incoming.seek(0)
    outgoing=io.BytesIO();calls=[]
    def handler(payload):calls.append(payload);return b'failed',False
    assert serve(incoming,outgoing,handler)==1
    assert calls==[b'first'] and read_frame(incoming)==b'second'


def test_successful_serve_releases_objects_before_next_request():
    incoming=io.BytesIO()
    payload=encode_bundle(bundle())
    for _ in range(3):write_frame(incoming,payload)
    incoming.seek(0)
    outgoing=io.BytesIO();refs=[];calls=[]
    def diagnose(b):
        assert all(ref() is None for ref in refs)
        refs.append(weakref.ref(b['student']['ranges_m']))
        raw=WeakDict(value=7);refs.append(weakref.ref(raw));calls.append(1)
        return raw
    def handler(data):
        return handle_payload(data,diagnose,lambda b,r:{'value':r['value']},{},'archive',lambda:None)
    assert serve(incoming,outgoing,handler)==0
    assert len(calls)==3 and all(ref() is None for ref in refs)
    outgoing.seek(0)
    responses=[read_frame(outgoing) for _ in range(3)]
    assert responses[0]==responses[1]==responses[2]
    assert read_frame(outgoing) is None


def test_diagnostic_failure_does_not_retain_bundle():
    refs=[]
    def diagnose(b):
        refs.append(weakref.ref(b['student']['ranges_m']))
        raise RuntimeError('diagnostic failed')
    def forbidden(*args):raise AssertionError('producer must not run')
    response,ok=handle_payload(encode_bundle(bundle()),diagnose,forbidden,{},'archive',lambda:None)
    assert not ok and all(ref() is None for ref in refs)
    assert 'RuntimeError: diagnostic failed' in json.loads(gzip.decompress(response))['error']


def test_persistent_scoped_v8_process_echo(tmp_path):
    with (tmp_path/'worker.log').open('wb') as log:
        with RequestScopedTeacherClient(log,timeout_s=20,echo=True,memory_bytes=3*1024**3) as client:
            pid=client.process.pid
            same(bundle(),client.request(bundle()));same(bundle(),client.request(bundle()))
            assert client.process.pid==pid
        assert client.process.poll()==0
