"""Persistent synthetic echo through the pinned V8 process; no labels."""
from test_teacher_bundle_wire_v1 import bundle,same
from v8_historical_teacher_client_v1 import V8HistoricalTeacherClient
from historical_teacher_client_v1 import HistoricalTeacherClient


def test_v8_two_requests_and_original_client_binding_preserved(tmp_path):
    original=HistoricalTeacherClient.archive_sha256
    with (tmp_path/'v8.log').open('wb') as log:
        with V8HistoricalTeacherClient(log,timeout_s=20,echo=True,memory_bytes=3*1024**3) as client:
            pid=client.process.pid
            same(bundle(),client.request(bundle()))
            same(bundle(),client.request(bundle()))
            assert client.process.pid==pid
        assert client.process.poll()==0
    assert HistoricalTeacherClient.archive_sha256==original
    assert original!=V8HistoricalTeacherClient.archive_sha256
