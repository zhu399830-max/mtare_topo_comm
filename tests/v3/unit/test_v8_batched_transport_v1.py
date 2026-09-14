from test_teacher_bundle_wire_v1 import bundle,same
from v8_batched_teacher_client_v1 import V8BatchedTeacherClient


def test_sealed_batch_override_echo(tmp_path):
    with (tmp_path/'worker.log').open('wb') as log:
        with V8BatchedTeacherClient(log,timeout_s=20,echo=True,memory_bytes=3*1024**3) as client:
            same(bundle(),client.request(bundle()))
            same(bundle(),client.request(bundle()))
        assert client.process.poll()==0
