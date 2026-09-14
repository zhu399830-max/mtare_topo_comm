import io
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/v3'))
from teacher_bundle_wire_v1 import ARRAYS,encode_bundle,decode_bundle,read_frame,write_frame


def bundle():
    b=dict(source=dict(task='S01_flat_tree_small_C01__ellipse',frame_rows=[0,1,2,3,4],source_sequence_id=0),
        construction_teacher_only={'synthetic':True},codebook_teacher_only={'synthetic':True},student={},sensor_teacher_only={})
    for k,(shape,dtype) in ARRAYS.items():
        section,name=k.split('/');b[section][name]=np.zeros(shape,dtype=dtype)
    b['student']['source_sequence_ids']=np.int64(0)
    return b


def same(a,b):
    for k in ('source','construction_teacher_only','codebook_teacher_only'):assert a[k]==b[k]
    for k in ARRAYS:
        section,name=k.split('/');np.testing.assert_array_equal(a[section][name],b[section][name])
        assert a[section][name].dtype==b[section][name].dtype


def test_roundtrip_original_arrays_and_metadata():
    b=bundle();same(b,decode_bundle(encode_bundle(b)))


def test_short_pipe_reads_and_writes():
    class ShortStream(io.BytesIO):
        def write(self,value):return super().write(value[:3])
        def read(self,size=-1):return super().read(min(size,3))
    stream=ShortStream();write_frame(stream,b'abcdefghijk');stream.seek(0)
    assert read_frame(stream)==b'abcdefghijk'
    assert read_frame(stream) is None


def test_frame_truncation_corruption_and_extra_fields_rejected():
    b=bundle();raw=encode_bundle(b);f=io.BytesIO();write_frame(f,raw);wire=f.getvalue()
    with pytest.raises(EOFError):read_frame(io.BytesIO(wire[:-1]))
    with pytest.raises(ValueError,match='checksum'):read_frame(io.BytesIO(wire[:-1]+bytes([wire[-1]^1])))
    b['student']['label']=np.ones(1)
    with pytest.raises(ValueError,match='closed'):encode_bundle(b)


def test_two_requests_same_historical_process_echo_no_teacher_execution():
    b=bundle();raw=encode_bundle(b);f=io.BytesIO();write_frame(f,raw);write_frame(f,raw)
    python='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'
    p=subprocess.run([python,str(ROOT/'tools/v3/historical_teacher_worker_v1.py'),'--echo'],input=f.getvalue(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
    assert p.returncode==0,p.stderr.decode()
    stream=io.BytesIO(p.stdout)
    same(b,decode_bundle(read_frame(stream)));same(b,decode_bundle(read_frame(stream)))
    assert read_frame(stream) is None


def test_persistent_client_echo_and_closed_rejection(tmp_path):
    from historical_teacher_client_v1 import HistoricalTeacherClient
    with (tmp_path/'worker.log').open('wb') as log:
        with HistoricalTeacherClient(log,timeout_s=20,echo=True) as client:
            pid=client.process.pid
            same(bundle(),client.request(bundle()))
            same(bundle(),client.request(bundle()))
            assert client.process.pid==pid
        assert client.process.poll()==0
        with pytest.raises(RuntimeError,match='closed'):
            client.request(bundle())


def test_client_deadline_terminates_without_retry(tmp_path,monkeypatch):
    import historical_teacher_client_v1 as module
    original=module.subprocess.Popen
    def sleeper(*args,**kwargs):
        return original([sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',sleeper)
    with (tmp_path/'worker.log').open('wb') as log:
        client=module.HistoricalTeacherClient(log,timeout_s=0.05,echo=True)
        with pytest.raises(TimeoutError):client.request(bundle())
        assert client.closed and client.process.poll() is not None


def test_client_bad_frame_terminates(tmp_path,monkeypatch):
    import historical_teacher_client_v1 as module
    original=module.subprocess.Popen
    def corrupt(*args,**kwargs):
        return original([sys.executable,'-c',
            "import sys,time;sys.stdout.buffer.write(bytes(8));sys.stdout.buffer.flush();time.sleep(30)"],**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',corrupt)
    with (tmp_path/'worker.log').open('wb') as log:
        client=module.HistoricalTeacherClient(log,timeout_s=5,echo=True)
        with pytest.raises(ValueError,match='cap'):client.request(bundle())
        assert client.closed and client.process.poll() is not None
