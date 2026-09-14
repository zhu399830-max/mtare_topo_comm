"""Sequential bounded client for the frozen teacher; no dataset access."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time

from teacher_bundle_wire_v1 import MAX_PACKET, encode_bundle, decode_bundle, read_frame, write_frame
from teacher_snapshot_probe_v1 import SHA

PYTHON = '/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'
WORKER = Path(__file__).with_name('historical_teacher_worker_v1.py')
MAX_RESPONSE_JSON = 256 * 1024**2


class HistoricalTeacherClient:
    """One request at a time; any error poisons and terminates this worker.

    stderr is an already-open run log, never an undrained pipe. The caller
    owns total run CPU/RAM/output budgets; this adapter bounds request IO.
    """

    worker_path = WORKER
    archive_sha256 = SHA

    def __init__(self, log, *, timeout_s=120, echo=False, memory_bytes=None):
        if not 0 < timeout_s <= 10800:
            raise ValueError('finite bounded request timeout required')
        self.timeout_s = timeout_s
        self.echo = echo
        self.closed = False
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0', PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH', None)
        limits=[]
        if memory_bytes is not None:
            if memory_bytes != 3*1024**3:raise ValueError('fixed worker memory budget')
            limits=['--memory-bytes',str(memory_bytes)]
        self.process = subprocess.Popen(
            [PYTHON, str(self.worker_path)] + (['--echo'] if echo else []) + limits,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            env=env, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def _io(self, stream, n, deadline, data=None):
        parts = []
        offset = 0
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_WRITE if data is not None else selectors.EVENT_READ)
            while offset < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('historical teacher request deadline')
                try:
                    if data is not None:
                        count = os.write(stream.fileno(), memoryview(data)[offset:offset+65536])
                    else:
                        chunk = os.read(stream.fileno(), min(n-offset, 65536))
                        if not chunk:
                            raise EOFError('historical teacher exited/truncated response')
                        parts.append(chunk)
                        count = len(chunk)
                except BlockingIOError:
                    continue
                if count == 0:
                    raise EOFError('historical teacher closed pipe')
                offset += count
        return b''.join(parts)

    def request(self, bundle):
        if self.closed:
            raise RuntimeError('teacher client closed; no automatic restart')
        try:
            payload = encode_bundle(bundle)
            stream = io.BytesIO()
            write_frame(stream, payload)
            packet = stream.getvalue()
            deadline = time.monotonic() + self.timeout_s
            self._io(self.process.stdin, len(packet), deadline, packet)
            header = self._io(self.process.stdout, 8, deadline)
            size = struct.unpack('!Q', header)[0]
            if not 0 < size <= MAX_PACKET:
                raise ValueError('response frame cap')
            body = self._io(self.process.stdout, 32+size, deadline)
            response = read_frame(io.BytesIO(header+body))
            if self.echo:
                return decode_bundle(response)
            with gzip.GzipFile(fileobj=io.BytesIO(response)) as zipped:
                decoded = zipped.read(MAX_RESPONSE_JSON+1)
            if len(decoded) > MAX_RESPONSE_JSON:
                raise ValueError('expanded response cap')
            result = json.loads(decoded)
            if result.get('status') != 'OK':
                raise RuntimeError('historical teacher failed: '+str(result.get('error')))
            if result.get('input_sha256') != hashlib.sha256(payload).hexdigest() or result.get('archive_sha256') != self.archive_sha256:
                raise ValueError('teacher response provenance mismatch')
            if not {'raw_interfaces', 'produced_targets', 'initial_modules'} <= result.keys():
                raise ValueError('teacher response incomplete')
            return result
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, *, abort=False):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        if abort and self.process.poll() is None:
            self.process.terminate()
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(timeout=5)
        finally:
            self.process.stdout.close()
        if code != 0 and not abort:
            raise RuntimeError('historical teacher exit code: '+str(code))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(abort=kind is not None)
