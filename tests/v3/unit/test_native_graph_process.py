import os
import subprocess
import sys

import pytest

from mtare_topo.evaluation import native_graph_process as runner


def invoke(tmp_path, script):
    return runner.execute_native(
        [sys.executable, '-c', script], b'',
        stdout_path=tmp_path/'stdout.json', stderr_path=tmp_path/'stderr.log',
        environment=os.environ)


def test_failure_preserves_logs(tmp_path):
    with pytest.raises(RuntimeError, match='native exit 7'):
        invoke(tmp_path, "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)")
    assert 'diagnostic' in (tmp_path/'stderr.log').read_text()


def test_existing_output_never_overwritten(tmp_path):
    path=tmp_path/'stdout.json'
    path.write_bytes(b'prior evidence')
    with pytest.raises(FileExistsError):
        invoke(tmp_path, "raise AssertionError('must not execute')")
    assert path.read_bytes()==b'prior evidence'


def test_timeout_stops_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'CHILD_WALL_S', .1)
    with pytest.raises(subprocess.TimeoutExpired):
        invoke(tmp_path, 'import time; time.sleep(10)')


def test_native_child_limits_and_log_routing(tmp_path):
    script='''import os, resource, sys
assert resource.getrlimit(resource.RLIMIT_AS)[0] == 4*1024**3
assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 16*1024**2
assert resource.getrlimit(resource.RLIMIT_CPU)[0] == 120
assert resource.getrlimit(resource.RLIMIT_CORE)[0] == 0
assert os.environ['GLOG_logtostderr'] == '1'
print('limits verified', file=sys.stderr)
sys.exit(7)
'''
    with pytest.raises(RuntimeError, match='native exit 7'):
        invoke(tmp_path, script)
    assert 'limits verified' in (tmp_path/'stderr.log').read_text()
