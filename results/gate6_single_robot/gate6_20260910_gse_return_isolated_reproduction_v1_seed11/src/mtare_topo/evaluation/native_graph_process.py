"""Exclusive logs and bounded child process; no run/data authorization here."""
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from .native_candidate_graph import validate_native_graph

CHILD_MEMORY_BYTES=4*1024**3
CHILD_FILE_BYTES=16*1024**2
CHILD_CPU_S=120
CHILD_WALL_S=150


def execute_native(command,packet,*,stdout_path,stderr_path,environment):
    if type(packet) is not bytes or len(packet)>4*1024**2: raise ValueError('bounded packet required')
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(CHILD_MEMORY_BYTES,CHILD_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU,(CHILD_CPU_S,CHILD_CPU_S))
        resource.setrlimit(resource.RLIMIT_FSIZE,(CHILD_FILE_BYTES,CHILD_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    started=time.monotonic()
    # Native glog must not escape the evidence directory through /tmp log files.
    environment=dict(environment, GLOG_logtostderr='1')
    with Path(stdout_path).open('xb') as out, Path(stderr_path).open('xb') as err:
        with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,
                              env=environment,preexec_fn=limits,start_new_session=True) as child:
            try: child.communicate(packet,timeout=CHILD_WALL_S)
            except BaseException:
                if child.poll() is None: os.killpg(child.pid,signal.SIGKILL)
                child.wait(); raise
            if child.returncode!=0: raise RuntimeError(f'native exit {child.returncode}; raw logs preserved')
    raw=Path(stdout_path).read_bytes()
    if len(raw)>CHILD_FILE_BYTES: raise ValueError('native output cap exceeded')
    return validate_native_graph(raw),dict(elapsed_s=time.monotonic()-started,stdout_bytes=len(raw))
