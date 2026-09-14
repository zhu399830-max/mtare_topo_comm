"""Persistent isolated historical teacher, framed stdin/stdout, no dataset IO."""
import argparse
import gzip
import hashlib
import json
import os
import resource
import sys
from teacher_snapshot_probe_v1 import load_teacher,ZIP,SHA
from teacher_bundle_wire_v1 import decode_bundle,encode_bundle,read_frame,write_frame


def check_module_origins():
    for name,module in sys.modules.items():
        if name=='mtare_topo' or name.startswith('mtare_topo.'):
            if not str(getattr(module,'__file__','')).startswith(str(ZIP)+'/src/'):
                raise RuntimeError('mixed project dependency: '+name)


def main(echo=False, memory_bytes=None):
    if memory_bytes is not None:
        if memory_bytes != 3*1024**3:raise ValueError('fixed 3GiB worker address-space cap')
        resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
    # Keep native library stdout away from the binary protocol as well.
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0);os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    diagnose,produce,modules=load_teacher()
    while True:
        payload=read_frame(sys.stdin.buffer)
        if payload is None:break
        bundle=decode_bundle(payload)
        if echo:
            check_module_origins();write_frame(output,encode_bundle(bundle));continue
        try:
            raw=diagnose(bundle);targets=produce(bundle,raw);check_module_origins()
            result=dict(status='OK',input_sha256=hashlib.sha256(payload).hexdigest(),archive_sha256=SHA,
                raw_interfaces=raw,produced_targets=targets,initial_modules=modules)
        except Exception:
            import traceback
            result=dict(status='FAILED',input_sha256=hashlib.sha256(payload).hexdigest(),error=traceback.format_exc())
        response=gzip.compress(json.dumps(result,allow_nan=False).encode(),compresslevel=6,mtime=0)
        write_frame(output,response)
        if result['status']!='OK':return 1
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--echo',action='store_true')
    parser.add_argument('--memory-bytes',type=int);args=parser.parse_args()
    raise SystemExit(main(args.echo,args.memory_bytes))
