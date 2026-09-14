"""Bound V8 transport with per-request lifetimes; no geometry overrides."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import traceback
from teacher_bundle_wire_v1 import decode_bundle,encode_bundle,read_frame,write_frame
from v8_teacher_snapshot_probe_v1 import load_teacher


def handle_payload(payload,diagnose,produce,modules,archive_sha256,check_origins,*,echo=False):
    """Only serialized bytes and a boolean can escape this call's scope."""
    bundle=decode_bundle(payload)
    if echo:
        check_origins()
        return encode_bundle(bundle),True
    try:
        raw=diagnose(bundle)
        targets=produce(bundle,raw)
        check_origins()
        result=dict(status='OK',input_sha256=hashlib.sha256(payload).hexdigest(),archive_sha256=archive_sha256,
                    raw_interfaces=raw,produced_targets=targets,initial_modules=modules)
    except Exception:
        result=dict(status='FAILED',input_sha256=hashlib.sha256(payload).hexdigest(),error=traceback.format_exc())
    response=gzip.compress(json.dumps(result,allow_nan=False).encode(),compresslevel=6,mtime=0)
    return response,result['status']=='OK'


def serve(incoming,outgoing,handler):
    while True:
        payload=read_frame(incoming)
        if payload is None:return 0
        response,ok=handler(payload)
        del payload
        write_frame(outgoing,response)
        del response
        if not ok:return 1


def main(echo=False,memory_bytes=None):
    if memory_bytes is not None:
        if memory_bytes!=3*1024**3:raise ValueError('original3GiB address-space cap required')
        resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    diagnose,produce,modules,manifest=load_teacher()
    archive=Path(__file__).resolve().parents[2]/manifest['archive_path']
    def check():
        for name,module in sys.modules.items():
            if name=='mtare_topo' or name.startswith('mtare_topo.'):
                if not str(getattr(module,'__file__','')).startswith(str(archive)+'/src/'):
                    raise RuntimeError('mixed project dependency: '+name)
    def handler(payload):
        return handle_payload(payload,diagnose,produce,modules,manifest['archive_sha256'],check,echo=echo)
    return serve(sys.stdin.buffer,output,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--echo',action='store_true');p.add_argument('--memory-bytes',type=int)
    args=p.parse_args();raise SystemExit(main(args.echo,args.memory_bytes))
