"""Frozen token worker with explicit GPU allocator bound and peak evidence."""
from _bootstrap import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import resource
import time
import socket
import torch
import numpy as np
from mtare_topo.integration.frozen_structural_token_worker import run_worker


def serve_stream(args):
    from mtare_topo.integration.streaming_structural_tokens import StreamingStructuralTokens
    from mtare_topo.integration.structural_token_transport import receive_json,send_json
    if (args.preprocessing != 'full_relative_se3_v1' or args.manifest or args.manifest_sha256
            or args.max_requests is None or args.max_wall_seconds is None
            or args.stream_input_root is None
            or not 0 < args.max_wall_seconds <= 43200):
        raise ValueError('EXPLICIT_STREAM_SE3_POPULATION_AND_WALL_BOUND_REQUIRED')
    path=args.stream_socket.resolve()
    if path.exists() or path.is_symlink(): raise ValueError('NEW_STREAM_SOCKET_REQUIRED')
    producer=StreamingStructuralTokens(PROJECT_ROOT,args.output,device='cuda',max_requests=args.max_requests,
        input_root=args.stream_input_root)
    producer.preload()  # Once; ready never means model inference has already occurred.
    deadline=time.monotonic()+args.max_wall_seconds
    counts=dict(received=0,responses_delivered=0,client_disconnected=0,rejected_requests=0,model_forwards=0,processed_windows=0)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(path)); server.listen(2); server.settimeout(.5)
        (args.output/'ready.json').write_text(json.dumps(dict(schema_version='structural_token_stream_ready_v1',
            socket=str(path),weights_loaded=True,model_forwards=0,training_steps=0,
            preprocessing_version=args.preprocessing))+'\n')
        with (args.output/'stream.jsonl').open('x') as log:
            while time.monotonic()<deadline and counts['received']<args.max_requests and not producer.failed:
                try: connection,_=server.accept()
                except socket.timeout: continue
                with connection:
                    connection.settimeout(2.)
                    request=None; counts['received']+=1
                    try:
                        request=receive_json(connection)
                        response=producer.process(request)
                        counts['model_forwards']+=int(response['model_forward'])
                        counts['processed_windows']+=int(response['status']=='PROCESSED')
                    except Exception as exc:
                        counts['rejected_requests']+=1
                        response=dict(schema_version='streaming_structural_token_error_v1',
                            request_id=request.get('request_id') if isinstance(request,dict) else None,
                            reason=type(exc).__name__+': '+str(exc),training_steps=0,control_published=False)
                    delivered=False
                    try: send_json(connection,response); delivered=True; counts['responses_delivered']+=1
                    except (OSError,ValueError): counts['client_disconnected']+=1
                    log.write(json.dumps(dict(response=response,delivered=delivered),allow_nan=False)+'\n');log.flush()
    # The closed socket inode is retained as evidence; never remove an arbitrary
    # pre-existing socket or reconnect/restart a stopped producer automatically.
    summary=dict(schema_version='structural_token_stream_summary_v1',counts=counts,
        training_steps=0,error='FATAL_SOURCE_OR_EXECUTION_ERROR' if producer.failed else None,
        no_valid_transfer=counts['processed_windows']==0,control_published=False,
        responses_delivered_semantics='socket send completed; client receipt is recorded separately',
        stop_reason='FATAL' if producer.failed else ('REQUEST_LIMIT' if counts['received']>=args.max_requests else 'WALL_LIMIT'))
    (args.output/'stream_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    import zarr
    p=argparse.ArgumentParser();p.add_argument('--manifest')
    p.add_argument('--manifest-sha256');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resource-output',type=Path,required=True)
    p.add_argument('--preprocessing', choices=('yaw_only_v1','full_relative_se3_v1'), default='yaw_only_v1')
    p.add_argument('--stream-socket',type=Path)
    p.add_argument('--stream-input-root',type=Path)
    p.add_argument('--max-requests',type=int)
    p.add_argument('--max-wall-seconds',type=float)
    a=p.parse_args()
    if a.resource_output.exists() or a.output.exists():raise ValueError('new evidence paths required')
    if not a.stream_socket and (not a.manifest or not a.manifest_sha256):raise ValueError('BOUND_MANIFEST_REQUIRED')
    if str(torch.__version__)!='2.9.0+cu129':raise ValueError('frozen Torch environment drift')
    if np.__version__!='2.1.3' or zarr.__version__!='2.18.7':raise ValueError('frozen NumPy/Zarr environment drift')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    maximum=28*1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.,maximum/torch.cuda.get_device_properties(0).total_memory))
    start=time.monotonic();summary=None
    try:
        worker=run_worker
        if a.preprocessing=='full_relative_se3_v1':
            from mtare_topo.integration.frozen_structural_se3_worker import run_worker as worker
        summary=(serve_stream(a) if a.stream_socket else
                 worker(PROJECT_ROOT,dict(path=a.manifest,sha256=a.manifest_sha256),a.output,device='cuda'))
    finally:
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        reserved=torch.cuda.max_memory_reserved()
        record=dict(torch=str(torch.__version__),numpy=np.__version__,zarr=zarr.__version__,gpu=torch.cuda.get_device_name(0),
            gpu_allocator_limit_bytes=maximum,peak_gpu_reserved_bytes=reserved,peak_host_rss_bytes=rss,
            limits_passed=reserved<=maximum and rss<=8*1024**3,elapsed_s=time.monotonic()-start,
            optimizer_steps=0,preprocessing_version=a.preprocessing)
        with a.resource_output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(summary),flush=True)
    return int(summary['error'] is not None or summary['no_valid_transfer'] or not record['limits_passed'])


if __name__=='__main__':raise SystemExit(main())
