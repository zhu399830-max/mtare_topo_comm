#!/usr/bin/env python3
"""Record native snapshots and return exact identity unless a pinned cache exists.

This node publishes only native_structure/advice, never a waypoint or finish.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import queue

from _bootstrap import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / 'integration/native_structure_bridge'))
from advice_sidecar_core import PinnedAdviceCache, reply_for_snapshot
from mtare_topo.integration.native_route_advice import decode_advice
from mtare_topo.integration.native_region_task_registry import NativeRegionTaskRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', default='/sensor_coverage_planner')
    parser.add_argument('--segment', default='tunnel_seed11_shadow')
    parser.add_argument('--trajectory-id', default='tunnel_seed11_native')
    parser.add_argument('--cache-manifest', type=Path)
    parser.add_argument('--cache-manifest-sha256')
    parser.add_argument('--live-token-artifacts', help='Exact project-relative service output directory')
    parser.add_argument('--live-token-inputs', help='Exact project-relative prefetch input directory')
    parser.add_argument('--live-token-max-entries', type=int)
    args = parser.parse_args()
    if bool(args.cache_manifest) != bool(args.cache_manifest_sha256):
        raise ValueError('Cache manifest and pinned SHA256 are required together')
    cache = PinnedAdviceCache(args.cache_manifest, args.cache_manifest_sha256) if args.cache_manifest else None
    live_fields=(args.live_token_artifacts,args.live_token_inputs,args.live_token_max_entries)
    if any(x is not None for x in live_fields) and not all(x is not None for x in live_fields):
        raise ValueError('Complete bounded live-token scope required')
    token_cache=None
    if args.live_token_artifacts:
        from mtare_topo.integration.live_structural_token_cache import LiveStructuralTokenCache
        token_cache=LiveStructuralTokenCache(PROJECT_ROOT,artifact_prefix=args.live_token_artifacts,
            input_prefix=args.live_token_inputs,segment=args.segment,max_entries=args.live_token_max_entries)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import rospy
    from std_msgs.msg import String
    rospy.init_node('native_structure_advice_sidecar', anonymous=False)
    prefix = args.namespace.rstrip('/') + '/native_structure/'
    publisher = rospy.Publisher(prefix + 'advice', String, queue_size=2)
    lock = threading.RLock()
    registry = NativeRegionTaskRegistry(segment=args.segment, trajectory_id=args.trajectory_id)
    trace = (output / 'trace.jsonl').open('x', encoding='utf-8')
    counts = dict(candidates=0, advice_published=0, feedback=0, decisions=0, failures=0,
                  identity_fallback=0, cache_applied=0, registry_records=0,
                  registry_rejected=0, registry_exceptions=0)
    sequence = 0
    failed = False
    closing = False
    reported_registry_reasons = set()
    token_queue=queue.Queue(maxsize=16)
    token_stop=threading.Event()
    token_counts=dict(received=0,loaded=0,lookup_ready=0,lookup_missing=0,failures=0)

    def serialized(callback):
        def wrapped(*values):
            with lock:
                if closing or failed:
                    return
                return callback(*values)
        return wrapped

    def record(kind, message, **fields):
        nonlocal sequence
        with lock:
            sequence += 1
            row = dict(trace_sequence=sequence, kind=kind, receive_ros_stamp_ns=rospy.Time.now().to_nsec(),
                receive_wall_monotonic_ns=time.monotonic_ns(), raw_message=message,
                raw_message_sha256=hashlib.sha256(message.encode()).hexdigest(), **fields)
            registry_error = None
            if kind in ('candidate_response', 'feedback', 'decision'):
                registry_kind = {'candidate_response':'snapshot','feedback':'feedback','decision':'decision'}[kind]
                payload = fields['snapshot' if kind == 'candidate_response' else 'payload']
                record_ref = str(output / 'trace.jsonl') + ':' + str(sequence)
                row['registry_record_ref'] = record_ref
                counts['registry_records'] += 1
                try:
                    verdict = registry.consume(registry_kind, payload, order=sequence, record_ref=record_ref)
                    row['registry_decision'] = verdict
                    if not verdict['accepted']:
                        counts['registry_rejected'] += 1
                        if verdict['reason'] not in reported_registry_reasons:
                            reported_registry_reasons.add(verdict['reason'])
                            rospy.logwarn('Native region evidence rejected explicitly: %s', verdict['reason'])
                except Exception as error:
                    counts['registry_exceptions'] += 1
                    registry_error = error
                    row['registry_exception'] = repr(error)
            trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            trace.flush()
            if registry_error is not None:
                raise RuntimeError('Native registry consumer failed; raw event preserved') from registry_error

    @serialized
    def candidate_callback(message):
        nonlocal failed
        if failed:
            return
        try:
            snapshot = decode_advice(message.data)
            token_status=None
            if token_cache is not None:
                token_status=token_cache.lookup(snapshot,decision_monotonic_ns=time.monotonic_ns())
                token_counts['lookup_ready' if token_status['ready'] else 'lookup_missing']+=1
            result = reply_for_snapshot(snapshot, now_ns=rospy.Time.now().to_nsec(), cache=cache)
            if token_status is not None:
                result['live_token_status']=token_status
                # Readiness alone never becomes route advice. The verified
                # direction/task consumer is a separate remaining connection.
                result['live_token_used_for_route']=False
            # Publish promptly inside the frozen 100 ms cache-only response budget.
            if result['advice'] is not None:
                publisher.publish(String(data=json.dumps(result['advice'], separators=(',', ':'), allow_nan=False)))
                counts['advice_published'] += 1
            counts['candidates'] += 1
            counts['identity_fallback'] += int(result['identity_fallback'])
            counts['cache_applied'] += int(result['reason'] == 'PINNED_EXACT_EPOCH_CACHE')
            record('candidate_response', message.data, snapshot=snapshot, **result)
        except Exception as error:
            failed = True
            counts['failures'] += 1
            record('candidate_failure', message.data, error=repr(error), waypoint_published=False)
            rospy.logerr('Native advice sidecar failed safely without control: %s', error)
            rospy.signal_shutdown(str(error))

    @serialized
    def read_callback(message, kind):
        nonlocal failed
        try:
            payload = decode_advice(message.data)
            counts['feedback' if kind == 'feedback' else 'decisions'] += 1
            record(kind, message.data, payload=payload)
        except Exception as error:
            failed = True
            counts['failures'] += 1
            record('feedback_failure', message.data, error=repr(error), waypoint_published=False)
            rospy.signal_shutdown(str(error))

    @serialized
    def token_callback(message):
        nonlocal failed
        try:
            if message._connection_header.get('callerid')!='/native_structure_live_inputs':
                raise ValueError('LIVE_TOKEN_PUBLISHER_MISMATCH')
            token_queue.put_nowait((message.data,time.monotonic_ns()))
            token_counts['received']+=1
        except Exception as error:
            failed=True;token_counts['failures']+=1
            record('live_token_failure',message.data,error=repr(error))
            rospy.signal_shutdown(str(error))

    def token_loader():
        nonlocal failed
        while not token_stop.is_set() or not token_queue.empty():
            try:raw,receipt=token_queue.get(timeout=.1)
            except queue.Empty:continue
            try:
                details=token_cache.ingest(decode_advice(raw),received_monotonic_ns=receipt)
                with lock:token_counts['loaded']+=1
                record('live_token_loaded',raw,details=details)
            except Exception as error:
                with lock:failed=True;token_counts['failures']+=1
                record('live_token_failure',raw,error=repr(error))
                rospy.signal_shutdown(str(error))
            finally:token_queue.task_done()

    subscribers = [
        rospy.Subscriber(prefix + 'candidates', String, candidate_callback, queue_size=2),
        rospy.Subscriber(prefix + 'feedback', String, lambda msg: read_callback(msg, 'feedback'), queue_size=2000),
        rospy.Subscriber(prefix + 'decision', String, lambda msg: read_callback(msg, 'decision'), queue_size=200),
    ]
    token_thread=None
    if token_cache is not None:
        token_thread=threading.Thread(target=token_loader,daemon=False)
        token_thread.start()
        subscribers.append(rospy.Subscriber('/native_structure/token_ready',String,token_callback,queue_size=16))
    (output / 'ready.json').write_text(json.dumps(dict(schema_version='native_advice_sidecar_ready_v1',
        advice_topic=prefix + 'advice', control_topics=[], model_executed=False,
        native_region_registry=True, segment=args.segment, trajectory_id=args.trajectory_id,
        live_token_cache=token_cache is not None,live_token_route_consumer=False,
        cache_manifest_sha256=None if cache is None else cache.manifest_sha256)) + '\n')
    try:
        rospy.spin()
    finally:
        with lock:
            closing = True
        for subscriber in subscribers:
            subscriber.unregister()
        token_stop.set()
        if token_thread is not None:token_thread.join()
        with lock:
            trace.flush()
            trace.close()
            registry_state = registry.snapshot()
        local_execution = registry_state['local_execution']
        motion_samples = [item for item in local_execution['events'] if item['kind'] == 'odometry_observed']
        (output / 'native_region_tasks.json').write_text(json.dumps(registry_state,
            sort_keys=True, allow_nan=False) + '\n')
        (output / 'summary.json').write_text(json.dumps(dict(schema_version='native_advice_sidecar_summary_v1',
            counts=counts, failed=failed, model_executed=False, waypoint_published=False,
            live_token_counts=token_counts,live_token_route_changes=0,
            local_execution_command_count=len(local_execution['commands']),
            local_execution_odometry_sample_count=len(motion_samples),
            local_execution_observed_chord_m=sum(item['observed_step_chord_m'] for item in motion_samples),
            local_execution_is_structural_completion=False,
            registry_task_count=len(registry_state['tasks']),
            registry_rejection_reasons=sorted(reported_registry_reasons),
            native_region_registry_sha256=hashlib.sha256((output / 'native_region_tasks.json').read_bytes()).hexdigest(),
            trace_sha256=hashlib.sha256((output / 'trace.jsonl').read_bytes()).hexdigest()), indent=2) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
