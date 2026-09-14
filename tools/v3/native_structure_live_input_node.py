#!/usr/bin/env python3
"""Prefetch exact five-frame tokens from live authenticated scan callbacks.

No advice or control publisher. Every attempted window and overload is logged.
The decision consumer must still check exact native source keys and whether the
response had actually arrived. Recorded sensor poses are commanded, not TF GT.
"""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import deque
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np

from export_native_structure_windows import commanded_pose,layout,ros_header_text,write,sha
from mtare_topo.integration.aee_organized_scan_adapter import validate_aee_organized_pointcloud2
from mtare_topo.integration.explicit_scan_sources import ObservedScan,ExplicitScanSources
from mtare_topo.integration.structural_token_transport import StructuralTokenClient


def write_window(root, output, index, parts, binding, *, segment):
    """Exact raw-byte window writer shared by ROS and synthetic tests."""
    if len(parts)!=5 or binding.get('status')!='EXPLICIT_SOURCE_BOUND':
        raise ValueError('EXACT_BOUND_FIVE_REQUIRED')
    if [p['frame']['source_key'] for p in parts]!=binding['raw_source_keys']:
        raise ValueError('WINDOW_RAW_SOURCE_BINDING')
    if any(p.get('error') for p in parts): raise ValueError('WINDOW_REJECTED_SOURCE_POSE')
    name=f'prefetch_{index:06d}'
    target=output/(name+'.npz')
    logical=output.resolve().relative_to(root.resolve())
    with target.open('xb') as stream:
        np.savez_compressed(stream,raw_pointcloud_bytes=np.stack([
            np.frombuffer(p['payload'],np.uint8) for p in parts]),
            world_from_sensor=np.stack([p['pose'] for p in parts]).astype(np.float64))
    pose_path=output/(name+'_poses.json')
    write(pose_path,dict(frames=[p['evidence'] for p in parts],physical_pose_verified=False))
    entry=dict(window_id=name,sequence_id=segment,input=dict(path=str(logical/target.name),sha256=sha(target)),
        pose_binding=dict(mode='commanded_pose',reference=str(logical/pose_path.name)),
        frames=[deepcopy(p['frame']) for p in parts])
    manifest=output/(name+'_manifest.json')
    write(manifest,dict(schema_version='frozen_structural_token_windows_v1',windows=[entry]))
    write(output/(name+'_binding.json'),binding)
    return dict(schema_version='streaming_structural_token_request_v1',request_id=name,
        manifest=dict(path=str(logical/manifest.name),sha256=sha(manifest)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--segment',required=True)
    p.add_argument('--producer-manifest',type=Path,required=True)
    p.add_argument('--producer-manifest-sha256',required=True)
    p.add_argument('--max-raw-frames',type=int,required=True)
    p.add_argument('--max-windows',type=int,required=True)
    args=p.parse_args()
    if not 5<=args.max_raw_frames<=20000 or not 1<=args.max_windows<=20000:
        raise ValueError('EXPLICIT_LIVE_POPULATION_BOUND')
    if sha(args.producer_manifest)!=args.producer_manifest_sha256:
        raise ValueError('PRODUCER_MANIFEST_DRIFT')
    producer=json.loads(args.producer_manifest.read_text())
    if (producer.get('original_sha256')!='ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e'
            or producer.get('output_sha256')!='72b5ccc1bb631bdcf89094010cb0188d43f0ecb6df63e61ad21fedf31fa2b7ae'
            or producer.get('scan_arithmetic_changed') is not False):
        raise ValueError('PINNED_INSTRUMENTED_PRODUCER_REQUIRED')
    output=args.output.resolve();output.relative_to(ROOT.resolve());output.mkdir(parents=True,exist_ok=False)
    import rospy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    rospy.init_node('native_structure_live_inputs',anonymous=False)
    token_publisher=rospy.Publisher('/native_structure/token_ready',String,queue_size=16,latch=False)
    proof=ExplicitScanSources(segment_id=args.segment,source_artifact_ref=str(args.producer_manifest))
    raw={};registered={};echoes={};odoms=deque(maxlen=400);history=deque(maxlen=5)
    work=queue.Queue(maxsize=16);lock=threading.RLock();stop=threading.Event()
    counts=dict(raw=0,registered=0,echoes=0,windows=0,overloads=0,processed=0,rejected=0,failures=0)
    trace=(output/'live_inputs.jsonl').open('x')
    def record(kind,**fields):
        with lock:
            trace.write(json.dumps(dict(kind=kind,receipt_monotonic_ns=time.monotonic_ns(),**fields),allow_nan=False)+'\n');trace.flush()
    def drain():
        # Join in registered sequence order. Missing earlier members wait; a
        # later echo cannot silently replace them or create a guessed source.
        for stamp,g in sorted(registered.items(),key=lambda item:item[1].seq):
            if g.seq<=proof.last_output_seq: continue
            echo=echoes.get(stamp)
            if echo is None or echo['payload']['source_stamp_ns'] not in raw: break
            source=raw[echo['payload']['source_stamp_ns']]
            proof.add(echo['payload'],raw=source['header'],registered=g,
                evidence_receipt_ns=echo['receipt_ns'],evidence_ref=echo['record_ref'])
            history.append(stamp)
            if len(history)<5: continue
            index=counts['windows']
            # This is a hard recording resource ceiling, not the experiment's
            # normal duration. Stop affected capture on an additional window;
            # do not count a window that was never accepted for recording.
            if index>=args.max_windows: raise ValueError('LIVE_WINDOW_CAP_EXCEEDED')
            counts['windows']+=1
            binding=proof.window([f'registered_scan:{s}' for s in history],segment_id=args.segment,
                snapshot_stamp_ns=stamp,snapshot_receipt_ns=time.monotonic_ns())
            parts=[raw[int(key.split(':')[1])] for key in binding['raw_source_keys']]
            if any(part.get('error') for part in parts):
                counts['rejected']+=1;record('window_rejected',index=index,binding=binding,
                    reasons=[part.get('error') for part in parts]);continue
            try:work.put_nowait((index,parts,binding))
            except queue.Full:
                counts['overloads']+=1;record('window_overload',index=index,binding=binding,
                    reason='BOUNDED_QUEUE_FULL_NO_WINDOW_SUBSTITUTION')
    expected={'raw':'/gazebo','registered':'/vehicleSimulator','echo':'/vehicleSimulator',
              'pose':'/vehicleSimulator'}
    def callback(message,kind):
        with lock:
            if stop.is_set():return
            try:
                caller=ros_header_text(message._connection_header.get('callerid'))
                if caller!=expected[kind]:raise ValueError('LIVE_PUBLISHER_MISMATCH:'+kind+':'+caller)
                receipt=time.monotonic_ns()
                if kind=='pose':odoms.append(message);return
                if kind=='echo':
                    payload=json.loads(message.data);stamp=payload['registered_stamp_ns']
                    if stamp in echoes:raise ValueError('DUPLICATE_SOURCE_ECHO')
                    counts['echoes']+=1
                    echo=dict(payload=payload,receipt_ns=receipt,
                        record_ref=str(output/'live_inputs.jsonl')+':echo:'+str(counts['echoes']))
                    echoes[stamp]=echo;record('source_echo',**echo)
                else:
                    stamp=message.header.stamp.to_nsec()
                    header=ObservedScan(message.header.seq,stamp,receipt)
                    if kind=='registered':
                        if stamp in registered:raise ValueError('DUPLICATE_REGISTERED_SCAN')
                        registered[stamp]=header;counts['registered']+=1
                    else:
                        if stamp in raw:raise ValueError('DUPLICATE_RAW_SCAN')
                        if counts['raw']>=args.max_raw_frames:raise ValueError('LIVE_RAW_CAP_REACHED')
                        position=evidence=None;error=None
                        try:
                            validate_aee_organized_pointcloud2(message)
                            past=[o for o in odoms if o.header.stamp.to_nsec()<=stamp]
                            if not past:raise ValueError('MISSING_CAUSAL_POSE')
                            odom=max(past,key=lambda o:o.header.stamp.to_nsec())
                            position,evidence=commanded_pose(odom,stamp)
                            evidence['source_key']='/state_estimation:'+str(odom.header.stamp.to_nsec())
                        except ValueError as exc:error=str(exc)
                        raw[stamp]=dict(header=header,payload=bytes(message.data),pose=position,evidence=evidence,error=error,
                            frame=dict(frame_order=counts['raw'],stamp_ns=stamp,source_key='/velodyne_points:'+str(stamp),
                                frame_id=message.header.frame_id,pose_source_key=None if evidence is None else evidence['source_key'],
                                pointcloud_layout=layout(message)))
                        counts['raw']+=1
                        record('raw_header',header=vars(header),payload_sha256=hashlib.sha256(bytes(message.data)).hexdigest(),error=error)
                drain()
            except Exception as exc:
                counts['failures']+=1;record('fatal',reason=repr(exc));stop.set();rospy.signal_shutdown(str(exc))
    def worker():
        client=StructuralTokenClient(args.socket,timeout_s=10.)
        while not stop.is_set() or not work.empty():
            try:index,parts,binding=work.get(timeout=.1)
            except queue.Empty:continue
            try:
                request=write_window(ROOT,output,index,parts,binding,segment=args.segment)
                result=client.request(request)
                counts['processed']+=int(result['response'].get('status')=='PROCESSED')
                counts['rejected']+=int(result['response'].get('status')!='PROCESSED')
                record('token_response',index=index,binding=binding,**result)
                if result['response'].get('status')=='PROCESSED':
                    notification=dict(schema_version='live_structural_token_ready_v1',segment=args.segment,
                        binding=binding,client_result=result)
                    token_publisher.publish(String(data=json.dumps(notification,allow_nan=False)))
            except Exception as exc:
                counts['failures']+=1;record('token_worker_failure',index=index,reason=repr(exc),binding=binding)
                stop.set();rospy.signal_shutdown(str(exc))
            finally:work.task_done()
    thread=threading.Thread(target=worker,daemon=False);thread.start()
    subscriptions=[rospy.Subscriber(topic,typ,lambda msg,k=kind:callback(msg,k),queue_size=400)
        for topic,typ,kind in [('/state_estimation',Odometry,'pose'),('/velodyne_points',PointCloud2,'raw'),
            ('/registered_scan',PointCloud2,'registered'),('/native_structure/scan_sources',String,'echo')]]
    write(output/'ready.json',dict(prefetch=True,control_topics=[],weights_service=str(args.socket),
        data_topics=['/native_structure/token_ready'],
        queue_capacity=16,physical_pose_verified=False,model_may_change_current_route=False,
        population_limits='hard resource ceilings, not normal completion criteria'))
    try:rospy.spin()
    finally:
        stop.set()
        for subscription in subscriptions:subscription.unregister()
        thread.join()
        with lock:trace.close()
        write(output/'summary.json',dict(counts=counts,control_published=False,training_steps=0,
            unresolved_registered=len(registered)-len(proof.by_registered),
            availability='client receipt must precede each consuming decision; no epoch rebasing'))
    return int(counts['failures']>0)


if __name__=='__main__':raise SystemExit(main())
