"""Original bytes/poses at every native snapshot, joined by callback evidence.

Never nearest-time or index-offset source pairing. Unknown/missing evidence is
retained per window; source identity does not prove actual sensor-link pose.
"""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import deque
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from export_native_structure_windows import sha,write,ros_header_text,commanded_pose,layout
from mtare_topo.integration.explicit_scan_sources import ObservedScan,ExplicitScanSources
from mtare_topo.integration.aee_organized_scan_adapter import validate_aee_organized_pointcloud2

SOURCE_TOPIC='/native_structure/scan_sources'
NATIVE_PREFIX='/sensor_coverage_planner/native_structure/'


def native_window_bindings(raw,registered,echoes,snapshots,*,segment,source_ref):
    """Pure source join, shared by real reader and synthetic tests."""
    proof=ExplicitScanSources(segment_id=segment,source_artifact_ref=source_ref)
    for echo in sorted(echoes,key=lambda x:x['payload']['registered_seq']):
        msg=echo['payload']; rk=msg['source_stamp_ns']; gk=msg['registered_stamp_ns']
        if rk not in raw or gk not in registered:
            # Capture head/tail can miss a member; never infer that member.
            continue
        proof.add(msg,raw=raw[rk],registered=registered[gk],
            evidence_receipt_ns=echo['receipt_ns'],evidence_ref=echo['record_ref'])
    result=[]
    for i,row in enumerate(snapshots):
        snap=row['payload']
        binding=proof.window(snap['source_frame_keys'],segment_id=segment,
            snapshot_stamp_ns=snap['stamp_ns'],snapshot_receipt_ns=row['receipt_ns'])
        result.append(dict(window_id='native_%03d'%i,epoch=snap['epoch'],
            native_snapshot_ref=row['record_ref'],native_snapshot=snap,**binding))
    return result


def export(bag_path,output,logical,*,source_manifest,source_manifest_logical):
    import rosbag
    output.mkdir(parents=True,exist_ok=False)
    source=json.loads(source_manifest.read_text())
    if (source.get('original_sha256')!='ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e'
            or source.get('output_sha256')!='72b5ccc1bb631bdcf89094010cb0188d43f0ecb6df63e61ad21fedf31fa2b7ae'
            or source.get('scan_arithmetic_changed') is not False):
        raise ValueError('PINNED_INSTRUMENTED_PRODUCER_REQUIRED')
    odoms=deque(maxlen=400); raw={}; reg={}; data={}; echoes=[]; snapshots=[]
    topics=['/state_estimation','/velodyne_points','/registered_scan',SOURCE_TOPIC,
            *[NATIVE_PREFIX+x for x in ('candidates','decision','feedback','advice')]]
    source_ref=str(logical/'source_echoes.jsonl')
    with rosbag.Bag(str(bag_path)) as bag, (output/'source_echoes.jsonl').open('x') as echo_file, (output/'native_traces.jsonl').open('x') as trace_file:
        callers={t:sorted({ros_header_text(c.header['callerid']) for c in bag._get_connections(topics=[t])})
                 for t in [SOURCE_TOPIC,'/velodyne_points','/registered_scan','/way_point']}
        expected={SOURCE_TOPIC:['/vehicleSimulator'],'/velodyne_points':['/gazebo'],
            '/registered_scan':['/vehicleSimulator'],'/way_point':['/sensor_coverage_planner/tare_planner_node']}
        if callers!=expected:raise ValueError('PINNED_SOURCE_OR_CONTROL_PUBLISHER_MISMATCH')
        for order,(topic,msg,receipt) in enumerate(bag.read_messages(topics=topics)):
            ns=receipt.to_nsec()
            if topic==SOURCE_TOPIC:
                row=dict(payload=json.loads(msg.data),receipt_ns=ns,record_ref=source_ref+':'+str(len(echoes)+1))
                echoes.append(row);echo_file.write(json.dumps(row)+'\n');continue
            if topic.startswith(NATIVE_PREFIX):
                row=dict(topic=topic,payload=json.loads(msg.data),receipt_ns=ns,
                    record_ref=str(logical/'native_traces.jsonl')+':message:'+str(order))
                trace_file.write(json.dumps(row)+'\n')
                if topic.endswith('/candidates'):snapshots.append(row)
                continue
            if topic=='/state_estimation':odoms.append(msg);continue
            stamp=msg.header.stamp.to_nsec();header=ObservedScan(msg.header.seq,stamp,ns)
            dest=raw if topic=='/velodyne_points' else reg
            if stamp in dest:raise ValueError('DUPLICATE_SCAN_HEADER')
            dest[stamp]=header
            if topic=='/registered_scan':continue
            if len(raw)>700:raise ValueError('BOUNDED120SEC_RAW_POPULATION_EXCEEDED')
            payload=bytes(msg.data);pose=evidence=None;error=None
            try:
                validate_aee_organized_pointcloud2(msg)
                past=[o for o in odoms if o.header.stamp.to_nsec()<=stamp]
                if not past:raise ValueError('MISSING_CAUSAL_POSE')
                odom=max(past,key=lambda o:o.header.stamp.to_nsec());pose,evidence=commanded_pose(odom,stamp)
                evidence['source_key']='/state_estimation:'+str(odom.header.stamp.to_nsec())
            except ValueError as e:error=str(e)
            data[stamp]=dict(payload=payload,pose=pose,evidence=evidence,error=error,
                frame=dict(frame_order=len(raw)-1,stamp_ns=stamp,source_key='/velodyne_points:'+str(stamp),
                    frame_id=msg.header.frame_id,pose_source_key=None if evidence is None else evidence['source_key'],
                    pointcloud_layout=layout(msg)))
    if not 1<=len(snapshots)<=120:raise ValueError('BOUNDED_NATIVE_EPOCH_POPULATION_REQUIRED')
    segment='tunnel_seed11_explicit_source_capture'
    bindings=native_window_bindings(raw,reg,echoes,snapshots,segment=segment,source_ref=source_manifest_logical)
    write(output/'source_bindings.json',bindings)
    windows=[];poses=[];rejected=[];selected=set()
    for b in bindings:
        name=b['window_id']
        if b['status']!='EXPLICIT_SOURCE_BOUND':
            rejected.append(dict(window_id=name,reason=b['reason']));continue
        stamps=[int(k.split(':')[1]) for k in b['raw_source_keys']];parts=[data[s] for s in stamps]
        errors=[p['error'] for p in parts if p['error']]
        if errors:rejected.append(dict(window_id=name,reason='INVALID_SOURCE_POSE',details=errors));continue
        selected.update(stamps);dest=output/(name+'.npz')
        with dest.open('xb') as f:
            np.savez_compressed(f,raw_pointcloud_bytes=np.stack([np.frombuffer(p['payload'],np.uint8) for p in parts]),
                world_from_sensor=np.stack([p['pose'] for p in parts]).astype(np.float64))
        windows.append(dict(window_id=name,sequence_id=segment,
            input=dict(path=str(logical/dest.name),sha256=sha(dest)),
            pose_binding=dict(mode='commanded_pose',reference=str(logical/'pose_evidence.json')+'#'+name),
            frames=[p['frame'] for p in parts]))
        poses.append(dict(window_id=name,frames=[p['evidence'] for p in parts]))
    write(output/'windows_manifest.json',dict(schema_version='frozen_structural_token_windows_v1',windows=windows))
    write(output/'pose_evidence.json',dict(actual_link_pose_measured=False,windows=poses,
        source_sha256=sha(ROOT/'src/mtare_topo/integration/aee_lidar_pose.py')))
    write(output/'rejected_windows.json',rejected)
    write(output/'raw_frame_inventory.json',[dict(**vars(h),payload_sha256=hashlib.sha256(data[s]['payload']).hexdigest()) for s,h in raw.items()])
    summary=dict(raw_frames=len(raw),registered_frames=len(reg),source_echoes=len(echoes),
        native_snapshots=len(snapshots),exported_windows=len(windows),rejected_windows=len(rejected),
        unique_selected_raw_frames=len(selected),model_input_population='all recorded native epochs; no model or reference selection',
        source_inference='explicit same-callback headers, not time proximity or seq offset',
        producer_manifest_ref=source_manifest_logical,producer_manifest_sha256=sha(source_manifest),
        bag_sha256=sha(bag_path),caller_ids=callers,physical_pose_verified=False,
        model_forwards=0,training_steps=0,learning_changed_control=False)
    write(output/'summary.json',summary)
    if rejected or not windows:raise RuntimeError('Exact native input windows incomplete; retained all failures')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ['bag','output','logical','source-manifest']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--source-manifest-logical',required=True);a=p.parse_args()
    print(json.dumps(export(a.bag,a.output,a.logical,source_manifest=a.source_manifest,source_manifest_logical=a.source_manifest_logical)))
