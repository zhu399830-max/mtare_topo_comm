"""One streaming export of real organized bytes; no reconstruction or inference."""
from _bootstrap import PROJECT_ROOT
from collections import deque
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from mtare_topo.integration.aee_lidar_pose import commanded_lidar_pose
from mtare_topo.integration.aee_organized_scan_adapter import validate_aee_organized_pointcloud2
from mtare_topo.integration.structural_recording_requirements import recording_readiness, BRIDGE_TOPICS


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    # Validate/serialize before creating the destination, so a type error cannot
    # leave a misleading truncated JSON document behind.
    serialized=json.dumps(obj,indent=2,allow_nan=False)
    with path.open('x') as f:
        f.write(serialized)


def ros_header_text(value):
    if isinstance(value,bytes):
        value=value.decode('utf-8',errors='strict')
    if not isinstance(value,str) or not value:
        raise ValueError('ROS connection header must contain an explicit text callerid')
    return value


def second_bin(stamp_ns, *, start_ns, end_ns):
    if start_ns <= stamp_ns < end_ns:
        return (stamp_ns - start_ns) // 1000000000
    return None


def quaternion_rpy(q):
    x, y, z, w = q
    if not all(math.isfinite(v) for v in q) or not math.isclose(sum(v*v for v in q), 1., abs_tol=1e-6):
        raise ValueError('finite normalized original odometry quaternion required')
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def commanded_pose(odom, scan_stamp):
    stamp = odom.header.stamp.to_nsec()
    if not 0 <= scan_stamp - stamp <= 100000000:
        raise ValueError('no causal original odometry within100ms')
    if odom.header.frame_id.lstrip('/') != 'map' or odom.child_frame_id.lstrip('/') != 'sensor':
        raise ValueError('original map/sensor odometry required')
    p, q = odom.pose.pose.position, odom.pose.pose.orientation
    xyz, quat = [p.x,p.y,p.z], [q.x,q.y,q.z,q.w]
    pose = commanded_lidar_pose(xyz, quaternion_rpy(quat), odom.twist.twist.angular.z)
    return pose, dict(odom_stamp_ns=stamp, scan_stamp_ns=scan_stamp,
        age_ns=scan_stamp-stamp, xyz_m=xyz, quaternion_xyzw=quat,
        angular_z=odom.twist.twist.angular.z, actual_link_pose_measured=False,
        source='original_aee_commanded_odometry_full_rotation')


def layout(scan):
    return dict(width=int(scan.width), height=int(scan.height), point_step=int(scan.point_step),
        row_step=int(scan.row_step), is_bigendian=bool(scan.is_bigendian), is_dense=bool(scan.is_dense),
        fields=[dict(name=f.name, offset=int(f.offset), datatype=int(f.datatype), count=int(f.count)) for f in scan.fields])


def export(bag_path, output, *, project_root, start_ns, end_ns, logical_output_relative=None):
    import rosbag
    if end_ns - start_ns != 120000000000:
        raise ValueError('exact120-second capture interval required')
    logical = Path(logical_output_relative) if logical_output_relative else output.relative_to(project_root)
    if logical.is_absolute() or '..' in logical.parts:
        raise ValueError('explicit repository-relative logical output binding required')
    output.mkdir(parents=True, exist_ok=False)
    windows, rejected, pose_records, raw_index, registered_index = [], [], [], [], []
    odoms, history = deque(maxlen=400), deque(maxlen=5)
    used_seconds, previous_stamp = set(), -1
    topics = ['/velodyne_points','/state_estimation','/registered_scan',*BRIDGE_TOPICS]
    with rosbag.Bag(str(bag_path)) as bag, (output/'native_traces.jsonl').open('x') as trace:
        inventory = {k:dict(type=v.msg_type, count=v.message_count) for k,v in bag.get_type_and_topic_info().topics.items()}
        readiness = recording_readiness(inventory, external_pose_binding=True, case_method='bridge')
        callers = sorted({ros_header_text(c.header.get('callerid')) for c in bag._get_connections(topics=['/way_point'])})
        for topic, message, arrival in bag.read_messages(topics=topics):
            if topic in BRIDGE_TOPICS:
                trace.write(json.dumps(dict(topic=topic, bag_stamp_ns=arrival.to_nsec(),
                    payload=json.loads(message.data), serialized_payload_sha256=hashlib.sha256(message.data.encode()).hexdigest()), allow_nan=False)+'\n')
                continue
            if topic == '/state_estimation':
                odoms.append(message)
                continue
            stamp = message.header.stamp.to_nsec()
            if topic == '/registered_scan':
                registered_index.append(dict(source_key='registered_scan:'+str(stamp), stamp_ns=stamp, bag_stamp_ns=arrival.to_nsec()))
                continue
            order = len(raw_index)
            source_key = '/velodyne_points:'+str(stamp)
            row = dict(frame_order=order, stamp_ns=stamp, source_key=source_key,
                frame_id=message.header.frame_id, bag_stamp_ns=arrival.to_nsec(),
                payload_sha256=hashlib.sha256(bytes(message.data)).hexdigest())
            raw_index.append(row)
            if stamp <= previous_stamp:
                raise ValueError('raw timestamps are not strictly increasing; no reordering')
            previous_stamp = stamp
            error, pose, pose_ref = None, None, None
            try:
                validate_aee_organized_pointcloud2(message)
                if message.header.frame_id.lstrip('/') != 'velodyne':
                    raise ValueError('original raw velodyne sensor frame required')
                past = [p for p in odoms if p.header.stamp.to_nsec() <= stamp]
                if not past:
                    raise ValueError('no past odometry received before this raw scan')
                odom = max(past, key=lambda p:p.header.stamp.to_nsec())
                pose, pose_ref = commanded_pose(odom, stamp)
                pose_ref['source_key'] = '/state_estimation:'+str(odom.header.stamp.to_nsec())
            except ValueError as exc:
                error = str(exc)
            history.append((message,row,pose,pose_ref,error))
            bin_id = second_bin(stamp,start_ns=start_ns,end_ns=end_ns)
            if bin_id is None or bin_id in used_seconds or len(history) < 5:
                continue
            used_seconds.add(bin_id)  # Fixed before validity/model; no substitute.
            window_id = 'second_%03d' % bin_id
            failures = [dict(source_key=r['source_key'],reason=e) for _,r,_,_,e in history if e]
            if failures:
                rejected.append(dict(window_id=window_id,frames=[r for _,r,_,_,_ in history], reasons=failures))
                continue
            array_path = output/(window_id+'.npz')
            with array_path.open('xb') as f:
                np.savez_compressed(f, raw_pointcloud_bytes=np.stack([np.frombuffer(bytes(m.data),np.uint8) for m,_,_,_,_ in history]),
                    world_from_sensor=np.stack([p for _,_,p,_,_ in history]).astype(np.float64))
            frames = [dict(frame_order=r['frame_order'],stamp_ns=r['stamp_ns'],source_key=r['source_key'],
                frame_id=r['frame_id'],pose_source_key=pr['source_key'],pointcloud_layout=layout(m)) for m,r,_,pr,_ in history]
            reference = str(logical/'pose_evidence.json')+'#'+window_id
            windows.append(dict(window_id=window_id, sequence_id='tunnel_seed11_native_capture',
                input=dict(path=str(logical/array_path.name),sha256=sha(array_path)),
                pose_binding=dict(mode='commanded_pose',reference=reference),frames=frames))
            pose_records.append(dict(window_id=window_id,frames=[pr for _,_,_,pr,_ in history]))
    manifest = dict(schema_version='frozen_structural_token_windows_v1',windows=windows)
    write(output/'windows_manifest.json',manifest)
    write(output/'pose_evidence.json',dict(source_module='src/mtare_topo/integration/aee_lidar_pose.py',
        source_sha256=sha(project_root/'src/mtare_topo/integration/aee_lidar_pose.py'),
        actual_link_pose_measured=False,windows=pose_records))
    write(output/'frame_index.json',dict(raw_frames=raw_index,registered_frames=registered_index))
    write(output/'rejected_windows.json',rejected)
    raw_stamps = {r['stamp_ns'] for r in raw_index}
    summary = dict(inventory=inventory,readiness=readiness,raw_frames=len(raw_index),
        registered_frames=len(registered_index),selected_windows=len(used_seconds),
        exported_windows=len(windows),rejected_windows=len(rejected),
        timestamp_equal_registered_frames=sum(r['stamp_ns'] in raw_stamps for r in registered_index),
        timestamp_equality_is_transform_or_payload_proof=False,
        waypoint_publishers=callers,single_original_control_publisher=callers==['/sensor_coverage_planner/tare_planner_node'],
        model_forwards=0,training_steps=0,start_ns=start_ns,end_ns=end_ns)
    write(output/'summary.json',summary)
    if not readiness['native_route_replay_readiness']['ready'] or not summary['single_original_control_publisher']:
        raise RuntimeError('missing capture topics or non-native control publisher; evidence preserved')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--start-ns',type=int,required=True);p.add_argument('--end-ns',type=int,required=True)
    p.add_argument('--logical-output-relative')
    a=p.parse_args();print(json.dumps(export(a.bag,a.output,project_root=a.project_root,start_ns=a.start_ns,end_ns=a.end_ns,logical_output_relative=a.logical_output_relative)))
