"""ROS-container decoder for the exact frozen development input card."""
import argparse
import json
from pathlib import Path
import numpy as np
import rosbag
from sensor_msgs import point_cloud2
from tf.transformations import euler_from_quaternion
from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
from mtare_topo.integration.aee_lidar_pose import commanded_lidar_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output)
    card=json.loads((root/'configs/v3/gate6/data_cards/gse_learned_geometry_development_inputs_v1.json').read_text())['scope']
    paths=card['source_sha256'];bagpath=next(k for k in paths if k.endswith('.bag'))
    tracepath=next(k for k in paths if k.endswith('live_geometry.jsonl'))
    rows=[json.loads(l) for l in (root/tracepath).open()]
    wanted={round(r['pose_evidence']['odom_stamp_sec'],9) for r in rows}
    poses={}
    with rosbag.Bag(str(root/bagpath)) as bag:
        for _,m,_ in bag.read_messages(topics=['/state_estimation']):
            stamp=round(m.header.stamp.to_sec(),9)
            if stamp not in wanted:continue
            if m.header.frame_id!='map' or m.child_frame_id!='sensor':raise ValueError('pose frame drift')
            t=m.pose.pose.position;q=m.pose.pose.orientation
            poses[stamp]=commanded_lidar_pose([t.x,t.y,t.z],euler_from_quaternion([q.x,q.y,q.z,q.w]),m.twist.twist.angular.z)
    if set(poses)!=wanted:raise ValueError('original causal odometry missing from bag')
    scans={}
    keys=sorted({k for w in card['windows'] for k in w['source_frame_keys']},key=lambda k:int(k.split(':')[-1]))
    with rosbag.Bag(str(root/bagpath)) as bag:
        for _,m,_ in bag.read_messages(topics=['/velodyne_points']):
            key=m.header.frame_id+':'+str(m.header.stamp.to_nsec())
            if key not in keys:continue
            if key in scans:raise ValueError('duplicate source scan')
            ranges,valid,_=aee_organized_pointcloud2_to_range_image(m,point_cloud2)
            scans[key]=np.stack([ranges/50.,valid.astype(np.float32)])
    if set(scans)!=set(keys) or len(rows)!=len(keys):raise ValueError('source population mismatch')
    matrices=[];stamps=[]
    for key,row in zip(keys,rows):
        ev=row['pose_evidence'];stamp=ev['scan_stamp_sec']
        if abs(stamp-int(key.split(':')[-1])*1e-9)>1e-7:raise ValueError('scan/trace order mismatch')
        if not 0<=stamp-ev['odom_stamp_sec']<=.1:raise ValueError('noncausal odometry')
        pose=poses[round(ev['odom_stamp_sec'],9)]
        if row.get('result') and not np.allclose(pose,row['result']['geometry']['sensor_to_local_odometry'],atol=1e-10,rtol=0):
            raise ValueError('reconstructed pose differs from original')
        matrices.append(pose);stamps.append(stamp)
    with out.open('xb') as f:
        np.savez_compressed(f,range_valid=np.stack([scans[k] for k in keys]),poses=np.stack(matrices),
                            stamps=np.array(stamps),source_keys=np.array(keys))
    print(json.dumps(dict(frames=len(keys),observations=len(card['windows']),pose_reconstruction_verified=True)))


if __name__=='__main__':main()
