#!/usr/bin/env python3
"""Live ROS geometry composition; tentative goals require explicit policy opt-in.

Run only inside a frozen development run. TF must resolve the actual LiDAR
frame at its timestamp; odometry child-frame names are not assumed equivalent.
"""
import argparse
import json
from pathlib import Path
import threading
from collections import deque
import numpy as np
from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pose-interface',choices=['tf','aee-commanded-odometry'],default='tf')
    parser.add_argument('--execution-policy',type=Path,
                        help='Explicit opt-in for tentative goals through the original local planner')
    parser.add_argument('--branch-policy',type=Path)
    parser.add_argument('--remote-returns',action='store_true')
    parser.add_argument('--policy',type=Path,required=True,
                        help='Frozen JSON with composition_policy, anchor_spacing_m, lookahead_m')
    args=parser.parse_args()
    policy=json.loads(args.policy.read_text())
    pipeline=LiveGeometryPipeline(**policy)
    import rospy
    import tf2_ros
    from tf.transformations import quaternion_matrix, euler_from_quaternion
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PointStamped
    from mtare_topo.integration.geometry_execution import GeometryExecution
    from mtare_topo.integration.aee_lidar_pose import commanded_lidar_pose
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    rospy.init_node('gse_geometry_shadow',anonymous=False)
    buffer=tf2_ros.Buffer()
    listener=tf2_ros.TransformListener(buffer)
    lock=threading.Lock()
    args.output.mkdir(parents=True,exist_ok=True)
    trace=(args.output/'live_geometry.jsonl').open('x')
    count=0
    goal_publish_count=0
    failed=False
    executor=None;waypoints=None
    if args.execution_policy:
        if args.pose_interface!='aee-commanded-odometry':
            raise ValueError('explicit body odometry required for execution')
        executor=GeometryExecution(**json.loads(args.execution_policy.read_text()))
        if args.branch_policy:
            from mtare_topo.integration.branch_geometry_execution import BranchGeometryExecution
            executor=BranchGeometryExecution(execution_policy=json.loads(args.execution_policy.read_text()),
                place_policy=json.loads(args.branch_policy.read_text()),remote_returns=args.remote_returns)
        waypoints=rospy.Publisher('/way_point',PointStamped,queue_size=1)
    elif args.branch_policy:raise ValueError('branch execution requires execution policy')
    if args.remote_returns and not args.branch_policy:raise ValueError('remote return requires branch tasks')
    poses=deque(maxlen=400)
    def odometry(message):
        with lock:
            poses.append(message)
    odom_subscriber=rospy.Subscriber('/state_estimation',Odometry,odometry,queue_size=400)

    def callback(scan):
        nonlocal count,failed,goal_publish_count
        with lock:
            if failed:return
            try:
                if not scan.header.frame_id or scan.header.frame_id.lstrip('/')=='map':
                    raise ValueError('organized input must name its actual sensor frame')
                pose_evidence={'interface':args.pose_interface}
                if args.pose_interface=='tf':
                    transform=buffer.lookup_transform('map',scan.header.frame_id,
                        scan.header.stamp,rospy.Duration(0.2)).transform
                    q=transform.rotation
                    matrix=quaternion_matrix([q.x,q.y,q.z,q.w])
                    t=transform.translation
                    matrix[:3,3]=[t.x,t.y,t.z]
                else:
                    if scan.header.frame_id.lstrip('/')!='velodyne':
                        raise ValueError('AEE adapter only supports original velodyne')
                    past=[p for p in poses if p.header.stamp<=scan.header.stamp]
                    if not past:raise ValueError('no causal odometry at scan time')
                    pose=max(past,key=lambda p:p.header.stamp.to_nsec())
                    delta=(scan.header.stamp-pose.header.stamp).to_sec()
                    if delta>.1:raise ValueError('scan/past odometry exceeds original 0.1 second contract')
                    if pose.header.frame_id.lstrip('/')!='map' or pose.child_frame_id.lstrip('/')!='sensor':
                        raise ValueError('original map/sensor odometry required')
                    q=pose.pose.pose.orientation;t=pose.pose.pose.position
                    matrix=commanded_lidar_pose([t.x,t.y,t.z],
                        euler_from_quaternion([q.x,q.y,q.z,q.w]),pose.twist.twist.angular.z)
                    pose_evidence.update(odom_stamp_sec=pose.header.stamp.to_sec(),
                        scan_stamp_sec=scan.header.stamp.to_sec(),past_pose_age_sec=delta,
                        actual_link_pose_measured=False)
                ranges,valid,audit=aee_organized_pointcloud2_to_range_image(scan,point_cloud2)
                result=pipeline.push(ranges,valid,sensor_to_map=matrix,
                    stamp_sec=scan.header.stamp.to_sec(),
                    source_key=scan.header.frame_id+':'+str(scan.header.stamp.to_nsec()),
                    coordinate_frame='map')
                if executor is not None and result is not None:
                    body_rotation=quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
                    execution_args=dict(body_xyz_m=[t.x,t.y,t.z],body_forward_world=body_rotation[:,0],stamp_sec=scan.header.stamp.to_sec())
                    if args.branch_policy:request=executor.update_geometry(result,navigation_graph=pipeline.graph,**execution_args)
                    else:request=executor.update(result['decision']['proposals'],**execution_args)
                    message=PointStamped();message.header.frame_id='map';message.header.stamp=scan.header.stamp
                    message.point.x,message.point.y,message.point.z=request['xyz_m']
                    waypoints.publish(message)
                    goal_publish_count+=1
                    result['local_planning_request']=request
                    result['control_published']=True
                count+=1
                trace.write(json.dumps(dict(scan=count,audit=audit.to_dict(),result=result,pose_evidence=pose_evidence),
                                       allow_nan=False)+'\n')
                trace.flush()
            except Exception as exc:
                failed=True
                trace.write(json.dumps(dict(error=str(exc),scan=count,goal_publish_count=goal_publish_count))+'\n')
                trace.flush()
                rospy.logerr('Geometry shadow stopped: %s',exc)
                rospy.signal_shutdown(str(exc))

    subscriber=rospy.Subscriber('/velodyne_points',PointCloud2,callback,queue_size=5)
    try:
        rospy.spin()
    finally:
        with lock:
            with (args.output/'live_geometry_snapshot.json').open('x') as stream:
                json.dump(dict(scans=count,failed=failed,control_published=goal_publish_count>0,
                               goal_publish_count=goal_publish_count,
                               local_goal_publisher_enabled=executor is not None,
                               execution_events=executor.events if executor else [],
                               branch_tasks=executor.snapshot() if args.branch_policy else None,
                               graph=pipeline.graph.snapshot()),stream,allow_nan=False)
            trace.close()
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
