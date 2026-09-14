"""Container-local bounded ROS diagnostic; no target publisher or global planner."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    out=Path('/evidence');children=[];handles=[];error=None
    def launch(name,args):
        handle=(out/(name+'.log')).open('x');handles.append(handle)
        p=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    try:
        system=launch('system',['roslaunch','/workspace/configs/v3/gate5/roslaunch/system_seeded.launch',
            'world_name:=tunnel','gazebo_seed:=11','vis_tools:=false','rviz:=false','gazebo_gui:=false'])
        import rospy
        from sensor_msgs.msg import PointCloud2
        rospy.init_node('geometry_trial_supervisor',anonymous=True,disable_signals=True)
        # Wait for an actual sensor message, not merely elapsed launch time.
        rospy.wait_for_message('/velodyne_points',PointCloud2,timeout=90)
        bag=launch('record',['rosbag','record','-O',str(out/'sensors.bag'),
            '/velodyne_points','/state_estimation','/state_estimation_at_scan',
            '/tf','/tf_static','/clock','/terrain_map','/path','/cmd_vel','/way_point'])
        node_args=['python3','/workspace/tools/v3/live_geometry_shadow_node.py',
            '--output',str(out/'geometry'),'--policy','/evidence/policy.json',
            '--pose-interface',os.environ.get('GSE_POSE_INTERFACE','tf')]
        if os.environ.get('GSE_EXECUTION')=='1':node_args+=['--execution-policy','/evidence/execution_policy.json']
        if os.environ.get('GSE_BRANCHES')=='1':node_args+=['--branch-policy','/evidence/branch_policy.json']
        if os.environ.get('GSE_RETURNS')=='1':node_args+=['--remote-returns']
        if os.environ.get('GSE_LEARNED_SOCKET'):
            node_args+=['--learned-socket',os.environ['GSE_LEARNED_SOCKET'],
                        '--checkpoint-sha256',os.environ['GSE_CHECKPOINT_SHA256']]
        node=launch('geometry',node_args)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if any(p.poll() is not None for p in (system,bag,node)):
                raise RuntimeError('required ROS process exited before diagnostic end')
            time.sleep(.2)
    except BaseException as exc:error=repr(exc)
    finally:
        for p in reversed(children):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in reversed(children):
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();error=error or 'shutdown timeout'
        for h in handles:h.close()
    snapshot=out/'geometry/live_geometry_snapshot.json'
    state=json.loads(snapshot.read_text()) if snapshot.exists() else {}
    if state.get('failed') or state.get('scans',0)<5:error=error or 'fewer than five accepted scans or frontend failure'
    counts={}
    travel=0.;previous=None
    try:
        import rosbag
        with rosbag.Bag(str(out/'sensors.bag')) as bag:
            counts={k:v.message_count for k,v in bag.get_type_and_topic_info().topics.items()}
            for _,m,_ in bag.read_messages(topics=['/state_estimation']):
                p=m.pose.pose.position;xyz=(p.x,p.y,p.z)
                if previous is not None:travel+=sum((a-b)**2 for a,b in zip(xyz,previous))**.5
                previous=xyz
        if counts.get('/velodyne_points',0)<5:error=error or 'raw scan evidence missing'
    except Exception as exc:error=error or repr(exc)
    arrivals=sum(e['kind']=='XY_ARRIVAL_FROM_ODOMETRY' for e in state.get('execution_events',[]))
    branch_arrivals=sum(e['kind']=='XY_ARRIVAL_FROM_ODOMETRY' and 'branch_task' in e['target'] for e in state.get('execution_events',[]))
    return_events=(state.get('branch_tasks') or {}).get('return_events',[])
    return_reached=sum(e['status']=='RETURN_PLACE_REACHED' for e in return_events)
    return_revalidated=sum(e['status']=='RETURN_DIRECTION_REOBSERVED' for e in return_events)
    if os.environ.get('GSE_RETURNS')=='1' and min(return_reached,return_revalidated)<1:error=error or 'no recorded return and reobservation completed'
    if os.environ.get('GSE_BRANCHES')=='1' and branch_arrivals<1:error=error or 'no geometry-place task arrival recorded'
    if os.environ.get('GSE_EXECUTION')=='1' and (travel<1 or arrivals<1):error=error or 'execution requires measured travel >=1m and >=1 XY arrival'
    (out/'trial_summary.json').write_text(json.dumps(dict(error=error,travel_m=travel,xy_arrivals=arrivals,branch_task_arrivals=branch_arrivals,
        return_places_reached=return_reached,return_directions_revalidated=return_revalidated,
        accepted_scans=state.get('scans',0),bag_topic_counts=counts,closed_loop=os.environ.get('GSE_EXECUTION')=='1',
        complete_exploration_verified=False)))
    return int(error is not None)

if __name__=='__main__':raise SystemExit(main())
