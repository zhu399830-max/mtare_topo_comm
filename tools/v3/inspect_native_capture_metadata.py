"""Recover only the failed summary's bag-header fields; no message replay."""
from _bootstrap import PROJECT_ROOT
import argparse
import json
from pathlib import Path
from export_native_structure_windows import ros_header_text,write
from mtare_topo.integration.structural_recording_requirements import recording_readiness


def main():
    import rosbag
    p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with rosbag.Bag(str(a.bag)) as bag:
        inventory={k:dict(type=v.msg_type,count=v.message_count) for k,v in bag.get_type_and_topic_info().topics.items()}
        publishers=sorted({ros_header_text(c.header.get('callerid')) for c in bag._get_connections(topics=['/way_point'])})
    result=dict(inventory=inventory,waypoint_publishers=publishers,
        single_original_control_publisher=publishers==['/sensor_coverage_planner/tare_planner_node'],
        readiness=recording_readiness(inventory,external_pose_binding=True,case_method='bridge'),
        message_payloads_decoded=0,raw_reconstructed=False,window_reexported=False,
        repair='decode_ros_callerid_bytes_only_no_input_or_model_change')
    write(a.output,result)
    if not result['single_original_control_publisher'] or not result['readiness']['native_route_replay_readiness']['ready']:
        raise RuntimeError('native capture metadata prerequisites failed')


if __name__=='__main__':main()
