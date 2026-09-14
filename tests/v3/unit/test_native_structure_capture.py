import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/v3'))
import run_native_structure_capture as run
import export_native_structure_windows as exporter
from mtare_topo.governance_native_structure_capture import validate_card,scope_digest,SCHEMA


def card():
    s=run.scope()
    return dict(schema_version=SCHEMA,scope=s,approval=dict(status='APPROVED',approved_by='user',
        authorized_operations=['closed_loop_single'],authorized_gates=[6],
        confirmation_reference='继续执行',scope_sha256=scope_digest(s)))


def test_exact_capture_scope_and_authorization():
    assert validate_card(card()).passed
    for key,value in [('runtime_sec',3600),('world','C08'),('training_steps',1),('bridge_mode','active')]:
        c=card();c['scope'][key]=value;c['approval']['scope_sha256']=scope_digest(c['scope'])
        assert not validate_card(c).passed


def test_command_keeps_one_native_shadow_no_second_control(tmp_path):
    command=run.container_command(tmp_path)
    assert '--network' in command and 'none' in command
    shell=command[-1]
    assert 'native_structure_capture_case.py' in shell and '--runtime-sec 120' in shell
    assert 'closed_loop_recording_topics_gse_native_replay_v1.json' in shell
    assert 'semantic_topology_global_node' not in shell and 'live_geometry_shadow_node' not in shell


def test_time_bins_no_future_and_quaternion_rejection():
    assert exporter.second_bin(100,start_ns=100,end_ns=120000000100)==0
    assert exporter.second_bin(120000000100,start_ns=100,end_ns=120000000100) is None
    assert exporter.second_bin(99,start_ns=100,end_ns=120000000100) is None
    assert exporter.quaternion_rpy([0,0,0,1])==(0,0,0)
    with pytest.raises(ValueError):exporter.quaternion_rpy([0,0,0,2])


class Stamp:
    def __init__(self,n):self.n=n
    def to_nsec(self):return self.n


def odom(stamp):
    return N(header=N(stamp=Stamp(stamp),frame_id='map'),child_frame_id='sensor',
        pose=N(pose=N(position=N(x=0.,y=0.,z=0.),orientation=N(x=0.,y=0.,z=0.,w=1.))),
        twist=N(twist=N(angular=N(z=0.))))


def scan(stamp):
    return N(header=N(stamp=Stamp(stamp),frame_id='velodyne'),width=16,height=350,
        point_step=22,row_step=352,is_bigendian=False,is_dense=False,data=bytes(123200),
        fields=[N(name=n,offset=o,datatype=d,count=1) for n,o,d in [('x',0,7),('y',4,7),('z',8,7),('ring',16,4)]])


def test_stream_export_is_raw_preserving_and_does_not_replace_first_invalid_window(tmp_path,monkeypatch):
    records=[]
    for i in range(15):
        ns=i*200000000
        # First scan has no previously received pose: first window stays rejected.
        if i:records.append(('/state_estimation',odom(ns),Stamp(ns)))
        records.append(('/velodyne_points',scan(ns),Stamp(ns)))
        records.append(('/registered_scan',scan(ns),Stamp(ns)))
    for topic in exporter.BRIDGE_TOPICS:
        records.append((topic,N(data='{}'),Stamp(3000000000)))
    class Bag:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def get_type_and_topic_info(self):
            values={'/velodyne_points':('sensor_msgs/PointCloud2',15),'/state_estimation':('nav_msgs/Odometry',14)}
            values.update({t:('std_msgs/String',1) for t in exporter.BRIDGE_TOPICS})
            return N(topics={k:N(msg_type=t,message_count=n) for k,(t,n) in values.items()})
        def _get_connections(self,topics):return [N(header={'callerid':'/sensor_coverage_planner/tare_planner_node'})]
        def read_messages(self,topics):return iter(records)
    monkeypatch.setitem(sys.modules,'rosbag',N(Bag=Bag))
    output=tmp_path/'export'
    r=exporter.export(tmp_path/'synthetic.bag',output,project_root=ROOT,start_ns=0,end_ns=120000000000,
        logical_output_relative='results/synthetic_only/export')
    assert r['selected_windows']==3 and r['rejected_windows']==1 and r['exported_windows']==2
    manifest=json.loads((output/'windows_manifest.json').read_text())
    assert manifest['windows'][0]['window_id']=='second_001'
    assert manifest['windows'][0]['frames'][-1]['frame_order']==5
    assert manifest['windows'][0]['input']['path'].startswith('results/synthetic_only/')
    with np.load(output/'second_001.npz') as payload:
        assert set(payload.files)=={'raw_pointcloud_bytes','world_from_sensor'}
        assert payload['raw_pointcloud_bytes'].shape==(5,123200)
        assert not payload['raw_pointcloud_bytes'].any()
    assert r['single_original_control_publisher'] and not r['timestamp_equality_is_transform_or_payload_proof']


def test_future_pose_cannot_be_used():
    with pytest.raises(ValueError,match='causal'):
        exporter.commanded_pose(odom(200),100)


def test_real_ros_bytes_callerid_and_atomic_json_validation(tmp_path):
    expected='/sensor_coverage_planner/tare_planner_node'
    assert exporter.ros_header_text(expected.encode())==expected
    assert exporter.ros_header_text(expected)==expected
    with pytest.raises(ValueError):exporter.ros_header_text(None)
    with pytest.raises(UnicodeDecodeError):exporter.ros_header_text(b'\xff')
    output=tmp_path/'must_not_be_partial.json'
    with pytest.raises(TypeError):exporter.write(output,dict(bytes=b'not_decoded'))
    assert not output.exists()
