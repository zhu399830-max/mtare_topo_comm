import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/v3'))
from export_explicit_native_windows import native_window_bindings
from native_explicit_source_capture_case import source_system_command,OLD,NEW
import run_explicit_source_capture as runner
from mtare_topo.integration.explicit_scan_sources import ObservedScan
from mtare_topo.governance_native_structure_capture import validate_explicit_source_card,scope_digest


def rows():
    raw={100+i*10:ObservedScan(i+6,100+i*10,103+i*10) for i in range(5)}
    reg={99+i*10:ObservedScan(i,99+i*10,102+i*10) for i in range(5)}
    echoes=[]
    for i in range(5):
        echoes.append(dict(payload=dict(schema_version='native_scan_source_v1',session_id='session',source_seq=i+6,
            source_stamp_ns=100+i*10,registered_seq=i,registered_stamp_ns=99+i*10,raw_topic='/velodyne_points',
            registered_topic='/registered_scan',source_frame='velodyne',registered_frame='map',raw_frame_matches=True,
            physical_pose_verified=False),receipt_ns=104+i*10,record_ref=f'source:{i}'))
    snapshots=[dict(payload=dict(epoch='session:1',stamp_ns=139,
        source_frame_keys=[f'registered_scan:{99+i*10}' for i in range(5)]),receipt_ns=144,record_ref='snapshot:1')]
    return raw,reg,echoes,snapshots


def test_all_native_epochs_bound_by_callback_not_equal_times():
    raw,reg,echoes,snapshots=rows()
    result=native_window_bindings(raw,reg,echoes[::-1],snapshots,segment='same',source_ref='producer.json')
    assert len(result)==1 and result[0]['status']=='EXPLICIT_SOURCE_BOUND'
    assert result[0]['raw_source_keys'][0]=='/velodyne_points:100'
    assert result[0]['epoch']=='session:1'
    assert not result[0]['usable_for_native_advice']


def test_missing_one_pair_retains_requested_window_as_unknown():
    raw,reg,echoes,snapshots=rows();echoes.pop(1)
    result=native_window_bindings(raw,reg,echoes,snapshots,segment='same',source_ref='producer.json')
    assert len(result)==1 and result[0]['reason']=='MISSING_EXPLICIT_SOURCE'


def test_evidence_cannot_relabel_registered_source():
    raw,reg,echoes,snapshots=rows();echoes[2]['payload']['source_seq']=10
    with pytest.raises(ValueError,match='HEADERS'):
        native_window_bindings(raw,reg,echoes,snapshots,segment='same',source_ref='producer.json')


def test_launcher_changes_only_instrumentation_wrapper_and_control_stays_native(tmp_path):
    command='roslaunch '+OLD+' world_name:=tunnel gazebo_seed:=11'
    assert source_system_command(command)==command.replace(OLD,NEW)
    assert source_system_command('rosbag record /clock')=='rosbag record /clock'
    shell=runner.container_command(tmp_path)[-1]
    assert 'export_explicit_native_windows.py' in shell and 'native_explicit_source_capture_case.py' in shell
    assert 'native_structure_capture_postprocess.py' not in shell
    assert 'prepare_scan_source_overlay.py' in shell and 'vehicleSimulator -- -j2' in shell
    assert runner.TOPICS in shell and 'semantic_topology_global_node' not in shell
    launch=ET.parse(ROOT/'configs/v3/gate6/roslaunch/system_explicit_scan_source.launch').getroot()
    assert launch.find('param').attrib==dict(name='/vehicleSimulator/record_native_scan_sources',type='bool',value='true')
    assert len(launch.findall('include'))==1 and not launch.findall('node')


def test_data_card_fixed_scope_and_missing_source_topic_is_not_optional():
    s=runner.scope();a=dict(status='APPROVED',approved_by='user',authorized_operations=['closed_loop_single'],
        authorized_gates=[6],scope_sha256=scope_digest(s),confirmation_reference='继续推进')
    c=dict(schema_version='gse_explicit_source_capture_card_v1',scope=s,approval=a)
    assert validate_explicit_source_card(c).passed
    s['world']='C08';a['scope_sha256']=scope_digest(s)
    assert not validate_explicit_source_card(c).passed
    topic=json.loads((ROOT/runner.TOPICS).read_text())
    row=next(r for r in topic['topic_contract'] if r['name']=='/native_structure/scan_sources')
    assert row['required'] and row['type']=='std_msgs/String'


def test_setup_environment_does_not_overwrite_create_run_environment():
    # create_run owns environment.json; runner supplements it, never replaces it.
    assert runner.EXECUTION_ENVIRONMENT=='config/execution_environment.json'
    assert runner.EXECUTION_ENVIRONMENT not in {'config/environment.json','config/run_spec.json','config/data_card.json'}
