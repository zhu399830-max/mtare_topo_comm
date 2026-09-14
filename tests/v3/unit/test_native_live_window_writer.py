from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/v3'))
from native_structure_live_input_node import write_window
from tests.v3.unit.test_frozen_structural_token_worker import fixture
from mtare_topo.integration.frozen_structural_se3_worker import prepare_window


def parts():
    entry,raw,poses=fixture()
    rows=[dict(payload=raw[i].tobytes(),pose=poses[i],error=None,
        frame=f,evidence={'physical_pose_verified':False,'source_key':f['pose_source_key']})
        for i,f in enumerate(entry['frames'])]
    binding=dict(status='EXPLICIT_SOURCE_BOUND',raw_source_keys=[f['source_key'] for f in entry['frames']],
        registered_source_keys=[f'registered_scan:{i}' for i in range(5)],physical_pose_verified=False)
    return rows,binding,raw,poses


def test_exact_live_window_can_use_existing_full_se3_decoder(tmp_path):
    output=tmp_path/'live';output.mkdir()
    rows,binding,raw,poses=parts()
    request=write_window(tmp_path,output,0,rows,binding,segment='live')
    manifest=json.loads((tmp_path/request['manifest']['path']).read_text())
    entry=manifest['windows'][0]
    with np.load(tmp_path/entry['input']['path'],allow_pickle=False) as data:
        np.testing.assert_array_equal(data['raw_pointcloud_bytes'],raw)
        np.testing.assert_array_equal(data['world_from_sensor'],poses)
        student,context,refs,rotation,audits=prepare_window(entry,data['raw_pointcloud_bytes'],data['world_from_sensor'])
    assert refs==tuple(binding['raw_source_keys'])
    assert context.sequence_id=='live'
    assert len(audits)==5
    with pytest.raises(FileExistsError):write_window(tmp_path,output,0,rows,binding,segment='live')


@pytest.mark.parametrize('fault',['unknown','source','pose'])
def test_missing_evidence_never_writes_substitute_window(tmp_path,fault):
    output=tmp_path/'live';output.mkdir()
    rows,binding,_,_=parts()
    if fault=='unknown':binding['status']='UNBOUND'
    if fault=='source':binding['raw_source_keys'][0]='different'
    if fault=='pose':rows[0]['error']='MISSING_CAUSAL_POSE'
    with pytest.raises(ValueError):write_window(tmp_path,output,0,rows,binding,segment='live')
    assert not list(output.iterdir())


def test_live_callbacks_prefetch_exact_registered_window_without_control(tmp_path,monkeypatch):
    import hashlib
    import types
    import native_structure_live_input_node as node
    from types import SimpleNamespace as NS
    callbacks={};requests=[];shutdown=[];published=[]
    class Stamp:
        def __init__(self,n):self.n=n
        def to_nsec(self):return self.n
    class Subscription:
        def __init__(self,topic,typ,callback,**kw):callbacks[topic]=callback
        def unregister(self):pass
    entry,raw,_=fixture()
    def spin():
        for i in range(11):
            stamp=1000000000+i*200000000
            odom=NS(_connection_header={'callerid':'/vehicleSimulator'},
                header=NS(stamp=Stamp(stamp),frame_id='map'),child_frame_id='sensor',
                pose=NS(pose=NS(position=NS(x=float(i),y=0.,z=1.),orientation=NS(x=0.,y=0.,z=0.,w=1.))),
                twist=NS(twist=NS(angular=NS(z=0.))))
            callbacks['/state_estimation'](odom)
            original=entry['frames'][0]['pointcloud_layout']
            message=NS(**{k:v for k,v in original.items() if k!='fields'},
                fields=[NS(**f) for f in original['fields']],data=raw[0].tobytes(),
                header=NS(seq=i,stamp=Stamp(stamp),frame_id='velodyne'),
                _connection_header={'callerid':'/gazebo'})
            callbacks['/velodyne_points'](message)
            if i<6:continue
            regstamp=stamp+10
            callbacks['/registered_scan'](NS(header=NS(seq=i-6,stamp=Stamp(regstamp)),
                _connection_header={'callerid':'/vehicleSimulator'}))
            echo=dict(schema_version='native_scan_source_v1',session_id='synthetic',
                source_seq=i,source_stamp_ns=stamp,registered_seq=i-6,registered_stamp_ns=regstamp,
                raw_topic='/velodyne_points',registered_topic='/registered_scan',
                source_frame='velodyne',registered_frame='map',raw_frame_matches=True,physical_pose_verified=False)
            callbacks['/native_structure/scan_sources'](NS(data=json.dumps(echo),
                _connection_header={'callerid':'/vehicleSimulator'}))
    rospy=types.ModuleType('rospy');rospy.Subscriber=Subscription;rospy.spin=spin
    rospy.Publisher=lambda topic,typ,**kw:NS(publish=lambda message:published.append((topic,message.data)))
    rospy.init_node=lambda *a,**kw:None;rospy.signal_shutdown=lambda msg:shutdown.append(msg)
    monkeypatch.setitem(sys.modules,'rospy',rospy)
    for package,name in [('nav_msgs.msg','Odometry'),('sensor_msgs.msg','PointCloud2'),('std_msgs.msg','String')]:
        module=types.ModuleType(package);setattr(module,name,(lambda data:NS(data=data)) if name=='String' else object);monkeypatch.setitem(sys.modules,package,module)
    class Client:
        def __init__(self,*a,**kw):pass
        def request(self,request):
            requests.append(request)
            return dict(response={'status':'PROCESSED'},client_received_monotonic_ns=123,
                usable_for_native_advice=False)
    monkeypatch.setattr(node,'StructuralTokenClient',Client);monkeypatch.setattr(node,'ROOT',tmp_path)
    manifest=tmp_path/'producer.json'
    manifest.write_text(json.dumps(dict(
        original_sha256='ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e',
        output_sha256='72b5ccc1bb631bdcf89094010cb0188d43f0ecb6df63e61ad21fedf31fa2b7ae',scan_arithmetic_changed=False)))
    output=tmp_path/'live'
    monkeypatch.setattr(sys,'argv',['node','--output',str(output),'--socket',str(tmp_path/'unused.sock'),
        '--segment','live','--producer-manifest',str(manifest),'--producer-manifest-sha256',
        hashlib.sha256(manifest.read_bytes()).hexdigest(),'--max-raw-frames','11','--max-windows','1'])
    assert node.main()==0 and not shutdown and len(requests)==1
    bound=json.loads((output/'prefetch_000000_binding.json').read_text())
    assert bound['raw_source_keys']==[f'/velodyne_points:{1000000000+i*200000000}' for i in range(6,11)]
    summary=json.loads((output/'summary.json').read_text())
    assert summary['counts']['raw']==11 and summary['counts']['windows']==1
    assert summary['counts']['processed']==1 and not summary['control_published']
    assert len(published)==1 and published[0][0]=='/native_structure/token_ready'
