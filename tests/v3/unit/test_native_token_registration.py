from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mtare_topo.integration.native_token_registration import SensorRegistrationInput, original_sensor_cloud, verify_native_candidate
from mtare_topo.topology.gse_registration import RegistrationConfig, RegistrationEvidence


def config():
    root = Path(__file__).resolve().parents[3]
    return RegistrationConfig(**json.loads((root/'configs/v3/gate6/native_sensor_registration_v1.json').read_text())['registration'])


def cloud(order=1):
    pose=np.eye(4);pose[0,3]=order
    return SensorRegistrationInput(str(order),'segment',order,order*1000,
        np.eye(3),pose,(0,1,2),'a'*64,'raw:'+str(order*1000))


def test_sensor_frame_transform_and_no_identity_upgrade():
    evidence=RegistrationEvidence(tuple(map(tuple,np.eye(4))),None,None,True,(),
        'synthetic',(),(),0)
    with patch('mtare_topo.integration.native_token_registration.register_local_clouds',return_value=evidence) as call:
        result=verify_native_candidate(cloud(2),cloud(1),config())
    np.testing.assert_array_equal(call.call_args.args[2][:3,3],[1,0,0])
    assert result['accepted']
    assert not any(result[k] for k in ('task_identity_verified','direction_verified','graph_mutated','control_changed','physical_pose_verified'))
    assert result['coordinate_frame']=='current_sensor_to_historical_sensor'


@pytest.mark.parametrize('past', [cloud(2),cloud(3),replace(cloud(1),segment='other')])
def test_no_future_or_cross_segment(past):
    with pytest.raises(ValueError,match='past same-segment'):
        verify_native_candidate(cloud(2),past,config())


def test_empty_returns_remain_unknown_without_icp():
    empty=replace(cloud(2),points_sensor_m=np.empty((0,3)),raw_return_indices=())
    with patch('mtare_topo.integration.native_token_registration.register_local_clouds') as call:
        result=verify_native_candidate(empty,cloud(1),config())
    call.assert_not_called()
    assert not result['accepted'] and result['rejection_reasons']==['insufficient_original_returns']


def test_original_raw_returns_not_model_resampling():
    dtype=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,'offsets':[0,4,8],'itemsize':22})
    points=np.zeros(5600,dtype=dtype)
    points['x'][:5]=[1,10,11,np.nan,0]
    raw=np.tile(np.frombuffer(points.tobytes(),np.uint8),(5,1));poses=np.tile(np.eye(4),(5,1,1))
    buffer=io.BytesIO();np.savez_compressed(buffer,raw_pointcloud_bytes=raw,world_from_sensor=poses)
    layout=dict(point_step=22,is_bigendian=False,height=350,width=16,row_step=352,
        fields=[dict(name=k,offset=o,datatype=7,count=1) for k,o in [('x',0),('y',4),('z',8)]])
    keys=[f'raw:{i}' for i in range(5)]
    archive=SimpleNamespace(run='r',index=[dict(epoch='e',raw_source_frame_keys=keys,input_ref=dict(path='p',sha256='a'*64))],
        _read=lambda path,digest:buffer.getvalue(),_json=lambda path:dict(source=dict(frames=[dict(pointcloud_layout=layout)]*5)))
    record=SimpleNamespace(order=0,record_id='e',source_refs=tuple(keys),segment='s',
        source_frames=[SimpleNamespace(stamp_ns=i) for i in range(5)])
    result=original_sensor_cloud(archive,record)
    assert result.raw_return_indices==(0,1)
    np.testing.assert_array_equal(result.points_sensor_m,[[1,0,0],[10,0,0]])
    with pytest.raises(ValueError):result.points_sensor_m.setflags(write=True)
    layout['is_bigendian']=True
    with pytest.raises(ValueError,match='layout'):original_sensor_cloud(archive,record)
