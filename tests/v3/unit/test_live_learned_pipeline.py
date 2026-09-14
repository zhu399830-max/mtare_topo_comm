import socket
import struct
import numpy as np
import pytest
from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline
from mtare_topo.integration.local_geometry_inference import send,receive

def test_local_transport_roundtrip_and_oversize_rejection():
    a,b=socket.socketpair()
    with a,b:
        send(a,dict(axes=np.ones((2,3,3)),source_frame_keys=['a','b']))
        result=receive(b);assert result['axes'].shape==(2,3,3)
        a.sendall(struct.pack('!I',9*1024*1024))
        with pytest.raises(ValueError,match='bound'):receive(b)

def test_live_prediction_drives_geometry_without_fitting(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('nonlearning fit must not run')
    monkeypatch.setattr('mtare_topo.integration.live_geometry_pipeline.full_sensor_local_fit',forbidden)
    calls=[]
    def predictor(**r):
        calls.append(r)
        return dict(axes=np.array([[[-8,0,0],[0,0,0],[8,0,0]]]),probabilities=[1.],
            checkpoint_sha256='a'*64,source_frame_keys=r['source_frame_keys'],timestamp=r['timestamp'])
    p=LiveGeometryPipeline(composition_policy={},anchor_spacing_m=4,lookahead_m=4,rigid_motion=True,
        predictor=predictor,checkpoint_sha256='a'*64)
    for i in range(5):
        result=p.push(np.full((16,720),10.),np.ones((16,720)),sensor_to_map=np.eye(4),stamp_sec=float(i),source_key=str(i),coordinate_frame='map')
    assert len(calls)==1 and np.array_equal(calls[0]['rotation'][-1],np.eye(3))
    assert result['geometry']['frontend']=='frozen_learned_geometry'
    assert len(result['decision']['proposals'])==2
    assert all(x['geometry_source_kind']=='model_prediction' for x in result['decision']['proposals'])
    assert result['control_published'] is False
    p.predictor=lambda **r:dict(predictor(**r),timestamp=999.)
    with pytest.raises(ValueError,match='source mismatch'):
        p.push(np.full((16,720),10.),np.ones((16,720)),sensor_to_map=np.eye(4),stamp_sec=5.,source_key='5',coordinate_frame='map')
    assert p.frames[-1][0]==4. and len(p.graph.decisions)==1
