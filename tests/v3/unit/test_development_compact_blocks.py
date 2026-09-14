import copy
import hashlib
import io
import numpy as np
import pytest
from mtare_topo.data.development_compact_blocks import read_compact_points, sensor_layout_partition
from mtare_topo.data.development_compact_blocks import bind_partition_payload
from mtare_topo.data.development_partition_handoff import encode_partitions


def fixture():
    points=np.zeros((1,57600,3),np.float32);valid=np.zeros((1,57600),bool)
    points[0,0]=[1,0,0]; points[0,720]=[2,0,0];points[0,11520]=[11,0,0]
    valid[0,[0,720,11520]]=True
    context=np.ones((1,900,128),np.float32)
    stream=io.BytesIO();np.savez(stream,points_xyz_m=points,frozen_sensor_context=context,valid=valid)
    payload=stream.getvalue()
    source=dict(task='example',source_sequence_id=3,frame_rows=[5,6,7,8,9])
    row=dict(source=dict(source,coordinate_frame='current_sensor',input_file_sha256='a'*64),
             sha256=hashlib.sha256(payload).hexdigest(),frozen_encoder_state_sha256='b'*64)
    return payload,row,source


def test_original_ray_mapping_shared_token_and_no_teacher_output():
    payload,row,source=fixture()
    cache=read_compact_points(payload,manifest_row=row,expected_source=source,encoder_sha256='b'*64)
    assert cache.source_flat_ray_index.tolist()==[0,720]
    assert not cache.points_xyz_m.flags.writeable
    item=sensor_layout_partition(cache)
    assert len(item['blocks'].block_ids)==1
    np.testing.assert_array_equal(item['context'],np.ones((1,128),np.float32))
    assert set(item)=={'blocks','context','binding'}


def test_native_packet_joins_all_three_methods_to_actual_cache():
    payload,row,source=fixture()
    cache=read_compact_points(payload,manifest_row=row,expected_source=source,encoder_sha256='b'*64)
    packet=encode_partitions(cache.points_xyz_m,cache.source_flat_ray_index,
        dict(r1=np.array([0,1]),r2=np.array([0,0])),source=source,cache_sha256=cache.cache_sha256)
    result=bind_partition_payload(cache,packet,expected_sha256=hashlib.sha256(packet).hexdigest(),expected_source=source)
    assert {k:len(v['blocks'].block_ids) for k,v in result.items()}=={'r0':1,'r1':2,'r2':1}
    for v in result.values():np.testing.assert_array_equal(v['context'],np.ones_like(v['context']))


@pytest.mark.parametrize('damage',['source','bytes','encoder','teacher'])
def test_mismatch_rejected(damage):
    payload,row,source=fixture();source=copy.deepcopy(source);encoder='b'*64
    if damage=='source':source['frame_rows'][-1]=10
    if damage=='bytes':payload+=b'changed'
    if damage=='encoder':encoder='c'*64
    if damage=='teacher':source['node_id']='hidden'
    with pytest.raises(ValueError):
        read_compact_points(payload,manifest_row=row,expected_source=source,encoder_sha256=encoder)
