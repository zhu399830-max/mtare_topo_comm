import hashlib
import numpy as np
import pytest
from mtare_topo.data.development_partition_handoff import encode_partitions,decode_partitions
from mtare_topo.data.development_partition_handoff import produce_partitions


def fixture():
    points=np.array([[1,0,0],[2,0,0]],np.float32);indices=np.array([0,720],np.int64)
    source=dict(task='example',source_sequence_id=1,frame_rows=[0,1,2,3,4])
    payload=encode_partitions(points,indices,dict(r1=np.array([0,1]),r2=np.array([0,0])),source=source,cache_sha256='a'*64)
    args=dict(expected_sha256=hashlib.sha256(payload).hexdigest(),expected_source=source,
        expected_cache_sha256='a'*64,points_xyz_m=points,source_flat_ray_index=indices)
    return payload,args


def test_two_methods_roundtrip_without_source_loss():
    payload,args=fixture();result=decode_partitions(payload,**args)
    assert result['r1'].tolist()==[0,1] and result['r2'].tolist()==[0,0]
    assert not result['r2'].flags.writeable


@pytest.mark.parametrize('damage',['bytes','source','cache','points','rays'])
def test_same_shape_wrong_identity_or_coordinates_rejected(damage):
    payload,args=fixture()
    if damage=='bytes':payload+=b'x'
    if damage=='source':args['expected_source']['source_sequence_id']=2
    if damage=='cache':args['expected_cache_sha256']='b'*64
    if damage=='points':args['points_xyz_m'][0,0]+=.001
    if damage=='rays':args['source_flat_ray_index'][0]=1
    with pytest.raises(ValueError):decode_partitions(payload,**args)


def test_missing_representation_cannot_silently_fallback():
    _,args=fixture()
    with pytest.raises(ValueError,match='both'):
        encode_partitions(args['points_xyz_m'],args['source_flat_ray_index'],{'r1':np.array([0,0])},
                          source=args['expected_source'],cache_sha256='a'*64)


def test_native_producer_consumes_same_compact_roi(monkeypatch):
    from test_development_compact_blocks import fixture as compact_fixture
    import mtare_topo.representation.gse_spg_extractor as native
    payload,row,source=compact_fixture()
    seen=[]
    def extractor(points,frames,**kwargs):
        seen.append(points.copy())
        return dict(original_point_to_component=np.zeros(len(points),np.int64))
    monkeypatch.setattr(native,'extract_spg',extractor)
    packet=produce_partitions(payload,source=source,cache_sha256=row['sha256'],
                              geof_backend=None,partition_backend=None)
    expected=np.array([[1,0,0],[2,0,0]],np.float32)
    result=decode_partitions(packet,expected_sha256=hashlib.sha256(packet).hexdigest(),
        expected_source=source,expected_cache_sha256=row['sha256'],points_xyz_m=expected,
        source_flat_ray_index=np.array([0,720]))
    np.testing.assert_array_equal(seen[0],expected)
    assert result['r2'].tolist()==[0,0]
