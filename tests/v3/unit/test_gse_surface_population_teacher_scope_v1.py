import pytest
from mtare_topo.data.gse_surface_population_teacher_scope_v1 import combine_fiveframe_plans


def rows():
    return [dict(frame_rows=list(range(i*5,i*5+5)),source_frame_count=100) for i in range(16)]


def test_shared_pose_chunk_is_counted_once_for_sixteen_windows():
    h=dict(shape=[100,3],chunks=[100,3],dtype='<f8',zarr_format=2,order='C')
    p=combine_fiveframe_plans(h,'sensor_xyz_m',rows())
    assert p['chunk_keys']==['0.0'] and p['decoded_padded_bytes']==2400
    assert len(p['selected_rows'])==80


def test_membership_chunks_union_not_sum_of_window_reads():
    h=dict(shape=[100,16,720],chunks=[16,16,720],dtype='<u2',zarr_format=2,order='C')
    p=combine_fiveframe_plans(h,'primitive_membership_code',rows())
    assert p['chunk_keys']==[f'{i}.0.0' for i in range(5)]
    assert p['decoded_rows_including_collateral']==80


def test_repeated_frames_rejected():
    r=rows();r[1]=r[0]
    h=dict(shape=[100],chunks=[100],dtype='<f8',zarr_format=2,order='C')
    with pytest.raises(ValueError,match='unique'):
        combine_fiveframe_plans(h,'yaw_deg',r)
