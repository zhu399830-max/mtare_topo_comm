import pytest

from mtare_topo.data.gse_surface_teacher_scope_v1 import plan_field


def header(field):
    return dict(shape=[100,3] if field=="sensor_xyz_m" else [100,16,720],
        chunks=[100,3] if field=="sensor_xyz_m" else [16,16,720],
        dtype="<f8" if field=="sensor_xyz_m" else "<u2",zarr_format=2,order="C")


def test_pose_collateral_is_not_selected_population():
    p = plan_field(header("sensor_xyz_m"),"sensor_xyz_m",[30,31,32,33,34],100)
    assert p["decoded_rows_including_collateral"]==100 and len(p["selected_rows"])==5
    assert p["chunk_keys"]==["0.0"]


def test_source_code_only_touched_chunks():
    p = plan_field(header("primitive_membership_code"),"primitive_membership_code",[30,31,32,33,34],100)
    assert p["chunk_keys"]==["1.0.0","2.0.0"]
    assert p["decoded_rows_including_collateral"]==32


@pytest.mark.parametrize("rows",[[0,1,2,3,100],[0,1,2,2,3],[4,3,2,1,0],[0,1,2,3]])
def test_bad_rows_fail(rows):
    with pytest.raises(ValueError): plan_field(header("sensor_xyz_m"),"sensor_xyz_m",rows,100)


def test_dtype_drift_rejected():
    h = header("sensor_xyz_m");h["dtype"]="<f4"
    with pytest.raises(ValueError): plan_field(h,"sensor_xyz_m",[0,1,2,3,4],100)


def test_large_pose_source_uses_original_4096_cap():
    h = header("sensor_xyz_m");h["shape"]=[6000,3];h["chunks"]=[4096,3]
    p = plan_field(h,"sensor_xyz_m",[4094,4095,4096,4097,4098],6000)
    assert p["chunk_keys"]==["0.0","1.0"]
    assert p["decoded_rows_including_collateral"]==6000
