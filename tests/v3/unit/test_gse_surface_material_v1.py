import numpy as np
import pytest
from mtare_topo.data.gse_surface_material_v1 import build_observed_material, draw_material


def sample():
    return (np.zeros((5,16,720),np.float32), np.zeros((5,16,720),np.uint8),
            np.zeros((5,3),np.float32), np.zeros(5,np.float32))


def test_empty_observation_does_not_produce_labels_or_free_space():
    arrays, metrics = build_observed_material(*sample())
    assert metrics["surface_patches"] == metrics["free_voxels"] == metrics["occupied_voxels"] == 0
    assert metrics["unknown_voxels"] == 80**3
    assert metrics["opening_labels"] == metrics["anchor_labels"] == 0
    assert not metrics["training_eligible"] and not metrics["physical_connectivity"]
    assert not arrays["points_current_sensor_m"].flags.writeable


def test_far_return_clears_ray_but_does_not_make_roi_surface_or_clip_endpoint():
    data = sample(); data[0][4,2,31] = 20; data[1][4,2,31] = 1
    before = [x.copy() for x in data]
    arrays, metrics = build_observed_material(*data)
    assert metrics["first_returns"] == 1 and metrics["surface_patches"] == 0
    assert metrics["free_voxels"] > 0 and metrics["occupied_voxels"] == 0
    np.testing.assert_allclose(np.linalg.norm(arrays["points_current_sensor_m"][arrays["first_return_valid"]],axis=1),[20],rtol=1e-6)
    for a,b in zip(data,before):np.testing.assert_array_equal(a,b)


def test_causal_relative_motion_applied_to_points_and_ray_origins():
    data = sample(); data[0][0,2,31] = 2; data[1][0,2,31] = 1
    data[2][0] = [1,2,3]; data[3][0] = 70
    arrays, metrics = build_observed_material(*data)
    keep = arrays["first_return_valid"]
    np.testing.assert_array_equal(arrays["ray_origins_current_sensor_m"][keep],[[1,2,3]])
    np.testing.assert_allclose(np.linalg.norm(arrays["points_current_sensor_m"][keep]-[1,2,3],axis=1),[2],rtol=1e-6)
    assert metrics["surface_patches"] == 1 and metrics["normal_supported_patches"] == 0
    assert np.all(arrays["observed_relation_fractions"][...,2] == 1)


@pytest.mark.parametrize("kind", ["range","mask","motion","yaw","dtype","nan"])
def test_bad_source_inputs_fail_without_adaptation(kind):
    data = list(sample())
    if kind=="range":data[0][0,0,0]=51
    if kind=="mask":data[1][0,0,0]=2
    if kind=="motion":data[2][-1,0]=1
    if kind=="yaw":data[3][-1]=1
    if kind=="dtype":data[0]=data[0].astype(np.float64)
    if kind=="nan":data[0][0,0,0]=np.nan
    with pytest.raises(ValueError):build_observed_material(*data)


def test_plot_saves_actual_empty_unknown_case(tmp_path):
    arrays, metrics = build_observed_material(*sample())
    path=tmp_path/"observed.png"
    draw_material(arrays,metrics,path,"合成空观察／非研究成绩")
    assert path.read_bytes().startswith(b"\x89PNG")
