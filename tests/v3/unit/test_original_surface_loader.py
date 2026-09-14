import json
from pathlib import Path
import numpy as np
import pytest
from mtare_topo.teacher.gse_original_surface_loader import load_original_surface
from mtare_topo.teacher.gse_original_surface_loader import load_original_ray_points
from mtare_topo.teacher import swept_superellipse_field as current


ROOT = Path(__file__).resolve().parents[3]


def binding():
    return json.loads((ROOT/'configs/v3/gate3/gse_original_surface_source_binding_v1.json').read_text())


def test_original_sampler_isolated_and_arc_order_preserved():
    original = current._sample_operand
    p = dict(primitive_id='synthetic', centerline_xyz_m=[[0.,0,0],[2.,0,0]],
             endpoint_half_axes_m=[[1.,1.],[1.,1.]], endpoint_shape_exponent=[2.,2.])
    r = load_original_surface(ROOT, binding(), p)
    assert current._sample_operand is original
    assert r['triangle_arc_bounds_m'].shape == (5248,2)
    assert np.array_equal(r['triangle_arc_bounds_m'][-128::2], np.zeros((64,2)))
    assert np.array_equal(r['triangle_arc_bounds_m'][-127::2], np.full((64,2),2.))
    assert not r['vertex_arc_m'].flags.writeable


def test_source_drift_rejected_before_execution():
    b = binding()
    b['sources'] = dict.fromkeys(b['sources'], '0'*64)
    with pytest.raises(ValueError, match='source SHA'):
        load_original_surface(ROOT, b, {})


def test_archive_drift_rejected_before_execution():
    b = binding()
    b['archive_sha256'] = '0'*64
    with pytest.raises(ValueError, match='archive SHA'):
        load_original_surface(ROOT, b, {})


def test_original_scene_ray_quantization_is_separate_from_reported_point():
    ranges=np.full((5,16,720),3.,dtype=np.float32)
    origins=np.tile([123456.123456,0.,0.],(5,1))
    rays=load_original_ray_points(ROOT,binding(),ranges,origins,np.zeros(5))
    assert rays['world_points_xyz_m'].shape==(5,11520,3)
    assert not np.array_equal(rays['world_points_xyz_m'],rays['scene_points_xyz_m'])
    # -15 degree elevation, zero azimuth: original float32 direction followed
    # by float64 normalization, then float32 cast without re-normalization.
    d=np.array([np.cos(np.deg2rad(-15)),0,np.sin(np.deg2rad(-15))],dtype=np.float32).astype(np.float64)
    d/=np.linalg.norm(d)
    assert np.array_equal(rays['world_points_xyz_m'][0,0],origins[0]+3*d)
    assert np.array_equal(rays['scene_points_xyz_m'][0,0],origins[0].astype(np.float32).astype(np.float64)+3*d.astype(np.float32).astype(np.float64))
