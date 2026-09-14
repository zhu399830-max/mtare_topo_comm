import numpy as np
from mtare_topo.teacher.gse_superellipse_constraints_v1 import finite_center_probe
from test_gse_superellipse_constraints_v1 import contour


def test_probe_recomputes_nonlinear_residual_at_one_metre_without_certifying_world():
    points=contour(np.arange(80)*2*np.pi/80+.01,p=6.)
    result=finite_center_probe(points,center_uv_m=[0,0],half_axes_m=[4,3],exponent=6.)
    assert not result['qualified_label']
    assert len(result['probes'])==2
    for p in result['probes']:
        assert p['status']=='FINITE_LOCAL_SECTION_PROBE'
        assert abs(p['center_shift_m']-1)<1e-12
        c,s=np.cos(p['angle_rad']),np.sin(p['angle_rad'])
        delta=points-p['center_uv_m']
        uv=np.column_stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]))
        residual=np.sum(np.abs(uv/p['half_axes_m'])**p['exponent'],axis=1)-1
        np.testing.assert_allclose(p['nonlinear_residual_rms'],np.sqrt(np.mean(residual**2)))


def test_sparse_wall_probe_keeps_invalid_outcomes_without_clipping_parameters():
    points=contour(np.linspace(-.03,.03,80),p=6.)
    a=finite_center_probe(points,center_uv_m=[0,0],half_axes_m=[4,3],exponent=6.)
    b=finite_center_probe(points,center_uv_m=[0,0],half_axes_m=[4,3],exponent=6.)
    assert a==b
    assert len(a['probes'])==2
    assert all(p['status'] in ('FINITE_LOCAL_SECTION_PROBE','OUTSIDE_SOURCE_PARAMETER_DOMAIN',
                              'NUMERICAL_OR_PARAMETER_LIMIT') for p in a['probes'])
