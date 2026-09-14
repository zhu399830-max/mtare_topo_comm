import numpy as np
from mtare_topo.teacher.gse_center_profile_v1 import profile_center
from test_gse_superellipse_constraints_v1 import contour


def test_profile_is_bounded_and_does_not_assert_label_or_world_equivalence():
    points=contour(np.arange(40)*2*np.pi/40+.01,p=6.)
    result=profile_center(points,half_axes_m=[4.,3.],exponent=6.)
    assert len(result['profiles'])==2
    assert not result['qualified_label'] and not result['scan_equivalence_verified']
    assert result['unshifted_profile']['center_shift_m']==0
    assert result['unshifted_profile']['nfev']<=100
    assert result['unshifted_profile']['residual_rms']<1e-10
    for p in result['profiles']:
        assert abs(p['center_shift_m']-1)<1e-12
        assert p['nfev']<=100 and p['residual_rms']>1e-3
        assert 2<=p['exponent']<=16
