import numpy as np
import pytest
from tests.v3.unit.test_mesh_interval_exit import mesh
from mtare_topo.teacher.mesh_origin_check import PreparedOriginCheck


def test_inside_and_exact_cache_reuse():
    prepared=PreparedOriginCheck([mesh([(-1,2)]),mesh([(3,4)])])
    result=prepared.check([0,.13,.17])
    assert result.status=='origin_checked' and result.inside==(True,False)
    assert prepared.check([0,.13,.17]) is result
    assert prepared.check([.1,.13,.17]) is not result


@pytest.mark.parametrize('point',[[0,1,1],[0,1-1e-9,.17],[3,.13,.17]])
def test_boundary_packed_boundary_and_outside_not_checked(point):
    assert PreparedOriginCheck([mesh([(-1,2)])]).check(point).status=='needs_reference'


def test_duplicate_shell_cannot_be_binary_origin():
    assert PreparedOriginCheck([mesh([(-1,2),(-1,2)])]).check([0,.13,.17]).reason=='nonbinary_origin_multiplicity'


def test_external_mutation_does_not_change_prepared_snapshot():
    m=mesh([(-1,2)]);prepared=PreparedOriginCheck([m]);digest=prepared.geometry_sha256
    m.vertices_xyz_m[:]+=100
    assert prepared.check([0,.13,.17]).status=='origin_checked'
    assert PreparedOriginCheck([m]).geometry_sha256!=digest


def test_cache_is_bounded_and_invalid_input_rejected():
    prepared=PreparedOriginCheck([mesh([(-1,2)])],cache_capacity=1)
    first=prepared.check([0,.13,.17]);prepared.check([.1,.13,.17])
    assert prepared.check([0,.13,.17])==first
    assert prepared.check([0,.13,.17]) is not first
    with pytest.raises(ValueError):prepared.check([np.nan,0,0])
