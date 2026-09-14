import numpy as np
import pytest
from mtare_topo.teacher.gse_ordered_interface_crossings_v1 import ordered_crossings


def run(ray, interface, t, returns=(5., 5.), roi=None):
    return ordered_crossings(ray_ids=np.array(ray, dtype=np.int64),
        interface_ids=np.array(interface, dtype=np.int64),
        hit_t=np.array(t, dtype=np.float32),
        first_return=np.array(returns, dtype=np.float32),
        valid=np.ones(len(returns), dtype=bool),
        inside_roi=np.ones(len(t), dtype=bool) if roi is None else np.array(roi))


def test_order_and_duplicate_triangle_hits():
    out = run([0, 0, 0], [2, 1, 1], [3, 1, 1])
    assert out['rays'][0]['ordered_adjacent_crossings'] == [dict(
        from_interface=1, to_interface=2, from_t=1., to_t=3.)]
    assert out['semantic_label'] is None and not out['training_eligible']


def test_different_rays_do_not_connect():
    assert all(not r['ordered_adjacent_crossings']
               for r in run([0, 1], [1, 2], [1, 2])['rays'])


def test_return_boundary_and_occlusion_excluded():
    out = run([0]*4, [1, 2, 3, 4], [1, 5, 6, 0])
    assert len(out['rays'][0]['crossing_groups']) == 1


def test_coincidence_is_not_an_order_and_not_bridged():
    out = run([0]*4, [1, 2, 3, 4], [1, 2, 2, 3])
    assert not out['rays'][0]['ordered_adjacent_crossings']


def test_roi_and_permutation():
    a = run([0, 0, 0], [1, 2, 3], [1, 2, 3], roi=[True, True, False])
    b = run([0, 0, 0], [3, 2, 1], [3, 2, 1], roi=[False, True, True])
    assert a == b


def test_bad_index_fails():
    with pytest.raises(ValueError):
        run([2], [1], [1])
