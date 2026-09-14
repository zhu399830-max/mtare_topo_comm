"""Run with the existing Open3D environment; no pytest install required."""
from types import SimpleNamespace
import numpy as np
from mtare_topo.teacher.gse_interface_caster_v1 import SourceInterfaces, replay_interfaces


def plane(x, z=0.):
    return SimpleNamespace(vertices_xyz_m=np.array([[x,-2,z-2],[x,2,z-2],[x,0,z+2.]]),
                           triangle_vertex_indices=np.array([[0,1,2]]))


def run(sources, distance=5.):
    return replay_interfaces(sources, origins=np.array([[0.,0.,0.]]),
        directions=np.array([[2.,0.,0.]]), first_return=np.array([distance], np.float32),
        valid=np.array([True]), source_frame_indices=np.array([4]), roi_center=np.zeros(3))


def check():
    a = SourceInterfaces('a', plane(1), {0: 0})
    b = SourceInterfaces('b', plane(3), {0: 1})
    out = run([a, b])
    assert len(out['rays'][0]['ordered_adjacent_crossings']) == 1
    assert all(r['source_frame_index'] == 4 for r in out['raw_interface_intersections'])
    assert out == run([b, a]), 'source order must not change evidence'
    assert not run([a, b], 2.)['rays'][0]['ordered_adjacent_crossings']
    stacked = SourceInterfaces('stacked', plane(3, 5.), {0: 2})
    assert not run([a, stacked])['rays'][0]['ordered_adjacent_crossings']
    coincident = SourceInterfaces('coincident', plane(1), {0: 3})
    same = run([a, coincident, b])
    assert len(same['rays'][0]['crossing_groups'][0]['interface_ids']) == 2
    assert not same['rays'][0]['ordered_adjacent_crossings']
    outside = SourceInterfaces('outside', plane(11), {0: 4})
    assert not run([a, outside], 20.)['rays'][0]['ordered_adjacent_crossings']
    try:
        run([a, SourceInterfaces('bad', plane(3), {0: 0})])
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous ownership accepted')
    assert out['semantic_label'] is None and out['training_eligible'] is False
    print('PASS: actual backend ordering, permutation, occlusion, stacked, '
          'coincident, ROI, ownership and causal provenance checks')


if __name__ == '__main__':
    check()
