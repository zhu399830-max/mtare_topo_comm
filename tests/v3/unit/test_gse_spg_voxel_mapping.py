import numpy as np
import pytest
from mtare_topo.representation.gse_spg_voxel_mapping import map_spg_voxels


def test_min_origin_first_occurrence_and_frames():
    p = np.array([[.041,0,0],[-.02,0,0],[-.019,0,0]],dtype='float32')
    r = map_spg_voxels(p,np.array([4,0,2]))
    assert r.point_to_voxel.tolist() == [0,1,1]
    assert r.counts.tolist() == [1,2]
    assert r.frame_counts.tolist() == [[0,0,0,0,1],[1,0,1,0,0]]
    assert np.array_equal(r.frame_counts.sum(1),r.counts)


def test_duplicates_all_accounted():
    p = np.ones((5,3),dtype='float32')
    r = map_spg_voxels(p,np.arange(5))
    assert r.counts.tolist() == [5]
    assert r.frame_counts.tolist() == [[1]*5]
    assert np.array_equal(r.centers_m,p[:1])


def test_empty():
    r = map_spg_voxels(np.empty((0,3),dtype='float32'),np.array([],dtype=int))
    assert r.centers_m.shape == (0,3) and r.frame_counts.shape == (0,5)


@pytest.mark.parametrize('p,f', [([[float('nan'),0,0]],[0]), ([[0,0,0]],[5]), ([[0,0,0]],[-1])])
def test_invalid(p,f):
    with pytest.raises(ValueError):
        map_spg_voxels(np.array(p,dtype='float32'),np.array(f))


def test_overflow_rejected():
    with pytest.raises(OverflowError):
        map_spg_voxels(np.array([[0,0,0],[1e30,0,0]],dtype='float32'),np.array([0,0]))
