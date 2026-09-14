import numpy as np
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere,classify_radial_caps
from mtare_topo.teacher.aperture_component_contract import partition_boundary


def classify(axes):
    v,f,a=octahedral_sphere(4)
    s=classify_radial_caps(v,f,axes,[np.arcsin(.2)]*len(axes))
    p=partition_boundary(a,s,np.full_like(s,-1))
    return s,p


def test_overlapping_radial_tubes_share_one_free_component():
    s,p=classify([[0,-1,0],[.05,-1,0]])
    assert len(p.geometry_components)==1
    assert len(p.observed_components)==0
    assert not p.complete_geometry_partition and not p.training_labels_qualified


def test_same_heading_different_height_stays_two_components():
    s,p=classify([[1,0,.5],[1,0,-.5]])
    assert len(p.geometry_components)==2 and not p.unresolved_component_pairs


def test_three_way_geometry_and_duplicate_primitive_invariance():
    axes=[[1,0,0],[-1,0,0],[0,1,0]]
    s,p=classify(axes);duplicate,q=classify([axes[2],axes[0],axes[1],axes[0]])
    assert len(p.geometry_components)==3 and not p.unresolved_component_pairs
    np.testing.assert_array_equal(s,duplicate)
    assert p.geometry_components==q.geometry_components


def test_whole_cell_classification_contains_dense_interior_checks():
    v,f,a=octahedral_sphere(3);axes=np.array([[1.,0.,0.]])
    radius=.35;s=classify_radial_caps(v,f,axes,[radius])
    # Deterministic barycentric points check the bound, not a replacement
    # for it. A cell center alone would not qualify the entire cell.
    for weights in ([.1,.2,.7],[.4,.3,.3],[.8,.1,.1]):
        x=np.einsum('ijk,j->ik',v[f],weights);x/=np.linalg.norm(x,axis=1)[:,None]
        angle=np.arccos(np.clip(x[:,0],-1,1))
        assert np.all(angle[s==1]<radius) and np.all(angle[s==0]>radius)
