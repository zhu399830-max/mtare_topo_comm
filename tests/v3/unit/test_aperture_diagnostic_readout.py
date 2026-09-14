from dataclasses import replace
import numpy as np
import pytest
from mtare_topo.teacher.aperture_component_contract import partition_boundary
from mtare_topo.teacher.aperture_sparse_support import SparseBoundarySupport
from mtare_topo.teacher.aperture_diagnostic_readout import diagnostic_readout,reference_seed_components
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere


def support(assignment,n):
    a=np.asarray(assignment,dtype=int)
    return SparseBoundarySupport(np.ones(len(a),bool),a,np.bincount(a[a>=0],minlength=n),int((a<0).sum()))


def test_unknown_bridge_forms_set_not_two_instances():
    p=partition_boundary([(1,),(0,2),(1,)],[1,-1,1],[1,-1,1])
    out=diagnostic_readout(p,support([0,1,1],2),(0,1))
    assert out['supported_isolated_components']==0 and out['supported_unresolved_sets']==1
    assert out['groups'][0]['independent_instance_count'] is None
    assert out['groups'][0]['support_ray_count']==3


def test_duplicate_references_do_not_change_candidates():
    p=partition_boundary([(1,),(0,2),(1,)],[1,0,1],[1,0,1]);s=support([0,0,1,-1],2)
    a=diagnostic_readout(p,s,(0,1));b=diagnostic_readout(p,s,(1,0,0,None))
    assert a['supported_isolated_components']==b['supported_isolated_components']==2
    assert b['groups'][0]['reference_seed_indices']==[1,2]
    assert b['unmapped_reference_seed_indices']==[3] and not b['training_labels_qualified']


def test_unsupported_is_not_negative_and_inconsistent_counts_reject():
    p=partition_boundary([(),()],[1,1],[-1,-1]);s=support([0],2)
    out=diagnostic_readout(p,s)
    assert out['groups'][1]['status']=='NO_POSITIVE_OBSERVATION_EVIDENCE'
    assert out['complete_instance_count'] is None
    with pytest.raises(ValueError):diagnostic_readout(p,replace(s,component_ray_counts=np.array([3,0])))


def test_reference_seed_outside_window_and_unknown_remain_unmapped():
    v,f,a=octahedral_sphere(2);p=partition_boundary(a,np.ones(len(f),int),np.full(len(f),-1))
    assert reference_seed_components([[10.,0,0],[0,10,0],[9.,0,0]],v,f,p)==(0,0,None)
    q=partition_boundary(a,np.full(len(f),-1),np.full(len(f),-1))
    assert reference_seed_components([[10.,0,0]],v,f,q)==(None,)
