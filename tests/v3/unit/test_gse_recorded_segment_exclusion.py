import numpy as np
import pytest
from mtare_topo.evaluation.gse_recorded_segment_exclusion import recorded_segment_exclusion as check


def test_separated_touching_crossing_and_invalid_remain_distinct():
    box=np.array([[1.,1.,1.],[2.,2.,2.]])
    origin=np.array([[0.,0.,0.],[0.,1.,1.],[0.,1.5,1.5],[0.,0.,0.]])
    rays=np.array([[1.,0.,0.]]*4);ranges=np.array([3.,1.,3.,3.],np.float32)
    result=check(box,origin,rays,ranges,np.array([True,True,True,False]))
    assert result['separated'].tolist()==[True,False,False,False]
    assert not result['full_sensor_equivalence_certified']


def test_quantized_near_boundary_not_falsely_separated():
    box=np.array([[70.000003,0.,0.],[71.,1.,1.]])
    origin=np.array([[70.0000029,.5,.5]])
    result=check(box,origin,np.array([[0.,0.,1.]]),np.array([.1],np.float32),np.array([True]))
    assert not result['all_recorded_segments_excluded']


def test_reject_missing_source_precision_and_empty_vacuous_proof():
    args=(np.array([[1.,1.,1.],[2.,2.,2.]]),np.zeros((1,3)),np.array([[1.,0.,0.]]),np.ones(1,np.float32),np.ones(1,bool))
    with pytest.raises(ValueError):check(args[0].astype(np.float32),*args[1:])
    with pytest.raises(ValueError):check(args[0],np.empty((0,3)),np.empty((0,3)),np.empty(0,np.float32),np.empty(0,bool))
