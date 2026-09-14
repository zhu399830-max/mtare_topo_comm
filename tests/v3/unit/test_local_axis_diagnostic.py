import numpy as np
from mtare_topo.evaluation.local_axis_diagnostic import clip_axis_to_ball,sampled_hausdorff


def test_clipping_retains_crossing_with_all_controls_outside():
    s=clip_axis_to_ball([[-20,0,0],[20,0,0],[30,0,0]],[0,0,0])
    assert len(s)==1
    np.testing.assert_allclose(s[0],[[-10,0,0],[10,0,0]])


def test_reversal_identical_and_separate_parallel_not_identical():
    a=clip_axis_to_ball([[-4,0,0],[0,0,0],[4,0,0]],[0,0,0])
    b=clip_axis_to_ball([[4,0,0],[0,0,0],[-4,0,0]],[0,0,0])
    assert sampled_hausdorff(a,b)==0
    b=clip_axis_to_ball([[-4,2,0],[0,2,0],[4,2,0]],[0,0,0])
    assert sampled_hausdorff(a,b)==2


def test_tangent_and_outside_have_no_positive_length():
    assert clip_axis_to_ball([[-20,10,0],[20,10,0],[30,10,0]],[0,0,0])==[]
    assert clip_axis_to_ball([[12,0,0],[13,0,0],[14,0,0]],[0,0,0])==[]
