import numpy as np

def test_mask_layout_and_actual_bit_drift():
    mask=np.zeros((5,16,720),dtype=np.uint8);mask[2,3,7]=1
    registered=mask.astype(bool).reshape(5,11520)
    assert np.array_equal(registered,mask.astype(bool).reshape(5,11520))
    assert not np.array_equal(registered,mask.astype(bool))
    changed=mask.copy();changed[2,3,7]=0
    assert not np.array_equal(registered,changed.astype(bool).reshape(5,11520))
