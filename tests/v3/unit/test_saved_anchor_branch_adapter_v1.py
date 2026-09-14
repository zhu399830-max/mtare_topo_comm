import numpy as np
import pytest
from mtare_topo.topology.saved_anchor_branch_adapter_v1 import convert_saved_prediction


def example():
    d=np.zeros((32,64,3),np.float32);d[:,:,0]=1
    return dict(position_m=np.zeros((32,3),np.float32),presence_logits=np.full(32,-1.,np.float32),
                directions=d,branch_logits=np.full((32,64),-1.,np.float32))


def convert(a):
    return convert_saved_prediction(a,stream_key='opaque',decision_index=0,
        frame_orders=(10,11,12,13,14),input_binding_sha256='a'*64)


def test_frozen_boundary_indices_and_geometry():
    a=example();a['presence_logits'][2]=0;a['branch_logits'][2,7]=0
    before={k:v.copy() for k,v in a.items()};o=convert(a)
    assert [x.anchor_query_index for x in o.anchors]==[2]
    assert [x.branch_query_index for x in o.anchors[0].branches]==[7]
    assert o.anchors[0].confidence==.5
    assert all(np.array_equal(a[k],v) for k,v in before.items())


def test_bad_unselected_slot_not_hidden_by_threshold():
    a=example();a['directions'][0,0]=0
    with pytest.raises(ValueError):convert(a)


def test_no_placeholder_for_empty_selection():
    assert convert(example()).anchors==()
