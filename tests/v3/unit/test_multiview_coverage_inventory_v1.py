from copy import deepcopy
import pytest
from mtare_topo.data.multiview_coverage_inventory_v1 import expand_interval


def fixture():
    return dict(traversal_id='S01_flat_tree_small_C01:edge_0000:d0',variants=[
        dict(variant=v,task='S01_flat_tree_small_C01__'+v,source_sequence_ids=[10,11,12],sequence_rows=[0,1,2],
             frame_rows=[list(range(i,i+5)) for i in range(3)],decision_arc_m=[4.,5.,6.])
        for v in ('ellipse','rounded_rectangle','c1_mixed')])


def test_all_windows_kept_no_label_invention_or_order_dependency():
    a=fixture();rows=expand_interval(a,'fit')
    assert len(rows)==9 and {r['decision_index'] for r in rows}=={0,1,2}
    assert {r['label_status'] for r in rows}=={'NOT_EVALUATED'}
    b=deepcopy(a);b['variants'].reverse()
    assert expand_interval(b,'fit')==rows


def test_misaligned_variant_and_broken_history_rejected():
    a=fixture();a['variants'][0]['source_sequence_ids'][1]=99
    with pytest.raises(ValueError,match='correspondence'):expand_interval(a,'fit')
    a=fixture();a['variants'][0]['frame_rows'][1][0]=9
    with pytest.raises(ValueError,match='history'):expand_interval(a,'fit')
