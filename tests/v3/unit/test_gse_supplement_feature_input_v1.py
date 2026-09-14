import io
import hashlib
import numpy as np
import pytest
from test_gse_surface_feature_input_v1 import fixture
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.data.gse_surface_feature_input_v1 import bind_feature_input as old


def test_variable_population_count_is_frozen_not_inferred():
    arrays={k:v[:7] for k,v in fixture().items()}
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);raw=stream.getvalue()
    kw=dict(expected_sha256=hashlib.sha256(raw).hexdigest(),task='S01_flat_tree_small_C01__ellipse',
        row=6,source_sequence_id=6,frame_rows=[30,31,32,33,34])
    r=bind_feature_input(raw,observation_count=7,**kw)
    assert r.provenance['input_row']==6 and r.range_valid.shape==(5,2,16,720)
    with pytest.raises(ValueError):bind_feature_input(raw,observation_count=8,**kw)
    with pytest.raises(ValueError):old(raw,**kw)
