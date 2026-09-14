import pytest
from mtare_topo.data.gse_structural_supplement_scope_v1 import plan_xyz_rows


def header():
    return dict(shape=[9000,3], chunks=[4096,3], dtype='<f8', zarr_format=2, order='C')


def test_exact_chunks_and_collateral():
    plan = plan_xyz_rows(header(), [1, 4095, 4096, 8999], 9000)
    assert plan['chunk_keys'] == ['0.0','1.0','2.0']
    assert plan['decoded_rows_including_collateral'] == 9000
    assert plan['decoded_padded_bytes'] == 3*4096*3*8


@pytest.mark.parametrize('rows', [[], [True], [3,1], [1,1], [-1], [9000], [1.0]])
def test_bad_rows(rows):
    with pytest.raises(ValueError):
        plan_xyz_rows(header(), rows, 9000)


def test_header_drift():
    h = header(); h['dtype'] = '<f4'
    with pytest.raises(ValueError):
        plan_xyz_rows(h, [1], 9000)
