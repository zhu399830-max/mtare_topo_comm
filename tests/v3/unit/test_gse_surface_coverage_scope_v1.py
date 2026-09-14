import pytest
from mtare_topo.data.gse_surface_coverage_scope_v1 import plan_current_xyz


def header():
    return dict(shape=[5000, 3], chunks=[4096, 3], dtype='<f8', zarr_format=2, order='C')


def test_current_only_selection_and_chunk_collateral_count():
    r = plan_current_xyz(header(), list(range(15))+[4999], 5000)
    assert r['chunk_keys'] == ['0.0', '1.0']
    assert r['decoded_rows_including_collateral'] == 5000
    assert r['decoded_padded_bytes'] == 2*4096*3*8
    assert len(r['selected_rows']) == 16


@pytest.mark.parametrize('rows', [list(range(15)), [0]*16, list(range(15))+[5000], [True]+list(range(1,16))])
def test_invalid_selected_population_rejected(rows):
    with pytest.raises(ValueError):
        plan_current_xyz(header(), rows, 5000)


def test_wrong_source_precision_rejected():
    h = header(); h['dtype'] = '<f4'
    with pytest.raises(ValueError):
        plan_current_xyz(h, list(range(16)), 5000)
