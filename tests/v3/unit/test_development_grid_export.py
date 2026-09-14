import hashlib
from copy import deepcopy
import pytest
import numpy as np
from test_gse_reference_exclusion_binding_v1 import fixture
from mtare_topo.data.development_grid_export import encode_grid, decode_grid, export_observation


def test_roundtrip_all_arrays_and_metadata():
    _, grid, _, binding = fixture()
    payload = encode_grid(grid, binding)
    recovered = decode_grid(payload, expected_sha256=hashlib.sha256(payload).hexdigest(), expected_binding=binding)
    for name in ('state', 'free_frame_bits', 'occupied_frame_bits'):
        np.testing.assert_array_equal(getattr(grid, name), getattr(recovered, name))
        assert not getattr(recovered, name).flags.writeable
    assert recovered.content_sha256 == grid.content_sha256
    assert recovered.source_geometry_sha256 == grid.source_geometry_sha256


def test_wrong_bytes_or_identity_rejected():
    _, grid, _, binding = fixture()
    payload = encode_grid(grid, binding)
    with pytest.raises(ValueError, match='hash'):
        decode_grid(payload+b'x', expected_sha256=hashlib.sha256(payload).hexdigest(), expected_binding=binding)
    wrong = deepcopy(binding); wrong['source']['frame_rows'][-1] = 999
    with pytest.raises(ValueError, match='binding'):
        decode_grid(payload, expected_sha256=hashlib.sha256(payload).hexdigest(), expected_binding=wrong)


def test_export_exclusive_and_unknown_preserved(tmp_path):
    bundle, grid, _, binding = fixture()
    entry = export_observation(bundle, expected_binding=binding, output_dir=tmp_path)
    assert entry['whole_region_complete'] is False
    assert entry['state_counts']['0'] == int(np.count_nonzero(grid.state == 0))
    before = (tmp_path/entry['file']).read_bytes()
    with pytest.raises(FileExistsError):
        export_observation(bundle, expected_binding=binding, output_dir=tmp_path)
    assert (tmp_path/entry['file']).read_bytes() == before
