from copy import deepcopy
import numpy as np
import pytest
from test_gse_reference_exclusion_binding_v1 import fixture
from mtare_topo.data.development_reference_grid import build_bound_reference_grid


def test_same_cpu_algorithm_and_original_grid_hashes():
    bundle,expected,_,binding=fixture()
    actual=build_bound_reference_grid(bundle,expected_binding=binding)
    assert actual.content_sha256==expected.content_sha256
    assert actual.source_geometry_sha256==expected.source_geometry_sha256
    np.testing.assert_array_equal(actual.state,expected.state)


def test_wrong_observation_binding_rejected():
    bundle,_,_,binding=fixture();wrong=deepcopy(binding);wrong['source']['frame_rows'][-1]=9
    with pytest.raises(ValueError,match='binding'):
        build_bound_reference_grid(bundle,expected_binding=wrong)
