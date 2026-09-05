import numpy as np

from mtare_topo.data.primitive_attachment_observability_sidecar import (
    endpoint_observed_from_gap,
    observable_endpoint_pair_mask,
    pack_endpoint_observed,
    unpack_endpoint_observed,
)


def test_gap_threshold_and_bit_exact_round_trip() -> None:
    gap = np.full((3, 32, 2), np.inf)
    gap[0, 0] = [0.0, 0.25]
    gap[1, 3] = [0.25000001, 0.5]
    observed = endpoint_observed_from_gap(gap)
    assert observed[0, 0].tolist() == [1, 1]
    assert observed[1, 3].tolist() == [1, 0]
    packed = pack_endpoint_observed(observed)
    assert packed.shape == (3, 8)
    np.testing.assert_array_equal(unpack_endpoint_observed(packed), observed)


def test_pair_mask_excludes_same_primitive_and_unknown_endpoints() -> None:
    observed = np.zeros((1, 32, 2), dtype=np.uint8)
    observed[0, 0] = [1, 1]
    observed[0, 1] = [1, 0]
    primitive = np.zeros((1, 32), dtype=np.uint8); primitive[0, :2] = 1
    pair = observable_endpoint_pair_mask(observed, primitive)
    assert pair.shape == (1, 32, 2, 32, 2)
    assert pair[0, 0, 0, 1, 0] == 1
    assert pair[0, 1, 0, 0, 1] == 1
    assert not pair[0, 0, :, 0, :].any()
    assert not pair[0, :, :, 1, 1].any()


def test_observed_endpoint_on_inactive_primitive_fails_closed() -> None:
    observed = np.zeros((32, 2), dtype=np.uint8); observed[4, 0] = 1
    primitive = np.zeros(32, dtype=np.uint8)
    try:
        observable_endpoint_pair_mask(observed, primitive)
    except ValueError as error:
        assert "inactive" in str(error)
    else:
        raise AssertionError("observed inactive endpoint was accepted")
