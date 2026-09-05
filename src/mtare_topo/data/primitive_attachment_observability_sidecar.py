"""Compact endpoint-observability targets for locally evidenced relations."""

from __future__ import annotations

import numpy as np


MAXIMUM_SLOTS = 32
ENDPOINTS_PER_PRIMITIVE = 2
ENDPOINT_COUNT = MAXIMUM_SLOTS * ENDPOINTS_PER_PRIMITIVE
PACKED_ENDPOINT_BYTES = ENDPOINT_COUNT // 8
PRIMARY_SUPPORT_BAND_M = 0.25


def endpoint_observed_from_gap(
    endpoint_gap_m: np.ndarray,
    *,
    support_band_m: float = PRIMARY_SUPPORT_BAND_M,
) -> np.ndarray:
    """Convert finite endpoint gaps to the frozen binary evidence target."""

    gap = np.asarray(endpoint_gap_m, dtype=np.float64)
    if gap.shape[-2:] != (MAXIMUM_SLOTS, ENDPOINTS_PER_PRIMITIVE):
        raise ValueError("endpoint gaps must end in [32,2]")
    if support_band_m <= 0 or np.any(gap < 0):
        raise ValueError("support band and endpoint gaps must be nonnegative")
    tolerance = max(1e-9, float(support_band_m) * 1e-7)
    return (gap <= float(support_band_m) + tolerance).astype(np.uint8)


def pack_endpoint_observed(endpoint_observed: np.ndarray) -> np.ndarray:
    values = np.asarray(endpoint_observed, dtype=np.uint8)
    if values.shape[-2:] != (MAXIMUM_SLOTS, ENDPOINTS_PER_PRIMITIVE):
        raise ValueError("endpoint observed mask must end in [32,2]")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("endpoint observed mask must be binary")
    return np.packbits(values.reshape(*values.shape[:-2], ENDPOINT_COUNT), axis=-1, bitorder="little")


def unpack_endpoint_observed(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.shape[-1:] != (PACKED_ENDPOINT_BYTES,):
        raise ValueError("packed endpoint observed mask must end in 8 bytes")
    unpacked = np.unpackbits(values, axis=-1, count=ENDPOINT_COUNT, bitorder="little")
    return unpacked.reshape(*values.shape[:-1], MAXIMUM_SLOTS, ENDPOINTS_PER_PRIMITIVE).astype(np.uint8)


def observable_endpoint_pair_mask(
    endpoint_observed: np.ndarray,
    primitive_mask: np.ndarray,
) -> np.ndarray:
    """Return valid ordered cross-primitive endpoint pairs.

    A pair is a known positive or known negative only when both of its
    candidate endpoints have direct five-frame support.  All other pairs are
    unknown and must contribute neither positive nor negative attachment loss.
    """

    observed = np.asarray(endpoint_observed, dtype=np.uint8)
    primitive = np.asarray(primitive_mask, dtype=np.uint8)
    if observed.shape[:-2] != primitive.shape[:-1] or observed.shape[-2:] != (32, 2) or primitive.shape[-1:] != (32,):
        raise ValueError("endpoint/primitive observability shapes disagree")
    if np.any((observed != 0) & (observed != 1)) or np.any((primitive != 0) & (primitive != 1)):
        raise ValueError("observability masks must be binary")
    if np.any(observed.astype(bool) & ~primitive.astype(bool)[..., :, None]):
        raise ValueError("inactive primitive endpoint cannot be observed")
    endpoint_active = observed.astype(bool)
    result = endpoint_active[..., :, :, None, None] & endpoint_active[..., None, None, :, :]
    eye = np.eye(32, dtype=bool)
    result &= ~eye.reshape((1,) * (result.ndim - 4) + (32, 1, 32, 1))
    return result.astype(np.uint8)


__all__ = [
    "ENDPOINT_COUNT", "ENDPOINTS_PER_PRIMITIVE", "MAXIMUM_SLOTS",
    "PACKED_ENDPOINT_BYTES", "PRIMARY_SUPPORT_BAND_M",
    "endpoint_observed_from_gap", "observable_endpoint_pair_mask",
    "pack_endpoint_observed", "unpack_endpoint_observed",
]
