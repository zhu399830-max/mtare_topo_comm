"""Cross-process binary16 parity for derived Composer probabilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np


BINARY16_ABSOLUTE_EPSILON = float(np.finfo(np.float16).eps)
DERIVED_PROBABILITY_FIELDS = (
    "token_count_probability", "transport_row_probability",
    "transport_reveal_probability",
)
PRIMARY_EXACT_FIELDS = (
    "global_sequence_index", "token_bearing_deg", "token_existence_logits",
    "token_opening_width_m", "token_vertical_profile_m",
    "token_geometry_uncertainty", "geometry", "observation_uncertainty",
)


def compare_explicit_cache_to_sealed_development(
    arrays: Mapping[str, np.ndarray], sealed: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Require exact primary state and <=1 binary16 epsilon for CUDA probabilities."""

    errors = {}
    unequal = {}
    for name in (*PRIMARY_EXACT_FIELDS, *DERIVED_PROBABILITY_FIELDS):
        actual = np.asarray(arrays[name]); expected = np.asarray(sealed[name])
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(f"sealed development shape/dtype drift: {name}")
        if np.issubdtype(actual.dtype, np.floating):
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        else:
            error = 0.0 if np.array_equal(actual, expected) else float("inf")
        errors[name] = error; unequal[name] = int(np.count_nonzero(actual != expected))
    primary_exact = all(errors[name] == 0.0 for name in PRIMARY_EXACT_FIELDS)
    derived_bounded = all(
        errors[name] <= BINARY16_ABSOLUTE_EPSILON for name in DERIVED_PROBABILITY_FIELDS
    )
    decisions = {
        "token_count_argmax": bool(np.array_equal(
            np.asarray(arrays["token_count_probability"]).argmax(-1),
            np.asarray(sealed["token_count_probability"]).argmax(-1),
        )),
        "transport_row_argmax": bool(np.array_equal(
            np.asarray(arrays["transport_row_probability"]).argmax(-1),
            np.asarray(sealed["transport_row_probability"]).argmax(-1),
        )),
        "transport_reveal_at_0p5": bool(np.array_equal(
            np.asarray(arrays["transport_reveal_probability"]) >= 0.5,
            np.asarray(sealed["transport_reveal_probability"]) >= 0.5,
        )),
    }
    if not primary_exact:
        raise RuntimeError(f"primary explicit state parity failed: {errors}")
    if not derived_bounded:
        raise RuntimeError(f"derived probability exceeds one binary16 epsilon: {errors}")
    if not all(decisions.values()):
        raise RuntimeError(f"derived probability decision parity failed: {decisions}")
    return {
        "all_primary_fields_exact": True,
        "all_derived_probabilities_within_binary16_epsilon": True,
        "all_discrete_decisions_equal": True,
        "binary16_absolute_epsilon": BINARY16_ABSOLUTE_EPSILON,
        "maximum_absolute_error": max(errors.values()),
        "by_field_maximum_absolute_error": errors,
        "by_field_unequal_values": unequal,
        "decisions": decisions,
    }


__all__ = [
    "BINARY16_ABSOLUTE_EPSILON", "DERIVED_PROBABILITY_FIELDS", "PRIMARY_EXACT_FIELDS",
    "compare_explicit_cache_to_sealed_development",
]
