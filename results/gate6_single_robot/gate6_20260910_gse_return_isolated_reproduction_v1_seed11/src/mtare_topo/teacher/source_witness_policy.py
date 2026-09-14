"""Separate source evidence from scan validity; no label qualification.

The diagnostic mask is deliberately distinct from the certified mask. An
agreement between approximate references is not a uniqueness certificate.
Consumers must explicitly select a mode, never fall back to reported IDs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceWitnessPolicy:
    diagnostic_singleton: np.ndarray
    certified_singleton: np.ndarray
    reported_owner: np.ndarray

    def owners(self, *, mode):
        if mode not in ('diagnostic', 'certified'):
            raise ValueError('explicit diagnostic or certified mode required')
        allowed = self.diagnostic_singleton if mode == 'diagnostic' else self.certified_singleton
        return np.where(allowed, self.reported_owner, -1)


def source_witness_policy(*, valid, reported, possible_surface,
                          uncertain_operands, surface_uniqueness_certified):
    """Build teacher-only masks from aligned NxK source audit arrays.

    Caller binds rows and operand columns to sealed scan/codebook identities.
    This pure function neither edits scans nor establishes that binding.
    """
    v, r, p, u, c = map(np.asarray, (valid, reported, possible_surface,
                                    uncertain_operands, surface_uniqueness_certified))
    if (r.ndim != 2 or r.shape[1] == 0 or p.shape != r.shape or u.shape != r.shape
            or v.shape != (r.shape[0],) or c.shape != v.shape
            or any(x.dtype != np.bool_ for x in (v, r, p, u, c))):
        raise ValueError('aligned explicit boolean ray and source masks required')
    if np.any(r.any(axis=1) != v):
        raise ValueError('reported source presence must match scan validity')
    eligible = v & (r.sum(axis=1) == 1) & (p.sum(axis=1) == 1)
    eligible &= (r == p).all(axis=1) & ~u.any(axis=1)
    if np.any(c & ~eligible):
        raise ValueError('certificate contradicts source evidence')
    arrays = (eligible.copy(), (eligible & c).copy(), np.argmax(r, axis=1))
    for array in arrays:
        array.flags.writeable = False
    return SourceWitnessPolicy(*arrays)
