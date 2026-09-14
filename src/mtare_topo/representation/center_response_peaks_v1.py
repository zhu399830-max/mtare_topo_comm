"""Deterministic 26-neighbour regional maxima, with no teacher input.

Equal-valued connected plateaus produce one representative (lowest flat index).
A higher neighbour anywhere on a plateau suppresses the entire plateau.
All above-threshold peaks are returned, including on capacity failure; no top-k.
This is a software decoder, not an assertion that its peaks are real junctions.
"""
from itertools import product
import numpy as np

from .center_response_lattice_v1 import SIZE, lattice, decode_positions

_DELTAS = tuple(d for d in product((-1, 0, 1), repeat=3) if d != (0, 0, 0))


def _neighbours(index):
    x, remainder = divmod(index, SIZE * SIZE)
    y, z = divmod(remainder, SIZE)
    for dx, dy, dz in _DELTAS:
        a, b, c = x + dx, y + dy, z + dz
        if 0 <= a < SIZE and 0 <= b < SIZE and 0 <= c < SIZE:
            yield (a * SIZE + b) * SIZE + c


def regional_peaks(probabilities, offsets_m):
    """Decode fixed >=0.5 probability threshold; retain all peaks above cap32."""
    p = np.asarray(probabilities)
    offsets = np.asarray(offsets_m)
    if (p.shape != (SIZE**3,) or p.dtype.kind != 'f'
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('finite probability for every fixed lattice cell required')
    if (offsets.shape != (SIZE**3, 3) or offsets.dtype.kind != 'f'
            or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > .25 + 1e-10)):
        raise ValueError('finite bounded offset for every cell required')
    domain = lattice().intersects_domain
    active = domain & (p >= .5)
    visited = np.zeros(SIZE**3, dtype=bool)
    peaks, plateau_sizes = [], []
    for start in np.flatnonzero(active):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        count, has_higher = 0, False
        while stack:
            current = stack.pop()
            count += 1
            for neighbour in _neighbours(current):
                if not domain[neighbour]:
                    continue
                if p[neighbour] > p[start]:
                    has_higher = True
                elif p[neighbour] == p[start] and not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        if not has_higher:
            peaks.append(int(start))
            plateau_sizes.append(count)
    ids = np.asarray(peaks, dtype=np.int64)
    return dict(cell_indices=ids, positions_m=decode_positions(ids, offsets[ids]),
                probabilities=p[ids].copy(), plateau_sizes=np.asarray(plateau_sizes),
                threshold=.5, neighbourhood=26, capacity=32,
                capacity_exceeded=len(ids) > 32, truncated=False)
