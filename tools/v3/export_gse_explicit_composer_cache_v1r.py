#!/usr/bin/env python3
"""V1R: export cache with explicit binary16 derived-probability parity."""

from __future__ import annotations

import numpy as np

import export_gse_explicit_composer_cache_v1 as v1
from mtare_topo.data.gse_explicit_composer_cache_parity import (
    compare_explicit_cache_to_sealed_development,
)


def _development_parity(arrays, sealed_path):
    if not sealed_path.is_file():
        raise RuntimeError(f"sealed development prediction missing: {sealed_path}")
    with np.load(sealed_path, allow_pickle=False) as sealed:
        return compare_explicit_cache_to_sealed_development(arrays, sealed)


v1._development_parity = _development_parity


if __name__ == "__main__":
    raise SystemExit(v1.main())
