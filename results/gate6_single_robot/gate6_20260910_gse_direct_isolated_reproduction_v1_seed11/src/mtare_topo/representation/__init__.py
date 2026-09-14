"""Gate 2 stable structural representations.

Torch-backed model symbols are loaded lazily so geometry-only sidecars can
import typed NumPy representations without acquiring a hidden Torch runtime
dependency.  The public names and their originating module remain unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GSEModelConfig",
    "GSELossWeights",
    "GeometrySemanticEventNet",
    "gse_multitask_loss",
    "supervised_association_loss",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from mtare_topo.representation import gse_graph

    value = getattr(gse_graph, name)
    globals()[name] = value
    return value
