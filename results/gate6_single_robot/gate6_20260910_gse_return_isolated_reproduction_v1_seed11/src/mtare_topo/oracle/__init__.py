"""Ground-truth development-map baselines.

These modules are evaluator/oracle boundaries.  They must never be imported by
the learned online method in a formal comparison.
"""

from .layered_gt_map import (
    GTMapOraclePrediction,
    LayeredGTMapConfig,
    LayeredGTMapOracle,
    LocalLayerEvidence,
    load_binary_xyz_ply,
)

__all__ = [
    "GTMapOraclePrediction",
    "LayeredGTMapConfig",
    "LayeredGTMapOracle",
    "LocalLayerEvidence",
    "load_binary_xyz_ply",
]
