"""Cross-traversal residual decomposition for learned structure centers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossTraversalPairs:
    pair_index: np.ndarray
    identity_code: np.ndarray
    identity_names: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.identity_code)
        if self.pair_index.shape != (count, 2) or self.pair_index.dtype != np.int64:
            raise ValueError("cross-traversal pair indices must be int64 [P,2]")
        if self.identity_code.shape != (count,) or self.identity_code.dtype != np.int32:
            raise ValueError("pair identity codes must be int32 [P]")
        if count == 0 or np.any(self.pair_index < 0):
            raise ValueError("cross-traversal pair population must be nonempty")
        if np.any(self.identity_code < 0) or np.any(self.identity_code >= len(self.identity_names)):
            raise ValueError("pair identity code is invalid")


def cross_traversal_pairs(identity: np.ndarray, traversal: np.ndarray) -> CrossTraversalPairs:
    """Enumerate every cross-traversal row pair once, grouped by identity."""

    identity = np.asarray(identity, dtype=str)
    traversal = np.asarray(traversal, dtype=str)
    if identity.ndim != 1 or traversal.shape != identity.shape or len(identity) == 0:
        raise ValueError("identity/traversal rows must be aligned nonempty vectors")
    all_names = tuple(sorted(set(identity.tolist())))
    names = tuple(
        name for name in all_names
        if len(np.unique(traversal[identity == name])) >= 2
    )
    pairs: list[tuple[int, int]] = []
    codes: list[int] = []
    for code, name in enumerate(names):
        rows = np.flatnonzero(identity == name)
        by_trace = {
            trace: rows[traversal[rows] == trace]
            for trace in sorted(set(traversal[rows].tolist()))
        }
        traces = tuple(by_trace)
        for left_index, left_name in enumerate(traces):
            for right_name in traces[left_index + 1:]:
                for left in by_trace[left_name]:
                    for right in by_trace[right_name]:
                        pairs.append((int(left), int(right)))
                        codes.append(code)
    return CrossTraversalPairs(
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(codes, dtype=np.int32),
        names,
    )


def pair_distance_metrics(
    centers_xyz_m: np.ndarray,
    pairs: CrossTraversalPairs,
    *,
    threshold_m: float = 4.0,
) -> dict[str, object]:
    """Report pair and identity-macro center agreement at a fixed radius."""

    centers = np.asarray(centers_xyz_m, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.all(np.isfinite(centers)):
        raise ValueError("center coordinates must be finite [N,3]")
    if np.any(pairs.pair_index >= len(centers)) or not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("pair center population or threshold is invalid")
    delta = centers[pairs.pair_index[:, 0]] - centers[pairs.pair_index[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    identity_rows = []
    for code, name in enumerate(pairs.identity_names):
        mask = pairs.identity_code == code
        if not np.any(mask):
            continue
        local = distance[mask]
        identity_rows.append({
            "identity": name,
            "pair_count": int(mask.sum()),
            "mean_distance_m": float(np.mean(local)),
            "within_fraction": float(np.mean(local <= threshold_m)),
        })
    return {
        "pair_count": len(distance),
        "pair_mean_distance_m": float(np.mean(distance)),
        "pair_within_fraction": float(np.mean(distance <= threshold_m)),
        "identity_count": len(identity_rows),
        "identity_macro_mean_distance_m": float(np.mean([row["mean_distance_m"] for row in identity_rows])),
        "identity_macro_within_fraction": float(np.mean([row["within_fraction"] for row in identity_rows])),
        "distance_quantiles_m": {
            key: float(np.quantile(distance, value))
            for key, value in (("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99))
        },
        "identity_rows": identity_rows,
        "pair_distance_m": distance,
    }


__all__ = ["CrossTraversalPairs", "cross_traversal_pairs", "pair_distance_metrics"]
