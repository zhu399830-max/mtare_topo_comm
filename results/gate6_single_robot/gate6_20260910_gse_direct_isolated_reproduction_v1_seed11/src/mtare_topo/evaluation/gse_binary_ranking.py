"""Tie-aware binary ranking diagnosis, never calibration/threshold search."""
import math

import numpy as np


def _distribution(scores):
    if len(scores) == 0:
        return None
    return {"count": int(len(scores)), "mean": float(np.mean(scores)),
            "median": float(np.median(scores)), "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)), "p10": float(np.percentile(scores, 10)),
            "p90": float(np.percentile(scores, 90))}


def binary_ranking(probabilities, labels):
    """Return block-tie AP, half-credit-tie AUROC, and fixed >=0.5 confusion.

Scores must be finite 1D probabilities in [0,1]; labels finite binary 0/1.
AP accumulates precision only after each entire exact-score tie group. It
never orders positives ahead of negatives inside a tie. AUROC is the fraction
of positive/negative pairs correctly ordered, with half credit for equal scores.

Empty/single-class populations return BOTH AP and AUROC as None and explicitly
mark rank_unidentifiable. In particular we do not report trivial AP=1 for an
all-positive population. No probability clipping, optimal threshold, or BCE is
produced; a caller needing BCE should use saved logits and its frozen loss.
"""
    raw_p, raw_y = np.asarray(probabilities), np.asarray(labels)
    if (raw_p.ndim != 1 or raw_y.ndim != 1 or raw_p.shape != raw_y.shape
            or raw_p.dtype.kind not in "iuf" or raw_y.dtype.kind not in "biuf"
            or raw_p.dtype.itemsize > 8 or raw_y.dtype.itemsize > 8):
        raise ValueError("equal-length 1D numeric probabilities and binary labels required")
    if (not np.isfinite(raw_p).all() or not np.isfinite(raw_y).all()
            or np.any((raw_p < 0) | (raw_p > 1)) or not np.isin(raw_y, [0, 1]).all()):
        raise ValueError("finite probabilities in [0,1] and binary labels required")
    p, y = raw_p.astype(np.float64), raw_y.astype(bool)
    size, positive = int(len(p)), int(y.sum())
    negative = size - positive
    prediction = p >= .5
    confusion = {"tp": int((prediction & y).sum()), "fp": int((prediction & ~y).sum()),
                 "fn": int((~prediction & y).sum()), "tn": int((~prediction & ~y).sum())}
    ap, auc = None, None
    groups = int(len(np.unique(p)))
    unidentifiable = positive == 0 or negative == 0
    if not unidentifiable:
        order = np.argsort(-p, kind="stable")
        ordered_p, ordered_y = p[order], y[order]
        ends = np.r_[np.flatnonzero(ordered_p[1:] != ordered_p[:-1]) + 1, size]
        start, cumulative_positive, cumulative_negative = 0, 0, 0
        ap_terms, auc_terms = [], []
        for end in ends:
            end = int(end)
            group_positive = int(ordered_y[start:end].sum())
            group_negative = end - start - group_positive
            cumulative_positive += group_positive
            cumulative_negative += group_negative
            ap_terms.append((group_positive / positive) * (cumulative_positive / end))
            negative_below = negative - cumulative_negative
            auc_terms.append(group_positive * (negative_below + .5 * group_negative))
            start = end
        ap = float(math.fsum(ap_terms))
        auc = float(math.fsum(auc_terms) / (positive * negative))
    return {"sample_count": size, "positive_count": positive, "negative_count": negative,
            "prevalence": positive / size if size else None,
            "average_precision": ap, "auroc": auc, "rank_unidentifiable": unidentifiable,
            "score_group_count": groups, "positive_scores": _distribution(p[y]),
            "negative_scores": _distribution(p[~y]), "fixed_threshold": .5, "confusion": confusion,
            "diagnostic_not_calibration": True, "threshold_search": False}
