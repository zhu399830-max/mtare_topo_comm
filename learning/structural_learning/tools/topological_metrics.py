from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:  # Keep the metric unit tests usable in ROS-only hosts.
    def average_precision_score(y_true: np.ndarray, scores: np.ndarray) -> float:
        order = np.argsort(-scores, kind="mergesort")
        truth = np.asarray(y_true, dtype=bool)[order]
        positives = max(int(truth.sum()), 1)
        precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
        return float(np.sum(precision * truth) / positives)

    def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
        truth = np.asarray(y_true, dtype=bool)
        pos = np.flatnonzero(truth)
        neg = np.flatnonzero(~truth)
        if not len(pos) or not len(neg):
            return float("nan")
        ranks = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")
        return float((ranks[pos].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def direction_metric_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Positive-class metrics for a [sample, direction] binary field.

    The mask is broadcast only when its shape is exactly compatible with the
    labels. Labels are thresholded at 0.5; soft probabilities are never used
    as hard truth. Sample-wise F1 is positive-class F1 per sample and is zero
    when a sample has no true or predicted positive direction.
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"labels/probabilities shape mismatch: {y.shape} vs {p.shape}")
    if valid_mask is None:
        mask = np.ones_like(y, dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(valid_mask, dtype=bool), y.shape)
    finite_mask = np.isfinite(y) & np.isfinite(p) & mask
    yt = (y[finite_mask] >= 0.5).astype(np.int64)
    pred = (p[finite_mask] >= float(threshold)).astype(np.int64)
    tp = int(np.sum((yt == 1) & (pred == 1)))
    fp = int(np.sum((yt == 0) & (pred == 1)))
    tn = int(np.sum((yt == 0) & (pred == 0)))
    fn = int(np.sum((yt == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pos_f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
    neg_precision = tn / max(tn + fn, 1)
    neg_recall = tn / max(tn + fp, 1)
    neg_f1 = 2.0 * tn / max(2 * tn + fn + fp, 1)
    rows = y.shape[0]
    row_f1 = []
    offset = 0
    for row in range(rows):
        count = int(np.sum(finite_mask[row]))
        if count == 0:
            continue
        row_y = yt[offset : offset + count]
        row_p = pred[offset : offset + count]
        offset += count
        row_tp = int(np.sum((row_y == 1) & (row_p == 1)))
        row_fp = int(np.sum((row_y == 0) & (row_p == 1)))
        row_fn = int(np.sum((row_y == 1) & (row_p == 0)))
        row_f1.append(2.0 * row_tp / max(2 * row_tp + row_fp + row_fn, 1))
    result = {
        "threshold": float(threshold),
        "label_zero_count": int(np.sum(yt == 0)),
        "label_one_count": int(np.sum(yt == 1)),
        "valid_count": int(len(yt)),
        "positive_rate": float(np.mean(yt == 1)) if len(yt) else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "positive_class_f1": pos_f1,
        "micro_f1": pos_f1,
        "negative_class_f1": neg_f1,
        "macro_f1": 0.5 * (pos_f1 + neg_f1),
        "sample_wise_positive_f1": float(np.mean(row_f1)) if row_f1 else float("nan"),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "prediction_positive_rate": float(np.mean(pred == 1)) if len(pred) else float("nan"),
        "pr_auc": float(average_precision_score(yt, p[finite_mask])) if len(np.unique(yt)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(yt, p[finite_mask])) if len(np.unique(yt)) == 2 else float("nan"),
    }
    return result


def direction_baselines(labels: np.ndarray, train_labels: np.ndarray) -> dict[str, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    prior = train_labels.mean(axis=0)
    majority = (prior >= 0.5).astype(np.float32)
    predictions = {
        "all_zero": np.zeros_like(labels),
        "all_one": np.ones_like(labels),
        "train_global_probability": np.full_like(labels, float(train_labels.mean())),
        "train_directional_probability": np.broadcast_to(prior, labels.shape),
        "train_directional_majority": np.broadcast_to(majority, labels.shape),
    }
    return {name: direction_metric_report(labels, pred) for name, pred in predictions.items()}
