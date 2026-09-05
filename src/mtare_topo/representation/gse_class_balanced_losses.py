"""Independent 2x2 class-balance wrapper around frozen geometry binding.

Pass global TRAINING counts explicitly: members=(negative, positive), events=
(corridor, junction, terminal). None disables that factor. No batch-frequency
estimation, resampling, threshold selection, model IO or scientific PASS.
"""
import math

import torch
from torch.nn import functional as F

from .gse_geometry_bound_losses import geometry_bound_region_losses


def _weights(counts, classes, reference):
    if (not isinstance(counts, tuple) or len(counts) != classes
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError("global class counts must be an explicit nonnegative integer tuple")
    total, supported = sum(counts), sum(n > 0 for n in counts)
    if supported < 2:
        raise ValueError("class balancing requires at least two supported global classes")
    try:
        weights = [total / (supported * n) if n else 0. for n in counts]
    except OverflowError as error:
        raise ValueError("global class weight overflow") from error
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("global class weights must be finite")
    result = reference.new_tensor(weights)
    if not torch.isfinite(result).all():
        raise ValueError("global class weights overflow prediction dtype")
    return result


def class_balanced_region_losses(prediction, target, *, member_class_counts=None,
                                 event_class_counts=None):
    """Enable member and/or event balancing without changing assignments.

    00: both None -> return the original loss dictionary directly, exactly.
    10/01/11: replace enabled task's numerator with sum(w[class] * loss_i),
    using w=N/(K*n_class) from the supplied FULL training population. K counts
    supported classes, so terminal=0 does not invent terminal supervision.
    Divide by the ORIGINAL task denominator, never by batch weights or by
    matched-label count. Geometry ties/UNKNOWN remain excluded but accounted.

    These weights are not pos_weight-only BCE: BOTH classes receive their
    global balancing weights, keeping the full-population mean weight one.
    Three event logits remain in the softmax, including unsupported terminal.
    """
    original = geometry_bound_region_losses(prediction, target)
    if member_class_counts is None and event_class_counts is None:
        return original
    member_weights = (_weights(member_class_counts, 2, prediction.membership_logits)
                      if member_class_counts is not None else None)
    event_weights = (_weights(event_class_counts, 3, prediction.event_logits)
                     if event_class_counts is not None else None)
    if member_weights is not None:
        known = target.members[target.member_valid]
        if not ((known == 0) | (known == 1)).all():
            raise ValueError("balanced member targets must be binary, not soft labels")
        # Global counts cover the full numerically usable training population,
        # not just the current Hungarian matches. Ties cannot hide a bad count.
        usable = target.member_valid & prediction.query_supported[:, None]
        labels = target.members[usable].long()
        if labels.numel() and (member_weights[labels] == 0).any():
            raise ValueError("known usable member class has zero global support")
    if event_weights is not None:
        labels = target.events[target.event_valid]
        if labels.numel() and (event_weights[labels] == 0).any():
            raise ValueError("known event class has zero global support")
    result = dict(original)
    member_terms, event_terms = [], []
    for row, query, ti in original["matches"]:
        if member_weights is not None:
            valid = target.member_valid[row, ti] & prediction.query_supported[row]
            if valid.any():
                labels = target.members[row, ti, valid]
                raw = F.binary_cross_entropy_with_logits(prediction.membership_logits[row, query, valid],
                                                        labels, reduction="none")
                member_terms.extend((raw * member_weights[labels.long()]).unbind())
        if event_weights is not None and target.event_valid[row, ti]:
            label = target.events[row, ti]
            # Explicit sum after weighting. Default CE weighted-mean reduction
            # on one example would cancel its class weight completely.
            raw = F.cross_entropy(prediction.event_logits[row, query][None], label[None], reduction="none")[0]
            event_terms.append(raw * event_weights[label])
    if member_weights is not None:
        result["membership"] = (torch.stack(member_terms).sum() / original["denominators"]["membership"]
                                if member_terms else original["membership"])
    if event_weights is not None:
        result["event"] = (torch.stack(event_terms).sum() / original["denominators"]["event"]
                           if event_terms else original["event"])
    result["total"] = sum(result[name] for name in ("center", "event", "membership", "presence", "uncertainty"))
    return result
