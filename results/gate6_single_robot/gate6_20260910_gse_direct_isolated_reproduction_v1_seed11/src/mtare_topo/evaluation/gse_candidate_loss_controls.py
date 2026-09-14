"""Analytic/software objective controls, not an approved training objective.

Peak-bag and hardest-background alternatives expose failure modes of averaged
soft support. They do not penalize all duplicate positives and are therefore
insufficient alone for a complete detector-training contract.
"""
import torch
from torch.nn import functional as F


def peak_and_hard_background_control(logits,positive_bags,known_background):
    if (logits.ndim!=1 or logits.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(logits).all() or positive_bags.ndim!=2
            or positive_bags.shape[1]!=len(logits) or positive_bags.dtype!=torch.bool
            or known_background.shape!=logits.shape or known_background.dtype!=torch.bool
            or positive_bags.device!=logits.device or known_background.device!=logits.device):
        raise ValueError('finite1D logits, K,N boolean bags, N background required')
    if (positive_bags & known_background[None]).any():raise ValueError('positive/background conflict')
    if len(positive_bags) and not positive_bags.any(-1).all():raise ValueError('uncovered target bag')
    peaks=[F.softplus(-logits[bag].max()) for bag in positive_bags]
    positive=torch.stack(peaks).mean() if peaks else logits.sum()*0
    negative=F.softplus(logits[known_background].max()) if known_background.any() else logits.sum()*0
    parts=[]
    if peaks:parts.append(positive)
    if known_background.any():parts.append(negative)
    loss=torch.stack(parts).mean() if parts else logits.sum()*0
    return loss,dict(positive=positive,hard_background=negative,
        duplicate_positives_controlled=False,training_qualified=False)
