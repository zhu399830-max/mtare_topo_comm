"""Small event-only readout of predicted membership-conditioned geometry.

Software interface only: no teacher, detector threshold, graph update or degree
claim. A numerical-mask match cannot prove that callers used the same scan;
the production input-binding layer must establish that identity separately.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .gse_region_queries import AxisTokens, RegionPrediction, relation_features


@dataclass(frozen=True)
class MembershipEventPrediction:
    event_logits: torch.Tensor  # B,N,3, all proposals retained
    query_supported: torch.Tensor  # B,N, numerical support, NOT detections
    features: torch.Tensor  # B,N,18; soft_count is NOT physical degree


def membership_event_features(prediction: RegionPrediction, tokens: AxisTokens, *,
                              use_membership=True, detach_sources=True):
    """[plain mean(8), membership-weighted mean(8), soft sum(1), count(1)]."""
    if type(use_membership) is not bool or type(detach_sources) is not bool:
        raise ValueError("explicit boolean feature switches required")
    tokens.validate()
    b, n = tokens.valid.shape
    dtype, device = tokens.positions_m.dtype, tokens.positions_m.device
    shapes = {"centers_m": (b, n, 3), "event_logits": (b, n, 3),
              "presence_logits": (b, n), "membership_logits": (b, n, n), "uncertainty": (b, n)}
    for key, shape in shapes.items():
        x = getattr(prediction, key)
        if x.shape != shape or x.dtype != dtype or x.device != device or not torch.isfinite(x).all():
            raise ValueError(f"invalid prediction {key} shape/dtype/device/finite values")
    if ((prediction.uncertainty < 0) | (prediction.uncertainty > 1)).any():
        raise ValueError("prediction uncertainty must be in [0, 1]")
    valid = tokens.valid
    mask = valid[:, :, None] & valid[:, None, :]
    for actual, expected in ((prediction.query_supported, valid), (prediction.member_supported, mask)):
        if (actual.dtype != torch.bool or actual.device != device or actual.shape != expected.shape
                or not torch.equal(actual, expected)):
            raise ValueError("prediction support differs from same-input numerical tokens")
    def source(x): return x.detach() if detach_sources else x
    p = source(tokens.positions_m).masked_fill(~valid[..., None], 0)
    t = source(tokens.tangents).masked_fill(~valid[..., None], 0)
    centers = source(prediction.centers_m).masked_fill(~valid[..., None], 0)
    relations, _ = relation_features(centers, t, AxisTokens(p, t, valid))
    relations = relations.masked_fill(~mask[..., None], 0)
    if not torch.isfinite(relations).all():
        raise ValueError("nonfinite derived relation features")
    count = mask.sum(-1).to(dtype)
    plain = relations.sum(2) / count.clamp_min(1)[..., None]
    if use_membership:
        probability = torch.sigmoid(source(prediction.membership_logits)).masked_fill(~mask, 0)
        soft_count = probability.sum(-1)
        weighted = (probability[..., None] * relations).sum(2) / soft_count.clamp_min(torch.finfo(dtype).tiny)[..., None]
    else:
        weighted, soft_count = torch.zeros_like(plain), torch.zeros_like(count)
    features = torch.cat((plain, weighted, soft_count[..., None], count[..., None]), -1)
    features = features.masked_fill(~valid[..., None], 0)
    if not torch.isfinite(features).all():
        raise ValueError("nonfinite derived event features")
    return features


class MembershipConditionedEventReadout(nn.Module):
    """Fixed 18→32→3 MLP (707 parameters), default frozen source tensors.

    The direct no-membership ablation zeros only feature columns 8:17, keeping
    unweighted geometry, numerical count and the identical parameter layout.
    No threshold or label mask is applied to membership probabilities.
    """
    def __init__(self, *, use_membership=True, detach_sources=True):
        super().__init__()
        if type(use_membership) is not bool or type(detach_sources) is not bool:
            raise ValueError("explicit boolean readout switches required")
        self.use_membership, self.detach_sources = use_membership, detach_sources
        self.mlp = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, prediction: RegionPrediction, tokens: AxisTokens):
        features = membership_event_features(prediction, tokens,
            use_membership=self.use_membership, detach_sources=self.detach_sources)
        if self.mlp[0].weight.dtype != features.dtype or self.mlp[0].weight.device != features.device:
            raise ValueError("readout and source features require identical dtype/device")
        logits = self.mlp(features).masked_fill(~tokens.valid[..., None], 0)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite event logits")
        return MembershipEventPrediction(logits, tokens.valid.clone(), features)
