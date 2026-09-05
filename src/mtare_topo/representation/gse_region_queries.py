"""Local region proposals from all axis tokens, never from teacher centers.

Software candidate, not a calibrated detector. Gravity-aligned common yaw
equivariance, not arbitrary SO(3) invariance. Cropped ends anchor proposals;
they are NOT asserted to be physical openings. No place identities, old
existence scores, dimensions, or teacher candidate masks are accepted here.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AxisTokens:
    positions_m: torch.Tensor  # B,N,3, common current sensor coordinates
    tangents: torch.Tensor     # B,N,3, outward; unit for valid tokens
    valid: torch.Tensor        # B,N, numerical direction support, NOT GT

    def validate(self):
        p, t, v = self.positions_m, self.tangents, self.valid
        if (p.ndim != 3 or p.shape[-1] != 3 or p.shape[0] < 1
                or not 1 <= p.shape[1] <= 64 or t.shape != p.shape
                or v.shape != p.shape[:2] or v.dtype != torch.bool):
            raise ValueError("axis token shapes/population must be B,N<=64,3")
        if (p.dtype not in (torch.float32, torch.float64) or t.dtype != p.dtype
                or t.device != p.device or v.device != p.device):
            raise ValueError("common floating dtype/device required")
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError("nonfinite axis tokens")
        if ((torch.linalg.vector_norm(t, dim=-1)[v] - 1).abs() > 1e-4).any():
            raise ValueError("supported tangents must be unit vectors")


def tokens_from_axes(axes):
    """All S<=32 finite predicted axes in, 2S tokens out; no target filtering.

    Exact zero-length end segments are unsupported, not invented directions.
    Relative floating-point resolution is used solely for numerical support.
    Unsupported tokens remain in the output population with an explicit mask.
    """
    if (axes.ndim != 4 or axes.shape[0] < 1 or not 1 <= axes.shape[1] <= 32
            or axes.shape[2:] != (3, 3) or axes.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(axes).all()):
        raise ValueError("finite floating B,S<=32,3,3 axes required")
    ends = axes[:, :, (0, 2)]
    direction = ends - axes[:, :, 1, None]
    norms = torch.linalg.vector_norm(direction, dim=-1)
    # Euclidean radius, unlike max component, is common-yaw invariant. This
    # remains a floating resolution guard, not a physical visibility threshold.
    eps = 32 * torch.finfo(axes.dtype).eps * torch.linalg.vector_norm(axes, dim=-1).amax(-1).clamp_min(1.)
    valid = norms > eps[..., None]
    tangents = direction / norms.clamp_min(torch.finfo(axes.dtype).tiny)[..., None]
    value = AxisTokens(ends.flatten(1, 2), tangents.masked_fill(~valid[..., None], 0).flatten(1, 2),
                       valid.flatten(1))
    value.validate()
    return value


def relation_features(query_positions, query_tangents, value):
    """Invariant scalars retain relative angle, side, distance and height."""
    delta = value.positions_m[:, None] - query_positions[:, :, None]
    ti, tj = query_tangents[:, :, None], value.tangents[:, None]
    shape = delta.shape[:-1]
    return torch.stack((
        torch.linalg.vector_norm(delta, dim=-1) / 50, delta[..., 2] / 50,
        (ti * tj).sum(-1), ti[..., 0] * tj[..., 1] - ti[..., 1] * tj[..., 0],
        (delta * ti).sum(-1) / 50, (delta * tj).sum(-1) / 50,
        ti[..., 2].expand(shape), tj[..., 2].expand(shape),
    ), -1), delta


@dataclass(frozen=True)
class RegionPrediction:
    centers_m: torch.Tensor           # B,N,3 predicted, not GT queries
    event_logits: torch.Tensor        # B,N,3 corridor/junction/terminal
    presence_logits: torch.Tensor     # B,N, uncalibrated
    membership_logits: torch.Tensor   # B,N,N region query x direction token
    uncertainty: torch.Tensor         # B,N, uncalibrated
    query_supported: torch.Tensor     # B,N
    member_supported: torch.Tensor    # B,N,N


class RegionQueryHead(nn.Module):
    """Shared small head, one proposal per directional anchor, no slot IDs.

    `use_relations=False` zeros both relational paths (and displacement vote),
    retaining identical parameters and unary features. It is a direct feature
    ablation, not a historical decoder. All proposals remain for loss/scoring;
    this module performs no NMS, threshold selection, node merge or graph edge.
    """
    def __init__(self, hidden=64, *, use_relations=True):
        super().__init__()
        if type(hidden) is not int or hidden < 1 or type(use_relations) is not bool:
            raise ValueError("invalid head configuration")
        self.use_relations = use_relations
        self.pair = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.encode = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.center_coefficients = nn.Linear(hidden, 3)
        self.displacement_weight = nn.Linear(hidden, 1)
        self.condition = nn.Sequential(nn.Linear(2 * hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.event = nn.Linear(hidden, 3)
        self.presence = nn.Linear(hidden, 1)
        self.membership = nn.Linear(hidden, 1)
        self.uncertainty = nn.Linear(hidden, 1)

    def forward(self, value: AxisTokens):
        value.validate()
        p, t, v = value.positions_m, value.tangents, value.valid
        # Zero padding before arithmetic so unsupported geometry cannot affect
        # output through intermediate expressions or pooled normalizers.
        p, t = p.masked_fill(~v[..., None], 0), t.masked_fill(~v[..., None], 0)
        value = AxisTokens(p, t, v)
        unary = torch.stack((torch.linalg.vector_norm(p[..., :2], dim=-1) / 50,
                             p[..., 2] / 50, t[..., 2]), -1)
        pair, delta = relation_features(p, t, value)
        mask = v[:, :, None] & v[:, None, :]
        if not self.use_relations:
            pair, delta = torch.zeros_like(pair), torch.zeros_like(delta)
        messages = self.pair(pair).masked_fill(~mask[..., None], 0)
        denominator = mask.sum(2).clamp_min(1)[..., None]
        context = messages.sum(2) / denominator
        encoded = self.encode(torch.cat((unary, context), -1)).masked_fill(~v[..., None], 0)
        coeff = self.center_coefficients(encoded)
        z = torch.zeros_like(t); z[..., 2] = 1
        centers = p + coeff[..., :1] * t + coeff[..., 1:2] * torch.cross(z, t, dim=-1) + coeff[..., 2:] * z
        votes = torch.tanh(self.displacement_weight(messages)) * delta
        centers = (centers + votes.masked_fill(~mask[..., None], 0).sum(2) / denominator).masked_fill(~v[..., None], 0)
        region_pair, _ = relation_features(centers, t, value)
        if not self.use_relations:
            region_pair = torch.zeros_like(region_pair)
        n = p.shape[1]
        query = encoded[:, :, None].expand(-1, -1, n, -1)
        token = encoded[:, None].expand(-1, n, -1, -1)
        conditioned = self.condition(torch.cat((query, token, region_pair), -1)).masked_fill(~mask[..., None], 0)
        pooled = conditioned.sum(2) / denominator
        return RegionPrediction(
            centers, self.event(pooled).masked_fill(~v[..., None], 0),
            self.presence(pooled).squeeze(-1).masked_fill(~v, -30),
            self.membership(conditioned).squeeze(-1).masked_fill(~mask, -30),
            torch.where(v, torch.sigmoid(self.uncertainty(pooled).squeeze(-1)), 1.), v, mask,
        )


@dataclass(frozen=True)
class RegionTargets:
    """LOSS ONLY: members must already map to prediction input token order.

    A teacher-to-predicted primitive correspondence belongs exclusively to
    this loss boundary, never to forward input selection. Incomplete labels
    do not create background negatives for unmatched proposals.
    """
    centers_m: torch.Tensor       # B,M,3
    center_valid: torch.Tensor    # B,M
    events: torch.Tensor         # B,M long
    event_valid: torch.Tensor    # B,M
    members: torch.Tensor        # B,M,N
    member_valid: torch.Tensor   # B,M,N
    label_complete: torch.Tensor # B; only permits unmatched presence negatives


def region_set_losses(prediction, target):
    """Hungarian alignment at loss time; no teacher query coordinates.

    Equal task weights and the existing fixed 50 m coordinate scale are a
    SOFTWARE candidate, not a frozen training spec. Return supervision counts
    so zero valid labels can never be misreported as successful fitting.
    Uncertainty uses event-error Brier loss; it requires separate calibration.
    """
    from scipy.optimize import linear_sum_assignment
    b, n = prediction.query_supported.shape
    if (prediction.query_supported.dtype != torch.bool
            or prediction.member_supported.shape != (b, n, n)
            or prediction.member_supported.dtype != torch.bool
            or not torch.equal(prediction.member_supported,
                prediction.query_supported[:, :, None] & prediction.query_supported[:, None, :])):
        raise ValueError("prediction support must follow the all-axis-token contract")
    for value in (prediction.centers_m, prediction.event_logits, prediction.presence_logits,
                  prediction.membership_logits, prediction.uncertainty):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite prediction")
    if target.centers_m.ndim != 3 or target.centers_m.shape[0] != b or target.centers_m.shape[2] != 3:
        raise ValueError("target centers must be B,M,3")
    m = target.centers_m.shape[1]
    shape = (b, m)
    device, dtype = prediction.centers_m.device, prediction.centers_m.dtype
    for field, expected in (("center_valid", shape), ("event_valid", shape),
                            ("member_valid", (b, m, n)), ("label_complete", (b,))):
        x = getattr(target, field)
        if x.shape != expected or x.dtype != torch.bool or x.device != device:
            raise ValueError("invalid target masks")
    if target.events.shape != shape or target.events.dtype != torch.long or target.events.device != device:
        raise ValueError("invalid event targets")
    for x, expected in ((target.centers_m, (b, m, 3)), (target.members, (b, m, n))):
        if x.shape != expected or x.dtype != dtype or x.device != device:
            raise ValueError("invalid floating targets")
    if (not torch.isfinite(target.centers_m[target.center_valid]).all()
            or ((target.events[target.event_valid] < 0) | (target.events[target.event_valid] > 2)).any()
            or not torch.isfinite(target.members[target.member_valid]).all()
            or ((target.members[target.member_valid] < 0) | (target.members[target.member_valid] > 1)).any()):
        raise ValueError("invalid known target")
    losses = {k: [] for k in ("center", "event", "membership", "presence", "uncertainty")}
    matches = []
    counts = dict.fromkeys(("targets", "matched", "unmatched_targets", "unsupported_member_targets", "unconfirmed_presence_targets", "center", "event", "membership", "presence_positive", "presence_negative"), 0)
    usable_members = target.member_valid & prediction.query_supported[:, None]
    # A negative-only/soft membership target without a known center/event
    # cannot certify that a region exists. Never invent a presence positive.
    eligible_all = target.center_valid | target.event_valid | (usable_members & (target.members == 1)).any(-1)
    presence_denominator = sum(int(prediction.query_supported[r].sum()) if target.label_complete[r]
                              else min(int(prediction.query_supported[r].sum()), int(eligible_all[r].sum())) for r in range(b))
    # Fixed batch denominators do not depend on which targets get matched.
    # This keeps assignment cost and training objective consistent even if
    # targets outnumber supported queries; unmatched targets remain explicit.
    denominators = {"center": max(1, int(target.center_valid.sum())),
                    "event": max(1, int(target.event_valid.sum())),
                    "uncertainty": max(1, int(target.event_valid.sum())),
                    "membership": max(1, int(usable_members.sum())),
                    "presence": max(1, presence_denominator)}
    for row in range(b):
        queries = torch.where(prediction.query_supported[row])[0]
        known = target.center_valid[row] | target.event_valid[row] | target.member_valid[row].any(-1)
        # Known membership on an unusable predicted direction has no trainable
        # correspondence. Retain it as an unmatched target in accounting; do
        # not turn a zero masked membership cost into positive presence.
        member_usable = usable_members[row].any(-1)
        eligible = eligible_all[row]
        targets = torch.where(eligible)[0]
        counts["targets"] += int(known.sum())
        counts["unsupported_member_targets"] += int((known & ~eligible & ~member_usable).sum())
        counts["unconfirmed_presence_targets"] += int((known & ~eligible & member_usable).sum())
        cost = prediction.centers_m.new_zeros((len(queries), len(targets)))
        # Match the task-normalized training objective, including the positive
        # presence cost (and removed background cost for complete labels).
        # Otherwise geometrically tied queries can have different total loss
        # purely because Hungarian picked their array index.
        center_count, event_count = denominators["center"], denominators["event"]
        member_count, presence_count = denominators["membership"], denominators["presence"]
        presence_cost = F.softplus(-prediction.presence_logits[row, queries])
        if target.label_complete[row]:
            presence_cost = presence_cost - F.softplus(prediction.presence_logits[row, queries])
        cost += presence_cost[:, None] / presence_count
        for col, ti in enumerate(targets):
            if target.center_valid[row, ti]:
                cost[:, col] += torch.linalg.vector_norm(prediction.centers_m[row, queries] - target.centers_m[row, ti], dim=-1) / (50 * center_count)
            if target.event_valid[row, ti]:
                cost[:, col] += -prediction.event_logits[row, queries].log_softmax(-1)[:, target.events[row, ti]] / event_count
                error = (prediction.event_logits[row, queries].detach().argmax(-1) != target.events[row, ti]).to(dtype)
                cost[:, col] += (prediction.uncertainty[row, queries] - error).square() / event_count
            pv = target.member_valid[row, ti]
            if pv.any():
                raw = F.binary_cross_entropy_with_logits(prediction.membership_logits[row, queries][:, pv],
                    target.members[row, ti, pv][None].expand(len(queries), -1), reduction="none")
                support = prediction.member_supported[row, queries][:, pv]
                cost[:, col] += (raw * support).sum(-1) / member_count
        if not torch.isfinite(cost).all():
            raise ValueError("nonfinite matching cost")
        qi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
        chosen = set()
        for qidx, tidx in zip(qi, ti):
            q, t = int(queries[qidx]), int(targets[tidx]); chosen.add(q)
            matches.append((row, q, t)); counts["matched"] += 1
            if target.center_valid[row, t]:
                losses["center"].append(torch.linalg.vector_norm(prediction.centers_m[row, q] - target.centers_m[row, t]) / 50)
                counts["center"] += 1
            if target.event_valid[row, t]:
                losses["event"].append(F.cross_entropy(prediction.event_logits[row, q][None], target.events[row, t][None]))
                error = (prediction.event_logits[row, q].detach().argmax() != target.events[row, t]).to(dtype)
                losses["uncertainty"].append((prediction.uncertainty[row, q] - error).square())
                counts["event"] += 1
            pv = target.member_valid[row, t] & prediction.member_supported[row, q]
            if pv.any():
                losses["membership"].extend(F.binary_cross_entropy_with_logits(prediction.membership_logits[row, q, pv], target.members[row, t, pv], reduction="none").unbind())
                counts["membership"] += int(pv.sum())
            losses["presence"].append(F.softplus(-prediction.presence_logits[row, q]))
            counts["presence_positive"] += 1
        counts["unmatched_targets"] += int(known.sum()) - len(chosen)
        if target.label_complete[row]:
            for q in queries.tolist():
                if q not in chosen:
                    losses["presence"].append(F.softplus(prediction.presence_logits[row, q]))
                    counts["presence_negative"] += 1
    zero = sum(x.sum() * 0 for x in (prediction.centers_m, prediction.event_logits,
               prediction.membership_logits, prediction.presence_logits, prediction.uncertainty))
    result = {key: torch.stack(values).sum() / denominators[key] if values else zero for key, values in losses.items()}
    return {**result, "total": sum(result.values()), "counts": counts, "matches": matches,
            "has_supervision": bool(counts["matched"] or counts["presence_negative"])}
