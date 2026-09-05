"""Caller-scoped, partial-label RegionQueryHead training; no data IO/backbone.

Targets are independently mapped into each branch's token order by the caller,
strictly at the loss boundary. This module does not choose an experiment budget,
produce labels, calibrate a detector, or claim three-class/generalization PASS.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import math
from typing import Sequence

import torch

from .gse_region_queries import RegionQueryHead, RegionTargets, region_set_losses, tokens_from_axes


BRANCHES = ("gt_axes", "predicted_axes", "predicted_no_relations")


@dataclass(frozen=True)
class PartialTrainingConfig:
    seed: int
    steps: int
    batch_size: int
    lr: float
    device: str

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2**63")
        if any(type(v) is not int or v < 1 for v in (self.steps, self.batch_size)):
            raise ValueError("positive integer steps/batch_size required")
        if isinstance(self.lr, bool) or not isinstance(self.lr, (float, int)) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("finite positive learning rate required")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("explicit CPU or CUDA training device required")


@dataclass(frozen=True)
class PartialStructureExample:
    axes: torch.Tensor  # 1,S,3,3, fixed cached/GT values; never optimized
    target: RegionTargets  # 1,M,...; separate mapping for each branch


@dataclass
class PartialTrainingResult:
    heads: dict[str, RegionQueryHead]
    initial_state: dict[str, torch.Tensor]
    schedule: list[list[int]]
    history: dict[str, list[dict]]


def paired_schedule(population: int, config: PartialTrainingConfig) -> list[list[int]]:
    """Seeded without-replacement epochs, including each final short batch."""
    config.validate()
    if type(population) is not int or population < 1:
        raise ValueError("positive population required")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    result = []
    while len(result) < config.steps:
        order = torch.randperm(population, generator=generator).tolist()
        result.extend(order[start:start + config.batch_size] for start in range(0, population, config.batch_size))
    return result[:config.steps]


def _batch(examples: Sequence[PartialStructureExample], device):
    axes = torch.cat([x.axes.detach() for x in examples]).to(device)
    b, s = axes.shape[:2]
    m = max(x.target.centers_m.shape[1] for x in examples)
    floating = dict(device=device, dtype=axes.dtype)
    centers = torch.zeros((b, m, 3), **floating)
    members = torch.zeros((b, m, 2 * s), **floating)
    center_valid = torch.zeros((b, m), device=device, dtype=torch.bool)
    event_valid = center_valid.clone()
    member_valid = torch.zeros_like(members, dtype=torch.bool)
    events = torch.zeros((b, m), device=device, dtype=torch.long)
    for i, example in enumerate(examples):
        t = example.target
        size = t.centers_m.shape[1]
        for destination, source in ((centers, t.centers_m), (members, t.members),
                (center_valid, t.center_valid), (event_valid, t.event_valid),
                (member_valid, t.member_valid), (events, t.events)):
            destination[i, :size] = source[0].detach().to(device)
    return axes, RegionTargets(centers, center_valid, events, event_valid, members,
                               member_valid, torch.zeros(b, device=device, dtype=torch.bool))


def _validate_examples(branches):
    if set(branches) != set(BRANCHES):
        raise ValueError("exact GT/predicted/predicted-no-relations branches required")
    population = len(branches[BRANCHES[0]])
    if population < 1 or any(len(branches[name]) != population for name in BRANCHES):
        raise ValueError("equal nonempty paired populations required")
    for name in BRANCHES:
        first = branches[name][0].axes
        for example in branches[name]:
            axes, target = example.axes, example.target
            if axes.shape != first.shape or axes.shape[0] != 1 or axes.dtype != torch.float32:
                raise ValueError("each branch requires fixed-shape B=1 float32 axes")
            tokens_from_axes(axes)
            if target.label_complete.shape != (1,) or target.label_complete.dtype != torch.bool or target.label_complete.any():
                raise ValueError("partial training requires label_complete=False for every example")
            # Validate exact original masks/dtypes before padding could cast them.
            if target.centers_m.ndim != 3 or target.centers_m.shape[0] != 1 or target.centers_m.shape[-1] != 3:
                raise ValueError("original target centers must be 1,M,3")
            m = target.centers_m.shape[1]
            n = axes.shape[1] * 2
            expected = {"centers_m": ((1, m, 3), axes.dtype), "center_valid": ((1, m), torch.bool),
                        "events": ((1, m), torch.long), "event_valid": ((1, m), torch.bool),
                        "members": ((1, m, n), axes.dtype), "member_valid": ((1, m, n), torch.bool)}
            for field, (shape, dtype) in expected.items():
                value = getattr(target, field)
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError(f"invalid original target {field}")
    # The direct ablation must consume exactly the same fixed predictions and
    # loss targets. Neither shape similarity nor close geometry is sufficient.
    for a, b in zip(branches["predicted_axes"], branches["predicted_no_relations"]):
        if not torch.equal(a.axes, b.axes):
            raise ValueError("direct ablation prediction inputs drift")
        for field in fields(RegionTargets):
            x, y = getattr(a.target, field.name), getattr(b.target, field.name)
            if x.shape != y.shape or x.dtype != y.dtype or not torch.all((x == y) | (torch.isnan(x) & torch.isnan(y))):
                raise ValueError("direct ablation loss targets drift")
    return population


def train_partial_structure(branches: dict[str, Sequence[PartialStructureExample]],
                            config: PartialTrainingConfig, *, hidden: int) -> PartialTrainingResult:
    """Train three paired heads with Adam and caller-explicit budget/width.

No gradients enter supplied cached axes or targets. Each branch independently
performs the same loss-only Hungarian routine, never shares GT query matches.
An entirely unsupervised batch is an error, not a successful zero-loss update.
"""
    config.validate()
    population = _validate_examples(branches)
    schedule = paired_schedule(population, config)
    # CPU-local initialization restores caller RNG and is identical for CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        template = RegionQueryHead(hidden=hidden)
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    heads, history = {}, {}
    for name in BRANCHES:
        head = deepcopy(template).to(config.device)
        head.use_relations = name != "predicted_no_relations"
        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=config.lr)
        records = []
        for step, indices in enumerate(schedule):
            axes, target = _batch([branches[name][i] for i in indices], config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = region_set_losses(head(tokens_from_axes(axes)), target)
            if not loss["has_supervision"]:
                raise ValueError("batch has no usable supervision; no zero-loss success")
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError("nonfinite partial loss")
            loss["total"].backward()
            gradients = [p.grad for p in head.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("missing/nonfinite head gradients")
            gradient_norm = float(torch.sqrt(sum(g.detach().double().square().sum() for g in gradients)).cpu())
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise FloatingPointError("nonfinite updated head parameters")
            records.append({"step": step + 1, "sample_indices": list(indices),
                "loss": {k: float(loss[k].detach().cpu()) for k in
                         ("total", "center", "event", "membership", "presence", "uncertainty")},
                "counts": dict(loss["counts"]), "matches": list(loss["matches"]),
                "gradient_l2": gradient_norm, "gradient_tensor_count": len(gradients),
                "optimizer_steps": step + 1})
        heads[name], history[name] = head, records
    return PartialTrainingResult(heads, initial, schedule, history)
