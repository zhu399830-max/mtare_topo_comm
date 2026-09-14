"""Paired fixed-budget heads with explicit population-checked class weights.

No data IO, backbone, resampling, threshold selection or budget selection.
The 00 path retains the geometry-bound trainer's exact records and arithmetic.
"""
from copy import deepcopy
from typing import Callable, Sequence

import torch

from .gse_class_balanced_losses import _weights, class_balanced_region_losses
from .gse_partial_structure_training import (
    BRANCHES, PartialStructureExample, PartialTrainingConfig, PartialTrainingResult,
    _validate_examples, _batch, paired_schedule,
)
from .gse_region_queries import RegionQueryHead, tokens_from_axes


def _population_balance(branches, member_counts, event_counts):
    """Check every branch before creating a head/optimizer; ignore UNKNOWN.

    Counts are full-population labels, not Hungarian-assigned counts. Numerical
    axis support filters direction labels only, exactly as the loss denominator.
    """
    specification = {}
    for field, counts, size, order in (
        ("membership", member_counts, 2, ("negative", "positive")),
        ("event", event_counts, 3, ("corridor", "junction", "terminal")),
    ):
        if counts is not None:
            weights = _weights(counts, size, branches[BRANCHES[0]][0].axes)
            specification[field] = {"class_order": list(order), "global_class_counts": list(counts),
                                    "weights": weights.detach().cpu().tolist()}
    if not specification:
        return None
    with torch.no_grad():
        for name in BRANCHES:
            members, events = [0, 0], [0, 0, 0]
            for example in branches[name]:
                target = example.target
                if member_counts is not None:
                    valid = tokens_from_axes(example.axes.detach()).valid
                    usable = target.member_valid & valid[:, None].to(target.member_valid.device)
                    labels = target.members[usable]
                    if not ((labels == 0) | (labels == 1)).all():
                        raise ValueError(f"{name}: usable member labels must be binary")
                    for category in range(2):
                        members[category] += int((labels == category).sum().item())
                if event_counts is not None:
                    labels = target.events[target.event_valid]
                    if not ((labels >= 0) & (labels < 3)).all():
                        raise ValueError(f"{name}: known event labels must be in [0, 2]")
                    for category in range(3):
                        events[category] += int((labels == category).sum().item())
            if member_counts is not None and tuple(members) != member_counts:
                raise ValueError(f"{name}: full-population membership counts {tuple(members)} != {member_counts}")
            if event_counts is not None and tuple(events) != event_counts:
                raise ValueError(f"{name}: full-population event counts {tuple(events)} != {event_counts}")
    return specification


def train_class_balanced_structure(branches: dict[str, Sequence[PartialStructureExample]],
                                   config: PartialTrainingConfig, *, hidden: int,
                                   member_class_counts=None, event_class_counts=None,
                                   on_step: Callable[[str, dict], None] | None = None) -> PartialTrainingResult:
    """Unchanged paired Adam loop, with independent member/event loss factors.

    Enabled factors require caller-supplied exact counts for ALL three complete
    training populations, checked before any update. Disabled factors add no
    count restriction. Cached axes/targets remain detached. Callback failure or
    absent supervision stops; neither retries nor best-step selection occur.
    """
    config.validate()
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    population = _validate_examples(branches)
    balance = _population_balance(branches, member_class_counts, event_class_counts)
    schedule = paired_schedule(population, config)
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
            loss = class_balanced_region_losses(head(tokens_from_axes(axes)), target,
                member_class_counts=member_class_counts, event_class_counts=event_class_counts)
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
                "geometry_counts": deepcopy(loss.get("geometry_counts", {})),
                "supervision_counts": deepcopy(loss.get("supervision_counts", {})),
                "denominators": deepcopy(loss.get("denominators", {})),
                "gradient_l2": gradient_norm, "gradient_tensor_count": len(gradients),
                "optimizer_steps": step + 1})
            if balance is not None:
                records[-1]["class_balance"] = deepcopy(balance)
            if on_step is not None:
                on_step(name, deepcopy(records[-1]))
        heads[name], history[name] = head, records
    return PartialTrainingResult(heads, initial, schedule, history)
