"""Paired partial-head training with geometry-only attribute assignment.

Separate implementation preserves the historical joint-loss training source.
The sole training intervention is geometry_bound_region_losses; initialization,
Adam, batches, branch modes, logging and failure checks match the old loop.
No dataset, checkpoint, training budget or scientific threshold is chosen here.
"""
from copy import deepcopy
from typing import Callable, Sequence

import torch

from .gse_geometry_bound_losses import geometry_bound_region_losses
from .gse_partial_structure_training import (
    BRANCHES, PartialStructureExample, PartialTrainingConfig, PartialTrainingResult,
    _validate_examples, _batch, paired_schedule,
)
from .gse_region_queries import RegionQueryHead, tokens_from_axes


def train_geometry_bound_structure(branches: dict[str, Sequence[PartialStructureExample]],
                                   config: PartialTrainingConfig, *, hidden: int,
                                   on_step: Callable[[str, dict], None] | None = None) -> PartialTrainingResult:
    """Train unchanged paired heads/budget with the new geometry-bound loss.

    Cached axes and targets remain detached. Callback receives a copy of the
    complete new loss counts, after a successful finite optimizer update.
    Missing supervision or callback failure stops; no silent retry occurs.
    """
    config.validate()
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    population = _validate_examples(branches)
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
            loss = geometry_bound_region_losses(head(tokens_from_axes(axes)), target)
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
            if on_step is not None:
                on_step(name, deepcopy(records[-1]))
        heads[name], history[name] = head, records
    return PartialTrainingResult(heads, initial, schedule, history)
