"""Paired geometry-only fit probe, not event/detection training.

Student tuples and targets are separate. No GT identities or candidate
preselection enter the head. Matching is loss/scoring-only; surplus queries
are retained, so this probe does not measure detector precision.
"""
from copy import deepcopy
import hashlib
import math
import numpy as np
import torch

from .gse_point_axis_loss import axis_set_loss, axis_set_metrics


VARIANTS = ("raw_no_offset", "raw_slot_offset")


def sample_schedule(count=180, epochs=3, seed=0):
    if any(type(v) is not int for v in (count, epochs, seed)) or min(count, epochs) < 1:
        raise ValueError("invalid schedule")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(count) for _ in range(epochs)]).tolist()


def tensor_state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def configure_variant(initial, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    model = deepcopy(initial)
    for layer in (model.offset, model.slot_offset):
        if any(bool(p.detach().any()) for p in layer.parameters()):
            raise ValueError("paired probe requires zero initial offsets")
        layer.requires_grad_(variant != "raw_no_offset")
    return model


def on_device(entry, device):
    return [v.to(device) for v in entry["student"]], entry["target"].to(device), entry["mask"].to(device)


@torch.no_grad()
def evaluate(model, cache):
    model.eval()
    device = next(model.parameters()).device
    axes, metrics = [], []
    for entry in cache:
        inputs, target, mask = on_device(entry, device)
        output = model(*inputs).votes.axis_control_m
        metrics.extend(axis_set_metrics(output, target, mask))
        axes.append(output.detach().cpu().numpy()[0])
    return np.stack(axes), metrics


def fit(model, cache, schedule, log_step, *, learning_rate=.001, guard=lambda: None):
    if (not isinstance(schedule,(list,tuple)) or not schedule
            or any(type(index) is not int or not 0<=index<len(cache) for index in schedule)):
        raise ValueError("nonempty schedule with valid indices required before any update")
    if not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("learning rate must be finite and positive")
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=0.)
    initial_sha = tensor_state_sha(model)
    for step, index in enumerate(schedule):
        if type(index) is not int or not 0 <= index < len(cache):
            raise ValueError("schedule index out of bounds")
        guard()
        inputs, target, mask = on_device(cache[index], device)
        optimizer.zero_grad(set_to_none=True)
        output = model(*inputs).votes.axis_control_m
        loss = axis_set_loss(output, target, mask)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad):
            raise ValueError("missing/nonfinite trainable gradient")
        guard()  # Check actual backward allocation BEFORE the optimizer writes.
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("nonfinite updated parameter")
        log_step({"step":step+1,"observation_index":index,"loss_coordinate_l1_div50":float(loss.detach())})
    guard()
    return {"steps":len(schedule),"initial_state_sha256":initial_sha,
            "final_state_sha256":tensor_state_sha(model), "optimizer_state_dict":optimizer.state_dict()}


def decide(initial, raw, full, legacy, parents):
    """Predeclared FIT utility decision, never a generalization/Gate decision.

    At least10% macro coordinate-MAE improvement vs raw/no-offset AND frozen
    legacy, both error measures improve vs initial, and >=6/10 parent MAEs
    improve vs raw. These thresholds concern this NEW probe only.
    """
    if not (len(initial) == len(raw) == len(full) == len(legacy) == len(parents) == 180):
        raise ValueError("decision requires the exact180 observations")
    if len(set(parents)) != 10 or any(parents.count(p) != 18 for p in set(parents)):
        raise ValueError("decision requires10 parents/18 observations each")
    for rows in (initial,raw,full,legacy):
        for row in rows:
            for key in ("coordinate_mae_m","point_mean_euclidean_m"):
                value=row.get(key)
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                    raise ValueError("decision metrics must be finite nonnegative numbers")
    def mean(rows, key):
        return float(np.mean([row[key] for row in rows]))
    all_rows = {"initial":initial,"raw_no_offset":raw,"raw_slot_offset":full,"legacy_frozen":legacy}
    summary = {name:{key:mean(rows,key) for key in ("coordinate_mae_m","point_mean_euclidean_m")}
               for name,rows in all_rows.items()}
    by_parent=[]
    for parent in sorted(set(parents)):
        indices=[i for i,p in enumerate(parents) if p==parent]
        by_parent.append({"parent":parent, **{name:float(np.mean([rows[i]["coordinate_mae_m"] for i in indices]))
                                              for name,rows in all_rows.items()}})
    f=summary["raw_slot_offset"]
    checks = {
        "coordinate_mae_at_least10pct_better_than_raw":f["coordinate_mae_m"] <= .9*summary["raw_no_offset"]["coordinate_mae_m"],
        "coordinate_mae_at_least10pct_better_than_legacy":f["coordinate_mae_m"] <= .9*summary["legacy_frozen"]["coordinate_mae_m"],
        "coordinate_mae_better_than_initial":f["coordinate_mae_m"] < summary["initial"]["coordinate_mae_m"],
        "euclidean_better_than_initial":f["point_mean_euclidean_m"] < summary["initial"]["point_mean_euclidean_m"],
        "euclidean_better_than_raw":f["point_mean_euclidean_m"] < summary["raw_no_offset"]["point_mean_euclidean_m"],
        "at_least6_parent_mae_improvements":sum(p["raw_slot_offset"]<p["raw_no_offset"] for p in by_parent)>=6,
    }
    return {"decision":"FIT_PROBE_POSITIVE" if all(checks.values()) else "STOP_BEFORE_EXPANSION",
            "checks":checks,"macro_observation_errors":summary,"parents":by_parent,
            "scientific_gate_pass":False,"generalization_claim":False,"detection_or_membership_accuracy_claim":False}
