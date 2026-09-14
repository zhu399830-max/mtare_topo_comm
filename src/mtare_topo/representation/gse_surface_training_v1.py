"""Pure-memory paired surface-head training core; no files or encoder forward.

This code is not a runner or an authorization. Callers must authenticate cached
payloads, their headers and loss targets under a new frozen data card. Header
fingerprints guard mutation here; they do not prove the declared source true.
No scientific evaluation is implemented: initial/final snapshots contain raw
predictions and diagnostic losses for a separate independent evaluator.
"""
from dataclasses import dataclass, fields, replace
import hashlib
import json
import re

import numpy as np
import torch
from torch import nn

from .gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from .gse_surface_relation_model_v1 import SurfacePatchBatch, SurfaceRelationModelV1
from .gse_surface_losses_v1 import SurfaceLossTargets, surface_relation_losses, UNSUPERVISED_HEADS

FROZEN_READOUT_MODULES = ("anchor_uncertainty", "dimension_evidence", "opening_support",
                          "validity_anchor", "validity_opening")


@dataclass(frozen=True)
class SurfaceTrainingExample:
    header: dict
    compact: CompactFrozenDualPathFeatures
    patches: SurfacePatchBatch
    targets: SurfaceLossTargets


@dataclass(frozen=True)
class SurfaceTrainingBudget:
    updates: int                        # attempted groups, not fabricated actual optimizer steps
    seed: int = 0
    learning_rate: float = .001
    weight_decay: float = .0001
    accumulation_steps: int = 4
    device: str = "cpu"


@dataclass(frozen=True)
class PairedSurfaceTrainingResult:
    shared_initial_state: dict
    initial_state_sha256: dict
    final_states: dict
    evaluations: dict
    history: dict
    counts: dict
    schedule: tuple
    schedule_sha256: str
    source_header_sha256: tuple
    frozen_encoder_state_sha256: str
    unsupervised_heads: tuple = UNSUPERVISED_HEADS
    reliability_status: str = "UNTRAINED_UNCALIBRATED"


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def state_dict_sha256(state):
    """Tensor-content fingerprint, NOT a serialized checkpoint-file SHA."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise ValueError("tensor-only state dictionary required")
        digest.update(_json_sha([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module):
    if not isinstance(module, nn.Module):
        raise ValueError("supplied frozen encoder module required")
    return state_dict_sha256(module.state_dict())


def _snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _move(value, device):
    return replace(value, **{f.name: getattr(value, f.name).detach().to(device=device)
                            for f in fields(value)})


def _validate_header(header, encoder_sha):
    names = {"schema_version", "observation_id", "frame_indices", "coordinate_frame",
             "input_binding_sha256", "target_binding_sha256", "frozen_encoder_state_sha256"}
    if type(header) is not dict or set(header) != names or header["schema_version"] != "gse_surface_cached_input_header_v1":
        raise ValueError("explicit versioned actual cache header required")
    if type(header["observation_id"]) is not str or not header["observation_id"]:
        raise ValueError("observation ID required for metadata only")
    indices = header["frame_indices"]
    if (type(indices) not in (tuple, list) or len(indices) != 5 or any(type(i) is not int or i < 0 for i in indices)
            or any(b <= a for a,b in zip(indices, indices[1:]))):
        raise ValueError("explicit five ordered causal source frame identities required")
    if header["coordinate_frame"] != "current_sensor":
        raise ValueError("cache must use common current sensor coordinates")
    if any(type(header[k]) is not str or not re.fullmatch(r"[a-f0-9]{64}", header[k])
           for k in ("input_binding_sha256", "target_binding_sha256", "frozen_encoder_state_sha256")):
        raise ValueError("source and target binding SHA256 required")
    if header["frozen_encoder_state_sha256"] != encoder_sha:
        raise ValueError("header encoder tensor-content fingerprint mismatch")
    return _json_sha(header)


def train_paired_surface_models(examples, *, budget, frozen_encoder, loss_contexts=None, progress=None, contract='legacy'):
    """A/B/C same initialization, fixed exposure order, final-only core.

    Each attempted update consumes exactly four listed microbatch1 examples.
    Unknown microbatches contribute zero and are counted separately; entirely
    unknown groups skip AdamW and are NOT replenished with additional samples.
    Report actual optimizer steps, rather than calling attempted groups steps.
    """
    if contract not in ('legacy','window_surface_set_v1'):
        raise ValueError('explicit versioned training contract required')
    model_type=SurfaceRelationModelV1;loss_function=surface_relation_losses
    if contract=='window_surface_set_v1':
        if loss_contexts is not None:raise ValueError('window contract cannot mix legacy reference losses')
        from .gse_window_surface_model_v1 import WindowSurfaceModelV1
        from .gse_window_set_loss_v1 import window_set_loss
        model_type=WindowSurfaceModelV1;loss_function=window_set_loss
    if (type(budget) is not SurfaceTrainingBudget or type(budget.updates) is not int or not 1 <= budget.updates <= 2000
            or type(budget.seed) is not int or budget.seed < 0 or type(budget.accumulation_steps) is not int
            or budget.accumulation_steps != 4 or type(budget.learning_rate) is not float or budget.learning_rate != .001
            or type(budget.weight_decay) is not float or budget.weight_decay != .0001):
        raise ValueError("frozen1..2000 attempted updates, seed, AdamW.001/wd.0001 and micro1x4 required")
    device = torch.device(budget.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested device unavailable; no silent CPU fallback")
    if not isinstance(examples, (list, tuple)) or not examples or any(type(e) is not SurfaceTrainingExample for e in examples):
        raise ValueError("nonempty explicit in-memory example population required")
    encoder_sha = module_state_sha256(frozen_encoder)
    if any(p.requires_grad for p in frozen_encoder.parameters()) or any(m.training for m in frozen_encoder.modules()):
        raise ValueError("encoder must already be frozen and eval; no implicit encoder mutation")
    header_hashes = tuple(_validate_header(e.header, encoder_sha) for e in examples)
    if len({e.header["observation_id"] for e in examples}) != len(examples):
        raise ValueError("duplicate observation IDs in population")
    for example in examples:
        if (type(example.compact) is not CompactFrozenDualPathFeatures or type(example.patches) is not SurfacePatchBatch
                or type(example.targets) is not SurfaceLossTargets or example.compact.points_xyz_m.shape[0] != 1
                or example.targets.anchor_valid.shape[0] != 1 or example.patches.valid.shape[0] != 1):
            raise ValueError("microbatch1 typed compact/patch/target population required")
        if example.compact.points_xyz_m.dtype != torch.float32:
            raise ValueError("fixed float32 training inputs required")

    def guard():
        if tuple(_json_sha(e.header) for e in examples) != header_hashes:
            raise ValueError("actual source cache header changed")
        if (any(p.requires_grad for p in frozen_encoder.parameters()) or any(m.training for m in frozen_encoder.modules())
                or module_state_sha256(frozen_encoder) != encoder_sha):
            raise ValueError("frozen encoder value/buffer/mode/trainability drift")

    if loss_contexts is not None:
        from .gse_surface_training_context_v1 import validate_context, source_bound_loss
        if not isinstance(loss_contexts,(tuple,list)) or len(loss_contexts)!=len(examples):
            raise ValueError('one source loss context per population observation required')
        for example,context in zip(examples,loss_contexts,strict=True):
            validate_context(example,context)

    def compute_loss(prediction,target,index):
        if loss_contexts is None:
            return loss_function(prediction,target),None
        return source_bound_loss(prediction,examples[index],loss_contexts[index])

    rng = np.random.default_rng(budget.seed); flat = []
    while len(flat) < budget.updates * 4:
        flat.extend(rng.permutation(len(examples)).tolist())
    schedule = tuple(tuple(flat[i:i+4]) for i in range(0, budget.updates*4, 4))
    # CPU-only initialization changes no caller RNG state and reads no weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(budget.seed)
        shared = _snapshot(model_type("A"))
    initial_sha = state_dict_sha256(shared)
    final_states, initial_hashes, evaluations, histories, counts = {}, {}, {}, {}, {}

    def evaluate(model):
        model.eval(); snapshots = []
        with torch.no_grad():
            for index,example in enumerate(examples):
                guard()
                compact, patches, target = (_move(v,device) for v in (example.compact,example.patches,example.targets))
                prediction = model.forward_compact(compact,patches); guard()
                losses,source_evidence = compute_loss(prediction,target,index)
                snapshots.append(dict(prediction=_move(prediction,"cpu"),
                    diagnostic_loss_terms={name:float(value) for name,value in losses.terms.items()},
                    denominators=dict(losses.denominators), has_supervision=losses.has_supervision,
                    source_loss_evidence=source_evidence))
        return tuple(snapshots)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for branch in ("A","B","C"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(budget.seed)
                model = model_type(branch)
            model.load_state_dict(shared,strict=True); model.to(device)
            initial_hashes[branch] = state_dict_sha256(model.state_dict())
            if initial_hashes[branch] != initial_sha:
                raise ValueError("paired initialization drift")
            for name in FROZEN_READOUT_MODULES:
                getattr(model,name).requires_grad_(False)
            frozen_heads = {name:value.clone() for name,value in model.state_dict().items()
                            if name.split(".")[0] in FROZEN_READOUT_MODULES}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=budget.learning_rate,weight_decay=budget.weight_decay)
            evaluations[branch] = {"initial":evaluate(model)}
            history = []; actual_steps = effective_microbatches = skipped_groups = 0
            for group_index, indices in enumerate(schedule):
                guard(); model.train(); optimizer.zero_grad(set_to_none=True)
                effective = 0; group_loss = 0.; denominators = {}
                for index in indices:
                    example = examples[index]
                    compact,patches,target = (_move(v,device) for v in (example.compact,example.patches,example.targets))
                    prediction = model.forward_compact(compact,patches); guard()
                    loss,_ = compute_loss(prediction,target,index)
                    if not bool(torch.isfinite(loss.total)):
                        raise ValueError("nonfinite paired training loss")
                    for name,count in loss.denominators.items():denominators[name]=denominators.get(name,0)+count
                    if loss.has_supervision:
                        (loss.total/4).backward(); effective += 1
                        group_loss += float(loss.total.detach())/4
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("nonfinite head gradients")
                if effective:
                    optimizer.step(); actual_steps += 1
                else:
                    skipped_groups += 1
                guard(); effective_microbatches += effective
                history.append(dict(attempted_update=group_index+1,example_indices=indices,
                    optimizer_step_applied=bool(effective),effective_microbatches=effective,
                    diagnostic_loss=group_loss,denominators=denominators))
                if progress is not None:
                    progress(dict(branch=branch,**history[-1]))
            evaluations[branch]["final"] = evaluate(model)
            for name,value in frozen_heads.items():
                if not torch.equal(model.state_dict()[name],value):
                    raise ValueError("untrained reliability readout changed despite freezing")
            final_states[branch] = _snapshot(model); histories[branch] = tuple(history)
            counts[branch] = dict(attempted_update_groups=budget.updates,optimizer_steps=actual_steps,
                skipped_all_unknown_groups=skipped_groups,training_microbatch_exposures=budget.updates*4,
                effective_supervised_microbatches=effective_microbatches,
                unique_population_observations=len(examples),initial_eval_windows=len(examples),final_eval_windows=len(examples),
                encoder_forward_windows=0,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),microbatch=1,gradient_accumulation=4)
            del model,optimizer
        guard()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    return PairedSurfaceTrainingResult(shared,initial_hashes,final_states,evaluations,histories,counts,
        schedule,_json_sha(schedule),header_hashes,encoder_sha)


__all__ = ["SurfaceTrainingExample", "SurfaceTrainingBudget", "PairedSurfaceTrainingResult",
           "train_paired_surface_models", "state_dict_sha256", "module_state_sha256", "FROZEN_READOUT_MODULES"]
