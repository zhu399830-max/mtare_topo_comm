#!/usr/bin/env python3
"""V6: preserve the proven direction branch while adapting count/role embedding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

import train_aee_corrective_full_encoder_v3 as base
import train_aee_corrective_staged_balanced_heads_v5 as v5
from mtare_topo.data.aee_domain_adaptation import sha256


STEPS_PER_EPOCH = v5.STEPS_PER_EPOCH
_DIRECTION_PARAMETERS: list[torch.nn.Parameter] = []
_DIRECTION_STAGE1_SHA256: str | None = None


def _tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for index, tensor in enumerate(tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(index).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_direction_digest(state: dict[str, torch.Tensor]) -> str:
    # state_dict preserves module registration order, matching named_parameters.
    names = [name for name in state if name.startswith("direction_head.")]
    if not names:
        raise RuntimeError("V6 checkpoint has no direction tensors")
    return _tensor_digest([state[name] for name in names])


def make_embedding_adaptation_trainable(model):
    """Freeze encoder; adapt embedding and all heads during their fixed stages."""
    global _DIRECTION_PARAMETERS
    trainable, frozen = [], []
    _DIRECTION_PARAMETERS = []
    allowed = base.EMBEDDING_PREFIXES + base.SEMANTIC_HEAD_PREFIXES
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(allowed)
        (trainable if parameter.requires_grad else frozen).append(name)
        if name.startswith("direction_head."):
            _DIRECTION_PARAMETERS.append(parameter)
    if not _DIRECTION_PARAMETERS or any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError("V6 trainable-parameter contract failed")
    if any(not name.startswith(base.ENCODER_PREFIXES) for name in frozen):
        raise RuntimeError("V6 encoder freeze contract failed")
    if not any(name.startswith(base.EMBEDDING_PREFIXES) for name in trainable):
        raise RuntimeError("V6 embedding must be trainable")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


class EmbeddingStagedAdamW(torch.optim.AdamW):
    """Update direction for 47 steps, then preserve its tensors exactly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_step = 0

    def step(self, closure=None):
        global _DIRECTION_STAGE1_SHA256
        if self.contract_step >= STEPS_PER_EPOCH:
            for parameter in _DIRECTION_PARAMETERS:
                parameter.grad = None
        result = super().step(closure)
        self.contract_step += 1
        if self.contract_step == STEPS_PER_EPOCH:
            _DIRECTION_STAGE1_SHA256 = _tensor_digest(_DIRECTION_PARAMETERS)
        return result


def v6_gate(adapted_dense, adapted_sparse, source_dense, source_sparse, b0_sparse, finite, encoder_changed, embedding_changed, heads_changed):
    return {
        "sparse_direction_improvement_0p05": adapted_sparse["direction"]["f1"] >= source_sparse["direction"]["f1"] + 0.05,
        "sparse_direction_vs_b0": adapted_sparse["direction"]["f1"] >= b0_sparse["f1"],
        "sparse_empty_rate": adapted_sparse["direction"]["empty_rate"] <= 0.05,
        "sparse_count_1_to_4_macro_f1": adapted_sparse["count"]["macro_f1_count_1_to_4"] >= 0.70,
        "sparse_role_macro_f1": adapted_sparse["role"]["macro_f1_present"] >= 0.70,
        "dense_direction_retention": adapted_dense["direction"]["f1"] >= source_dense["direction"]["f1"] - 0.02,
        "all_model_tensors_finite": finite,
        "encoder_identity": not encoder_changed,
        "embedding_changed": embedding_changed,
        "semantic_heads_changed": heads_changed,
    }


def main() -> int:
    global _DIRECTION_STAGE1_SHA256
    _DIRECTION_STAGE1_SHA256 = None
    dataset_run = Path(sys.argv[sys.argv.index("--dataset-run") + 1]).resolve()
    v5.audit_effective_mass(dataset_run)
    base.make_full_model_trainable = make_embedding_adaptation_trainable
    base.weighted_multitask_loss = v5.staged_balanced_loss
    base._gate_metrics = v6_gate
    base.torch.optim.AdamW = EmbeddingStagedAdamW
    code = base.main()
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final_direction_sha256 = _checkpoint_direction_digest(checkpoint["model"])
    if _DIRECTION_STAGE1_SHA256 is None or final_direction_sha256 != _DIRECTION_STAGE1_SHA256:
        raise RuntimeError("V6 direction changed after the fixed stage-1 boundary")
    checkpoint["mode"] = "M1D_AEE_CORRECTIVE_EMBEDDING_ADAPTATION_V6"
    checkpoint["config"].update(
        direction_loss="equal_positive_negative_soft_mass_v1",
        count_class_weights=v5.COUNT_CLASS_WEIGHTS.tolist(),
        role_class_weights=v5.ROLE_CLASS_WEIGHTS.tolist(),
        direction_update_epochs=1,
        embedding_count_role_update_epochs=10,
        direction_stage1_sha256=_DIRECTION_STAGE1_SHA256,
    )
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        schema_version="aee_corrective_embedding_adaptation_seed_summary_v6",
        status="COMPLETED_AEE_CORRECTIVE_EMBEDDING_ADAPTATION_SEED_V6",
        best_checkpoint_sha256=sha256(checkpoint_path),
        direction_loss="equal_positive_negative_soft_mass_v1",
        count_effective_mass=v5.COUNT_EFFECTIVE_MASS.tolist(),
        role_effective_mass=v5.ROLE_EFFECTIVE_MASS.tolist(),
        count_class_weights=v5.COUNT_CLASS_WEIGHTS.tolist(),
        role_class_weights=v5.ROLE_CLASS_WEIGHTS.tolist(),
        staged_update={"direction_epochs": 1, "embedding_count_role_epochs": 10, "steps_per_epoch": STEPS_PER_EPOCH},
        representation_contract="encoder_frozen_embedding_adapted_direction_stage1_exact",
        direction_stage1_sha256=_DIRECTION_STAGE1_SHA256,
        final_direction_sha256=final_direction_sha256,
        direction_stage1_identity=True,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
