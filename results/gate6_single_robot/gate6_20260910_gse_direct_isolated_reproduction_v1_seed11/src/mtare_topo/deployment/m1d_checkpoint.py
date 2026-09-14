"""Lossless M1D checkpoint export for the pinned ROS Python 3.8 runtime."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


DEPLOYMENT_SCHEMA = "m1d_ros_deployment_state_v1"
COMPOSITE_V9_MODE = "M1D_AEE_CORRECTIVE_COMPOSITE_V9"
COMPOSITE_V9_RUNTIME_CONTRACT = {
    "direction": "model.direction_logits",
    "count": "frozen_b0.branch_count",
    "role": "terminal_if_count_le_1_interior_if_2_junction_if_ge_3",
    "neural_count_role": "diagnostic_only",
}
SUPPORTED_SOURCE_MODES = ("M1D", "M1D_AEE_HEAD_ADAPTED_V1", COMPOSITE_V9_MODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_source(
    payload: Mapping[str, Any], expected_seed: int, expected_mode: str
) -> Mapping[str, torch.Tensor]:
    if expected_mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(f"unsupported deployment source mode: {expected_mode}")
    if payload.get("mode") != expected_mode or int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("source checkpoint mode/seed identity mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source checkpoint lacks a non-empty model state")
    expected = StructuralSemanticNet().state_dict()
    if list(state) != list(expected):
        raise ValueError("source checkpoint parameter names/order differ from StructuralSemanticNet")
    for name, reference in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"invalid source tensor contract: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite source tensor: {name}")
    return state


def verify_m1d_forward_parity(
    source_state: Mapping[str, torch.Tensor], deployment_state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if list(source_state) != list(deployment_state):
        raise ValueError("source/deployment parameter keys differ")
    parameter_mismatches = []
    for name in source_state:
        if not torch.equal(source_state[name].detach().cpu(), deployment_state[name].detach().cpu()):
            parameter_mismatches.append(name)
    source_model = StructuralSemanticNet().cpu().eval()
    deployment_model = StructuralSemanticNet().cpu().eval()
    source_model.load_state_dict(source_state, strict=True)
    deployment_model.load_state_dict(deployment_state, strict=True)
    total = 2 * 16 * 720
    probe = torch.arange(total, dtype=torch.float32).reshape(1, 2, 16, 720)
    probe = torch.remainder(probe * 0.61803398875, 997.0) / 997.0
    with torch.inference_mode():
        source_output = source_model(probe)
        deployment_output = deployment_model(probe)
    output_max_abs = {
        name: float(torch.max(torch.abs(source_output[name] - deployment_output[name])).item())
        for name in source_output
    }
    passed = not parameter_mismatches and all(math.isfinite(value) and value == 0.0 for value in output_max_abs.values())
    return {
        "parameter_count": len(source_state),
        "parameter_mismatches": parameter_mismatches,
        "output_max_abs": output_max_abs,
        "bit_exact_same_runtime": passed,
    }


def export_m1d_ros_checkpoint(
    *,
    source: Path,
    destination: Path,
    expected_source_sha256: str,
    expected_seed: int,
    expected_mode: str = "M1D",
) -> dict[str, Any]:
    """Export tensors plus primitive metadata; never train or mutate weights."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite deployment checkpoint: {destination}")
    actual_source_hash = sha256(source)
    if actual_source_hash != expected_source_sha256:
        raise ValueError(f"source checkpoint hash drift: {actual_source_hash}")
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    source_state = _validate_source(source_payload, expected_seed, expected_mode)
    runtime_contract = source_payload.get("runtime_contract")
    if expected_mode == COMPOSITE_V9_MODE:
        if runtime_contract != COMPOSITE_V9_RUNTIME_CONTRACT:
            raise ValueError("V9 source checkpoint runtime contract mismatch")
    elif runtime_contract is not None:
        raise ValueError("legacy deployment source unexpectedly declares a runtime contract")
    frozen_state = OrderedDict((str(name), value.detach().cpu().contiguous().clone()) for name, value in source_state.items())
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "model_class": "StructuralSemanticNet",
        "mode": expected_mode,
        "seed": int(expected_seed),
        "source_checkpoint_sha256": actual_source_hash,
        "source_epoch": int(source_payload.get("epoch", -1)),
        "model": frozen_state,
    }
    if runtime_contract is not None:
        deployment["runtime_contract"] = dict(runtime_contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deployment, destination, pickle_protocol=2, _use_new_zipfile_serialization=False)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    if loaded.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise RuntimeError("deployment checkpoint did not round-trip")
    parity = verify_m1d_forward_parity(source_state, loaded["model"])
    if not parity["bit_exact_same_runtime"]:
        raise RuntimeError(f"deployment checkpoint parity failed: {parity}")
    return {
        "schema_version": DEPLOYMENT_SCHEMA,
        "source": str(source),
        "source_sha256": actual_source_hash,
        "destination": str(destination),
        "destination_sha256": sha256(destination),
        "seed": int(expected_seed),
        "mode": expected_mode,
        "runtime_contract": runtime_contract,
        "source_epoch": int(source_payload.get("epoch", -1)),
        "tensor_sha256": {name: _tensor_digest(value) for name, value in frozen_state.items()},
        "parity": parity,
        "training_steps": 0,
        "optimizer_steps": 0,
    }
