#!/usr/bin/env python3
"""Load one deployment checkpoint and emit deterministic forward evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mtare_topo.deployment.m1d_checkpoint import DEPLOYMENT_SCHEMA
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity output: {args.output}")
    payload = torch.load(args.checkpoint, map_location="cpu")
    if payload.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise RuntimeError("deployment schema mismatch")
    model = StructuralSemanticNet().cpu().eval()
    model.load_state_dict(payload["model"], strict=True)
    total = 2 * 16 * 720
    probe = torch.arange(total, dtype=torch.float32).reshape(1, 2, 16, 720)
    probe = torch.remainder(probe * 0.61803398875, 997.0) / 997.0
    with torch.inference_mode():
        result = model(probe)
    arrays = {name: value.detach().cpu().numpy() for name, value in result.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "schema_version": "m1d_ros_checkpoint_runtime_verification_v1",
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": int(payload["seed"]),
        "mode": payload["mode"],
        "runtime_contract": payload.get("runtime_contract"),
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "tensor_sha256": {name: tensor_sha256(value) for name, value in payload["model"].items()},
        "outputs": {name: {"shape": list(value.shape), "finite": bool(np.isfinite(value).all())} for name, value in arrays.items()},
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
