#!/usr/bin/env python3
"""CUDA-generator corrective for local composition-slot readiness V1.

V1 reached the post-backward slot-permutation probe, then constructed a CPU
``torch.Generator`` while requesting a CUDA ``randperm`` result.  This wrapper
keeps the frozen V1 implementation intact and changes only that device bridge:
the same fixed-seed permutation is generated on CPU and copied to CUDA.
"""

from __future__ import annotations

import torch

import run_primitive_local_composition_slot_readiness_v1 as base


RUN_ID = "gate3_20260903_primitive_local_composition_slot_readiness_v1r_seed0"
PASS = "PASS_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1R"
FAIL = "FAIL_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1R"


def main() -> int:
    base.RUN_ID = RUN_ID
    base.PASS = PASS
    base.FAIL = FAIL
    base.CARD_STATUS = CARD_STATUS
    original_randperm = base.torch.randperm

    def cpu_seeded_randperm(*args, **kwargs):
        device = kwargs.get("device")
        generator = kwargs.get("generator")
        if device is not None and torch.device(device).type == "cuda" and generator is not None:
            kwargs = dict(kwargs)
            kwargs["device"] = torch.device("cpu")
            return original_randperm(*args, **kwargs).to(device)
        return original_randperm(*args, **kwargs)

    base.torch.randperm = cpu_seeded_randperm
    try:
        return base.main()
    finally:
        base.torch.randperm = original_randperm


if __name__ == "__main__":
    raise SystemExit(main())
