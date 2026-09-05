#!/usr/bin/env python3
"""Numerical-contract corrective for the sealed V1R attribution runner."""

from __future__ import annotations

from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_primitive_composition_anchor_failure_attribution_v1 as source


RUN_ID = "gate3_20260903_primitive_composition_anchor_failure_attribution_v1r2_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_FAILURE_ATTRIBUTION_V1R2"
ANCHOR_TARGET_ROOT = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0/"
    "artifacts/materialized/anchor_targets"
)
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_primitive_composition_anchor_failure_attribution_v1r2.py"
ACTUAL_NUMERICAL_CONTRACT = {
    "deterministic_algorithms": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "reason": "explicit reproduction of the source C07 evaluator's actual fresh-process state",
}


class _CorrectedAnchorRoot:
    def __truediv__(self, suffix: str) -> Path:
        if suffix != "artifacts/anchors":
            raise RuntimeError(f"unexpected V1 anchor suffix: {suffix}")
        return ANCHOR_TARGET_ROOT


def main() -> int:
    source.RUN_ID = RUN_ID
    source.CARD_STATUS = CARD_STATUS
    source.ANCHORS = _CorrectedAnchorRoot()
    original_run = source.subprocess.run
    original_write = source.write_json

    def corrected_run(command, *args, **kwargs):
        command = list(command)
        old = str(PROJECT_ROOT / "tools/v3/execute_primitive_composition_anchor_failure_attribution_v1.py")
        if old in command:
            command[command.index(old)] = str(EXECUTOR)
        return original_run(command, *args, **kwargs)

    def corrected_write(path, value):
        if Path(path).name == "environment.json" and isinstance(value, dict):
            value = dict(value)
            value["numerical_contract"] = ACTUAL_NUMERICAL_CONTRACT
        return original_write(path, value)

    source.subprocess.run = corrected_run
    source.write_json = corrected_write
    return source.main()


if __name__ == "__main__":
    raise SystemExit(main())
