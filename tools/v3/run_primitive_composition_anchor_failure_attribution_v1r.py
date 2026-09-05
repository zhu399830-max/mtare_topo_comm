#!/usr/bin/env python3
"""Path-binding corrective for the sealed V1 C07 attribution runner."""

from __future__ import annotations

from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_primitive_composition_anchor_failure_attribution_v1 as source


RUN_ID = "gate3_20260903_primitive_composition_anchor_failure_attribution_v1r_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_FAILURE_ATTRIBUTION_V1R"
ANCHOR_TARGET_ROOT = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0/"
    "artifacts/materialized/anchor_targets"
)


class _CorrectedAnchorRoot:
    """Preserve V1 code while replacing its single incorrect joined path."""

    def __truediv__(self, suffix: str) -> Path:
        if suffix != "artifacts/anchors":
            raise RuntimeError(f"unexpected V1 anchor suffix: {suffix}")
        return ANCHOR_TARGET_ROOT


def main() -> int:
    source.RUN_ID = RUN_ID
    source.CARD_STATUS = CARD_STATUS
    source.ANCHORS = _CorrectedAnchorRoot()
    return source.main()


if __name__ == "__main__":
    raise SystemExit(main())
