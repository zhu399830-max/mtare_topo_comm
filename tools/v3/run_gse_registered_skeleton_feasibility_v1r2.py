#!/usr/bin/env python3
"""Final ERCSS V1R2 wrapper around the unchanged scientific V1R runner."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json
import run_gse_registered_skeleton_feasibility_v1r as base


def main() -> int:
    previous_root = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_registered_skeleton_feasibility_v1r_seed0"
    previous = load_json(previous_root / "metrics/summary.json")
    previous_log = (previous_root / "logs/01_registered_skeleton_feasibility.log").read_text(encoding="utf-8")
    if (
        "No module named 'torch'" not in previous_log
        or previous.get("registered_skeleton_feasibility")
        or previous.get("returncode") != 1
    ):
        raise RuntimeError("V1R was not the expected pre-world eager-Torch import failure")
    base.RUN_ID = "gate3_20260830_gse_registered_skeleton_feasibility_v1r2_seed0"
    base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R2"
    return base.main()


if __name__ == "__main__": raise SystemExit(main())
