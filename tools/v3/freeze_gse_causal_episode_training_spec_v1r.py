#!/usr/bin/env python3
"""Freeze the V1R system-corrective causal episode training specification."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
import freeze_gse_causal_episode_training_spec_v1 as base


base.RUN_ID = "gate3_20260827_gse_causal_episode_training_v1r_seed0"
base.CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_training_v1r.json"
base.SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_episode_training_v1r.json"
base.CARD_ID = "gse_causal_episode_training_v1r"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_TRAINING_V1R"
base.SLUG = "gse_causal_episode_training_v1r"
base.FREEZER_PATH = "tools/v3/freeze_gse_causal_episode_training_spec_v1r.py"
base.SYSTEM_PREDECESSOR = "results/gate3_semantics/gate3_20260827_gse_causal_episode_training_v1_seed0"


if __name__ == "__main__":
    raise SystemExit(base.main())
