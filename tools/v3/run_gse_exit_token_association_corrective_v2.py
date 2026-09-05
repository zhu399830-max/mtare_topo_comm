#!/usr/bin/env python3
"""Execute the immutable exit-token-aware open-set corrective V2."""

from __future__ import annotations

import run_gse_open_set_association_corrective_v1 as base


base.RUN_ID = "gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
base.PASS_STATUS = "PASS_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
base.FAIL_STATUS = "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
base.DATA_CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
base.SUMMARY_SCHEMA_VERSION = "gse_exit_token_association_corrective_v2"
base.TRAINER = base.PROJECT_ROOT / "tools/v3/train_gse_exit_token_association_v2.py"
base.MODEL_EXPECTED_PARAMETERS = 112898
base.MODEL_REQUIRED_FILES = (
    "best.pt",
    "normalization.npz",
    "selection_outputs.npz",
    "epoch_metrics.jsonl",
    "frozen_observation_features.npy",
    "frozen_exit_token_outputs.npz",
    "summary.json",
)


if __name__ == "__main__":
    raise SystemExit(base.main())
