#!/home/zeng-workstation/anaconda3/bin/python
"""Execute and seal three-seed V4 balanced semantic-head adaptation."""

from __future__ import annotations

from typing import Any

import run_aee_corrective_full_encoder_v3 as base


RUN_ID = "gate2_20260822_aee_corrective_balanced_heads_v4_seed20260822"
TRAINER = base.PROJECT_ROOT / "tools/v3/train_aee_corrective_balanced_heads_v4.py"
STATUS_PASS = "PASS_AEE_CORRECTIVE_BALANCED_HEADS_V4"
STATUS_FAIL = "FAIL_AEE_CORRECTIVE_BALANCED_HEADS_V4"


def validate_seed_summary(summary: dict[str, Any], seed: int) -> None:
    if summary.get("status") != "COMPLETED_AEE_CORRECTIVE_BALANCED_HEADS_SEED_V4":
        raise RuntimeError(f"V4 seed {seed} did not complete")
    if summary.get("seed") != seed or summary.get("epochs_completed") != 10:
        raise RuntimeError(f"V4 seed/epoch identity drift: {seed}")
    if summary.get("independent_train_samples_per_epoch") != {"cano": 5000, "aee": 1000}:
        raise RuntimeError(f"V4 independent sample drift: {seed}")
    if summary.get("training_views_per_epoch") != {"cano_dense": 5000, "cano_mask_matched": 5000, "aee": 1000}:
        raise RuntimeError(f"V4 view count drift: {seed}")
    if summary.get("direction_loss") != "per_frame_equal_positive_negative_soft_mass_v1" or summary.get("representation_contract") != "encoder_and_embedding_frozen_exact":
        raise RuntimeError(f"V4 loss/representation contract drift: {seed}")
    if any(summary.get(key) != 0 for key in ("c09_frames_read", "c10_frames_read", "formal_benchmark_frames_read")):
        raise RuntimeError(f"V4 forbidden data read: {seed}")
    expected = {
        "sparse_direction_improvement_0p05", "sparse_direction_vs_b0", "sparse_empty_rate",
        "sparse_count_1_to_4_macro_f1", "sparse_role_macro_f1", "dense_direction_retention",
        "all_model_tensors_finite", "encoder_identity", "embedding_identity", "semantic_heads_changed",
    }
    if not isinstance(summary.get("gate"), dict) or set(summary["gate"]) != expected:
        raise RuntimeError(f"V4 gate schema drift: {seed}")


base.RUN_ID = RUN_ID
base.TRAINER = TRAINER
base.STATUS_PASS = STATUS_PASS
base.STATUS_FAIL = STATUS_FAIL
base.validate_seed_summary = validate_seed_summary


if __name__ == "__main__":
    raise SystemExit(base.main())
