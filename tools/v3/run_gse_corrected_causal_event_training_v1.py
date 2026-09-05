#!/usr/bin/env python3
"""Run and seal corrected causal structural-event training once."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
import run_gse_directional_structural_event_training_v1 as base


base.RUN_ID = "gate3_20260827_gse_corrected_causal_event_training_v1_seed0"
base.PASS_STATUS = "PASS_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1"
base.FAIL_STATUS = "FAIL_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1"
base.TRAINER = PROJECT_ROOT / "tools/v3/train_gse_corrected_causal_event_v1.py"


CORRECTED_TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
OLD_DIRECTIONAL = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260826_gse_directional_structural_event_training_v1_seed0"
)


def _sources() -> dict:
    sources = {
        "verifier": base.verify_failed_component_run_seal(
            PROJECT_ROOT, base.VERIFIER, "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
        ),
        "dataset": base.verify_complete_run_seal(
            PROJECT_ROOT, base.DATASET, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        ),
        "training": base.verify_complete_run_seal(
            PROJECT_ROOT, base.TRAINING, "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
        ),
        "corrected_teacher": base.verify_complete_run_seal(
            PROJECT_ROOT,
            CORRECTED_TEACHER,
            "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R",
        ),
        "old_directional_baseline": base.verify_failed_component_run_seal(
            PROJECT_ROOT,
            OLD_DIRECTIONAL,
            "FAIL_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1",
        ),
    }
    return sources


base._sources = _sources


if __name__ == "__main__":
    raise SystemExit(base.main())
