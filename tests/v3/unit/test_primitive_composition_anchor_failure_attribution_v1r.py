from pathlib import Path

import run_primitive_composition_anchor_failure_attribution_v1r as corrective


def test_corrective_changes_only_the_anchor_root_binding():
    root = corrective._CorrectedAnchorRoot()
    assert root / "artifacts/anchors" == corrective.ANCHOR_TARGET_ROOT
    assert corrective.ANCHOR_TARGET_ROOT.name == "anchor_targets"
    assert corrective.ANCHOR_TARGET_ROOT.parent.name == "materialized"


def test_corrective_has_new_immutable_run_identity():
    assert corrective.RUN_ID.endswith("_failure_attribution_v1r_seed0")
    assert corrective.CARD_STATUS.endswith("FAILURE_ATTRIBUTION_V1R")
