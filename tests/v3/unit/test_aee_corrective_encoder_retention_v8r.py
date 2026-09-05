"""V8R evidence-interface corrective contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("v8r", TOOLS / "train_aee_corrective_encoder_retention_v8r.py")
assert spec and spec.loader
v8r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8r)


def test_v8r_frozen_effective_mass_and_weights_are_serializable() -> None:
    assert v8r.v5.COUNT_EFFECTIVE_MASS.tolist() == [1244, 6793, 1329, 504, 105, 25]
    assert v8r.v5.ROLE_EFFECTIVE_MASS.tolist() == [6495, 2105, 1400]
    assert len(v8r.v5.COUNT_CLASS_WEIGHTS.tolist()) == 6
    assert len(v8r.v5.ROLE_CLASS_WEIGHTS.tolist()) == 3


def test_v8r_changes_no_training_mechanism() -> None:
    assert v8r.STEPS_PER_EPOCH == 47
    assert v8r.retained_balanced_loss is not None
    assert v8r.EncoderRetentionAdamW is not None
