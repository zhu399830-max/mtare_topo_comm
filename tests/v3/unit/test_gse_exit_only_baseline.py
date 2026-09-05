from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_exit_only_baseline import (
    five_event_scores,
    m1d_role_logits_to_gse,
    m1d_role_probabilities_to_gse,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


def test_role_probabilities_have_fixed_five_event_embedding() -> None:
    result = m1d_role_probabilities_to_gse(np.asarray(((0.2, 0.3, 0.5),)))
    assert result.shape == (1, 5)
    assert result[0, EVENT_NAMES.index("corridor")] == pytest.approx(0.2)
    assert result[0, EVENT_NAMES.index("junction")] == pytest.approx(0.3)
    assert result[0, EVENT_NAMES.index("terminal")] == pytest.approx(0.5)
    assert result[0, EVENT_NAMES.index("turn")] == 0.0
    assert result[0, EVENT_NAMES.index("geometry_transition")] == 0.0


def test_role_logits_use_stable_softmax() -> None:
    result = m1d_role_logits_to_gse(np.asarray(((1001.0, 1000.0, 999.0),)))
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)
    assert result.argmax(axis=1).tolist() == [EVENT_NAMES.index("corridor")]


def test_five_event_scores_penalize_unavailable_geometric_events() -> None:
    target = [0, 1, 2, 3, 4]
    roles = np.eye(3)[[0, 1, 2, 0, 0]]
    result = five_event_scores(target, m1d_role_probabilities_to_gse(roles))
    assert result["per_class"]["corridor"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["turn"]["f1"] == 0.0
    assert result["per_class"]["geometry_transition"]["f1"] == 0.0
    assert result["macro_f1"] < 1.0


def test_role_adapter_rejects_non_normalized_probabilities() -> None:
    with pytest.raises(ValueError, match="normalized"):
        m1d_role_probabilities_to_gse(np.asarray(((0.2, 0.3, 0.4),)))
