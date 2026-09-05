import numpy as np

import evaluate_gse_composer_observability_v1 as evaluation


def _source(seed: int) -> dict[str, np.ndarray]:
    event = np.asarray([0, 0, 0, 0, 1, 2, 3, 4] * 2, dtype=np.int8)
    split = np.asarray(["C07"] * 8 + ["C08"] * 8, dtype=object)
    feature = np.eye(5, dtype=np.float32)[event]
    feature = np.concatenate(
        (feature, np.full((len(event), 1), seed * 0.01, dtype=np.float32)), axis=1
    )
    probability = np.full((len(event), 5), 0.01, dtype=np.float32)
    probability[np.arange(len(event)), event] = 0.96
    scores = {
        name: (event == index).astype(np.float32) + seed * 0.001
        for index, name in enumerate(evaluation.EVENT_NAMES[1:], 1)
    }
    return {
        "global_sequence_index": np.arange(len(event), dtype=np.int64),
        "feature_matrix": feature,
        "event_probability": probability,
        **{f"train_free_{name}": value for name, value in scores.items()},
        "event_index": event,
        "identity": np.asarray(
            [None if value == 0 else f"event-{value}" for value in event], dtype=object
        ),
        "parent": np.asarray([f"p-{value}" for value in split], dtype=object),
        "family": np.asarray(["S01"] * len(event), dtype=object),
        "split": split,
    }


def test_metric_rows_fit_twelve_seed_probes_and_score_ensemble() -> None:
    sources = {f"seed{seed}": _source(seed) for seed in range(3)}
    reference = evaluation._ensemble(list(sources.values()))
    assert "feature_matrix" not in reference
    rows, probe_fits = evaluation._metric_rows(sources, reference)
    assert probe_fits == 12
    assert len(rows) == 48
    assert sum(row["source"] == "ensemble" for row in rows) == 12
    for mode in ("train_free", "independent_head", "explicit_probe"):
        selected = [
            row
            for row in rows
            if row["source"] == "ensemble" and row["mode"] == mode
        ]
        assert len(selected) == 4
        assert all(row["transfer_roc_auc"] == 1.0 for row in selected)


def test_ensemble_rejects_cross_seed_identity_drift() -> None:
    seeds = [_source(seed) for seed in range(3)]
    seeds[2]["global_sequence_index"] = seeds[2]["global_sequence_index"].copy()
    seeds[2]["global_sequence_index"][0] = 999
    try:
        evaluation._ensemble(seeds)
    except RuntimeError as exc:
        assert "alignment drift" in str(exc)
    else:
        raise AssertionError("cross-seed identity drift was accepted")
