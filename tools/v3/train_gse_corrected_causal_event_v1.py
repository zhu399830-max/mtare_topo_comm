#!/usr/bin/env python3
"""Train the directional event head against the corrected causal Teacher."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.representation.gse_corrected_causal_event import (
    OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1,
    corrected_causal_event_gate,
)
from mtare_topo.representation.gse_rare_event_corrective import (
    evaluate_rare_event_corrective,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES
import train_gse_directional_structural_event_v1 as base


PASS_STATUS = "PASS_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1"
CORRECTED_TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
OLD_DIRECTIONAL = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260826_gse_directional_structural_event_training_v1_seed0"
)
EXPECTED_EVENT_COUNTS = {
    0: (114617, 36347),
    1: (19743, 6865),
    2: (5551, 1974),
    3: (1482, 516),
    4: (791, 240),
}
EXPECTED_IDENTITY_COUNTS = {
    1: (417, 146),
    2: (375, 128),
    3: (297, 95),
    4: (59, 17),
}


def _load_corrected_labels(
    compact_global: np.ndarray,
    compact_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_by_global = {int(value): row for row, value in enumerate(compact_global)}
    event = np.full(len(compact_global), -1, dtype=np.int64)
    identity_text: list[str | None] = [None] * len(compact_global)
    teacher_parent: list[str | None] = [None] * len(compact_global)
    event_index = {name: index for index, name in enumerate(EVENT_NAMES)}
    with (CORRECTED_TEACHER / "artifacts/teacher_observations.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            record = json.loads(line)
            try:
                row = row_by_global[int(record["global_sequence_index"])]
            except KeyError as exc:
                raise RuntimeError("corrected Teacher has an unknown global identity") from exc
            if event[row] >= 0:
                raise RuntimeError("corrected Teacher global identity is duplicated")
            event[row] = event_index[str(record["event"])]
            identity_text[row] = (
                None if record.get("identity") is None else str(record["identity"])
            )
            teacher_parent[row] = str(record["parent_id"])
    if np.any(event < 0) or any(value is None for value in teacher_parent):
        raise RuntimeError("corrected Teacher does not cover the complete C01-C08 population")
    if not np.array_equal(np.asarray(teacher_parent, dtype=str), compact_parent.astype(str)):
        raise RuntimeError("corrected Teacher parent/global identity alignment drift")
    identities = sorted({value for value in identity_text if value is not None})
    numeric = {identity: index for index, identity in enumerate(identities)}
    identity = np.asarray(
        [-1 if value is None else numeric[value] for value in identity_text], dtype=np.int64
    )
    return event, identity, np.asarray(identity_text, dtype=object)


def _load_population(
    dataset_run: Path,
    verifier_run: Path,
) -> tuple[GSESequenceDataset, dict[str, np.ndarray]]:
    dataset = GSESequenceDataset(dataset_run, "train", augment_azimuth=False)
    with np.load(verifier_run / "artifacts/pair_cache/pairs.npz", allow_pickle=False) as archive:
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        partition = archive["partition_code"].astype(np.uint8)
        parent = archive["parent_id"].astype(str)
    if (
        compact_global.shape != (188126,)
        or partition.shape != (188126,)
        or len(np.unique(compact_global)) != 188126
        or not np.all(np.diff(compact_global) > 0)
        or int(np.sum(partition == 0)) != 142184
        or int(np.sum(partition == 1)) != 45942
    ):
        raise RuntimeError("corrected causal event population/index contract drift")
    row_by_global = {
        int(record["global_sequence_index"]): row
        for row, record in enumerate(dataset.records)
    }
    try:
        dataset_row = np.asarray(
            [row_by_global[int(value)] for value in compact_global], dtype=np.int64
        )
    except KeyError as exc:
        raise RuntimeError("corrected event population is absent from the LiDAR dataset") from exc
    event, identity, _ = _load_corrected_labels(compact_global, parent)
    for event_index, expected in EXPECTED_EVENT_COUNTS.items():
        actual = tuple(
            int(np.sum((partition == partition_index) & (event == event_index)))
            for partition_index in (0, 1)
        )
        if actual != expected:
            raise RuntimeError(f"corrected event count drift for {EVENT_NAMES[event_index]}")
    for event_index, expected in EXPECTED_IDENTITY_COUNTS.items():
        actual = tuple(
            len(set(identity[(partition == partition_index) & (event == event_index)].tolist()))
            for partition_index in (0, 1)
        )
        if actual != expected:
            raise RuntimeError(f"corrected identity count drift for {EVENT_NAMES[event_index]}")
    if np.any((event == 0) != (identity < 0)):
        raise RuntimeError("corrected corridor/identity contract drift")
    probabilities = []
    for seed in (0, 1, 2):
        features = np.load(
            verifier_run / f"artifacts/models/seed{seed}/frozen_observation_features.npy",
            mmap_mode="r",
        )
        if features.shape != (188126, 146):
            raise RuntimeError(f"seed{seed} frozen observation feature drift")
        probability = np.asarray(features[:, :5], dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        probabilities.append(probability)
    return dataset, {
        "global_sequence_index": compact_global,
        "dataset_row": dataset_row,
        "partition": partition,
        "parent": parent,
        "family": np.asarray([value[:3] for value in parent]),
        "event": event,
        "identity": identity,
        "baseline_probability": np.mean(np.stack(probabilities), axis=0).astype(np.float32),
    }


def _baseline_evidence(
    dataset_run: Path,
    verifier_run: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _, arrays = _load_population(dataset_run, verifier_run)
    selection = np.where(arrays["partition"] == 1)[0]
    evidence: dict[str, Any] = {
        "backbone_event_heads": {"seeds": {}},
        "old_directional_head": {"seeds": {}},
    }
    backbone_probabilities = []
    for seed in (0, 1, 2):
        features = np.load(
            verifier_run / f"artifacts/models/seed{seed}/frozen_observation_features.npy",
            mmap_mode="r",
        )
        probability = np.asarray(features[selection, :5], dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        backbone_probabilities.append(probability)
        evidence["backbone_event_heads"]["seeds"][str(seed)] = evaluate_rare_event_corrective(
            probability,
            arrays["event"][selection],
            arrays["identity"][selection],
            arrays["family"][selection],
        )
    evidence["backbone_event_heads"]["ensemble"] = evaluate_rare_event_corrective(
        np.mean(np.stack(backbone_probabilities), axis=0),
        arrays["event"][selection],
        arrays["identity"][selection],
        arrays["family"][selection],
    )
    old_probabilities = []
    old_dir = OLD_DIRECTIONAL / "artifacts/training"
    for seed in (0, 1, 2):
        with np.load(old_dir / f"seed{seed}/selection_outputs.npz", allow_pickle=False) as archive:
            global_index = archive["global_sequence_index"].astype(np.int64)
            probability = archive["probability"].astype(np.float64)
        if not np.array_equal(global_index, arrays["global_sequence_index"][selection]):
            raise RuntimeError("old directional baseline selection identity drift")
        old_probabilities.append(probability)
        evidence["old_directional_head"]["seeds"][str(seed)] = evaluate_rare_event_corrective(
            probability,
            arrays["event"][selection],
            arrays["identity"][selection],
            arrays["family"][selection],
        )
    with np.load(old_dir / "ensemble_selection_outputs.npz", allow_pickle=False) as archive:
        ensemble_global = archive["global_sequence_index"].astype(np.int64)
        ensemble_probability = archive["probability"].astype(np.float64)
    if not np.array_equal(ensemble_global, arrays["global_sequence_index"][selection]):
        raise RuntimeError("old directional ensemble selection identity drift")
    old_ensemble = evaluate_rare_event_corrective(
        ensemble_probability,
        arrays["event"][selection],
        arrays["identity"][selection],
        arrays["family"][selection],
    )
    if not np.isclose(
        old_ensemble["event"]["macro_f1"],
        OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("strongest corrected-label event baseline drift")
    evidence["old_directional_head"]["ensemble"] = old_ensemble
    evidence["corrected_teacher_capacity"] = {
        "fit_event_counts": {
            EVENT_NAMES[index]: value[0] for index, value in EXPECTED_EVENT_COUNTS.items()
        },
        "selection_event_counts": {
            EVENT_NAMES[index]: value[1] for index, value in EXPECTED_EVENT_COUNTS.items()
        },
        "fit_identity_counts": {
            EVENT_NAMES[index]: value[0] for index, value in EXPECTED_IDENTITY_COUNTS.items()
        },
        "selection_identity_counts": {
            EVENT_NAMES[index]: value[1] for index, value in EXPECTED_IDENTITY_COUNTS.items()
        },
        "fit_change_point_families": sorted(
            set(arrays["family"][(arrays["partition"] == 0) & (arrays["event"] == 4)].tolist())
        ),
        "selection_change_point_families": sorted(
            set(arrays["family"][(arrays["partition"] == 1) & (arrays["event"] == 4)].tolist())
        ),
    }
    return evidence, arrays


_original_train = base.train


def _train_with_corrected_evidence(args) -> dict[str, Any]:
    result = _original_train(args)
    baseline, _ = _baseline_evidence(Path(args.dataset_run), Path(args.verifier_run))
    seed_gains = {}
    for seed in (0, 1, 2):
        current = float(result["seeds"][str(seed)]["metrics"]["event"]["macro_f1"])
        prior = float(
            baseline["old_directional_head"]["seeds"][str(seed)]["event"]["macro_f1"]
        )
        seed_gains[str(seed)] = current - prior
    ensemble_current = float(result["ensemble_metrics"]["event"]["macro_f1"])
    ensemble_prior = float(
        baseline["old_directional_head"]["ensemble"]["event"]["macro_f1"]
    )
    aggregate_requirements = {
        "ensemble_corrected_causal_gate": bool(result["ensemble_gate"]["passed"]),
        "ensemble_macro_f1_gain_at_least_0p05": ensemble_current - ensemble_prior >= 0.05,
        "at_least_two_seed_macro_f1_gains_at_least_0p05": sum(
            gain >= 0.05 for gain in seed_gains.values()
        )
        >= 2,
        "no_seed_macro_f1_regression": all(gain >= 0.0 for gain in seed_gains.values()),
        "selection_change_point_families_recorded_exact_4": len(
            baseline["corrected_teacher_capacity"]["selection_change_point_families"]
        )
        == 4,
    }
    scientific_pass = all(aggregate_requirements.values())
    result.update(
        {
            "schema_version": "gse_corrected_causal_event_training_v1",
            "overall_status": PASS_STATUS if scientific_pass else FAIL_STATUS,
            "scientific_pass": scientific_pass,
            "method": "corrected_teacher_three_seed_frozen_backbone_directional_binary_plus_conditional_event",
            "corrected_teacher_run": str(CORRECTED_TEACHER.relative_to(PROJECT_ROOT)),
            "baseline_evidence": baseline,
            "seed_macro_f1_gain_vs_old_directional": seed_gains,
            "ensemble_macro_f1_gain_vs_old_directional": ensemble_current - ensemble_prior,
            "aggregate_requirements": aggregate_requirements,
            "capacity_interpretation": "C07-C08 has 17 independent change-point identities across four families; this run is a development learnability gate, not the final all-family generalization claim.",
        }
    )
    output_dir = Path(args.output_dir).resolve()
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


base.PASS_STATUS = PASS_STATUS
base.FAIL_STATUS = FAIL_STATUS
base._load_population = _load_population
base.corrective_gate = corrected_causal_event_gate
base.train = _train_with_corrected_evidence


if __name__ == "__main__":
    raise SystemExit(base.main())
