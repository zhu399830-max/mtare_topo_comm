#!/usr/bin/env python3
"""Audit whether V2R5 explicit outputs can support the two GSE composers.

This evaluator never reads encoder context, descriptors, GT identity as an
input, or the old event label.  Corrected identities are used only after
scoring to report event coverage.  A fixed linear probe is fitted on C07 and
applied unchanged to C08; the probe is diagnostic and is not a deployable
GSE checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_composer_observability import (
    binary_transfer_metrics,
    canonical_explicit_feature_matrix,
    causal_row_history,
    explicit_composer_features,
    train_free_event_scores,
)
from mtare_topo.governance import write_json


EVENT_NAMES = (
    "corridor",
    "junction",
    "terminal",
    "turn",
    "geometry_transition",
)
EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}
EXPECTED_FRAME_COUNTS = {
    "C07": (17113, 3189, 900, 279, 67),
    "C08": (19234, 3676, 1074, 237, 173),
}
EXPECTED_ROWS = {"C07": 21548, "C08": 24394}
EXPECTED_WORLDS = {"C07": 10, "C08": 10}
EXPLICIT_KEYS = (
    "token_count_probability",
    "token_bearing_deg",
    "token_opening_width_m",
    "token_vertical_profile_m",
    "token_geometry_uncertainty",
    "transport_row_probability",
    "transport_reveal_probability",
)


def _read_teacher(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    counts = {split: np.zeros(len(EVENT_NAMES), dtype=np.int64) for split in EXPECTED_ROWS}
    parents = {split: set() for split in EXPECTED_ROWS}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            parent = str(record.get("parent_id", ""))
            split = parent.rsplit("_", 1)[-1]
            if split not in EXPECTED_ROWS:
                continue
            required = {
                "global_sequence_index",
                "parent_id",
                "traversal_id",
                "sequence_index",
                "event",
                "identity",
            }
            missing = required - record.keys()
            if missing:
                raise RuntimeError(
                    f"corrected Teacher row {line_number} misses {sorted(missing)}"
                )
            event = str(record["event"])
            if event not in EVENT_INDEX:
                raise RuntimeError(f"unknown corrected event {event}")
            global_index = int(record["global_sequence_index"])
            if global_index in rows:
                raise RuntimeError("duplicate corrected Teacher global identity")
            rows[global_index] = record
            counts[split][EVENT_INDEX[event]] += 1
            parents[split].add(parent)
    for split in EXPECTED_ROWS:
        if int(counts[split].sum()) != EXPECTED_ROWS[split]:
            raise RuntimeError(f"{split} corrected Teacher row count drift")
        if tuple(counts[split]) != EXPECTED_FRAME_COUNTS[split]:
            raise RuntimeError(f"{split} corrected Teacher event count drift")
        if len(parents[split]) != EXPECTED_WORLDS[split]:
            raise RuntimeError(f"{split} corrected Teacher world count drift")
    return rows


def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def _load_seed(
    seed_dir: Path,
    teacher: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    prediction_dir = seed_dir / "development_predictions"
    files = sorted(prediction_dir.glob("*.npz"))
    allowed = [path for path in files if path.stem.endswith(("_C07", "_C08"))]
    forbidden = [path for path in files if path not in allowed]
    if forbidden or len(allowed) != 20:
        raise RuntimeError(f"development archive scope drift in {seed_dir}")

    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "global_sequence_index",
            "feature_matrix",
            "event_probability",
            "train_free_junction",
            "train_free_terminal",
            "train_free_turn",
            "train_free_geometry_transition",
            "event_index",
            "identity",
            "parent",
            "family",
            "split",
        )
    }
    world_counts = {"C07": 0, "C08": 0}
    for path in allowed:
        parent = path.stem
        split = parent.rsplit("_", 1)[-1]
        if split not in EXPECTED_ROWS:
            raise RuntimeError("forbidden prediction split")
        world_counts[split] += 1
        with np.load(path, allow_pickle=False) as archive:
            missing = {*EXPLICIT_KEYS, "event_logits", "geometry", "global_sequence_index"} - set(
                archive.files
            )
            if missing:
                raise RuntimeError(f"{path.name} misses {sorted(missing)}")
            global_index = np.asarray(archive["global_sequence_index"], dtype=np.int64)
            if len(np.unique(global_index)) != len(global_index):
                raise RuntimeError(f"duplicate prediction identity in {path.name}")
            try:
                records = [teacher[int(value)] for value in global_index]
            except KeyError as exc:
                raise RuntimeError("prediction identity absent from corrected Teacher") from exc
            if any(str(record["parent_id"]) != parent for record in records):
                raise RuntimeError("prediction/Teacher parent alignment drift")
            traversal = [str(record["traversal_id"]) for record in records]
            sequence = [int(record["sequence_index"]) for record in records]
            geometry = np.asarray(archive["geometry"], dtype=np.float64)
            geometry_sequence, geometry_valid = causal_row_history(
                traversal, sequence, geometry
            )
            prediction = {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in EXPLICIT_KEYS
            }
            features = explicit_composer_features(
                prediction,
                geometry_sequence=geometry_sequence,
                geometry_valid_mask=geometry_valid,
            )
            train_free = train_free_event_scores(features)
            matrix = canonical_explicit_feature_matrix(
                prediction,
                geometry_sequence=geometry_sequence,
                geometry_valid_mask=geometry_valid,
            ).astype(np.float32)
            probability = _softmax(np.asarray(archive["event_logits"], dtype=np.float64))

        chunks["global_sequence_index"].append(global_index)
        chunks["feature_matrix"].append(matrix)
        chunks["event_probability"].append(probability.astype(np.float32))
        for event in EVENT_NAMES[1:]:
            chunks[f"train_free_{event}"].append(
                np.asarray(train_free[event], dtype=np.float32)
            )
        chunks["event_index"].append(
            np.asarray([EVENT_INDEX[str(record["event"])] for record in records], dtype=np.int8)
        )
        chunks["identity"].append(
            np.asarray(
                [None if record.get("identity") is None else str(record["identity"]) for record in records],
                dtype=object,
            )
        )
        chunks["parent"].append(np.full(len(records), parent, dtype=object))
        chunks["family"].append(np.full(len(records), parent[:3], dtype=object))
        chunks["split"].append(np.full(len(records), split, dtype=object))

    if world_counts != EXPECTED_WORLDS:
        raise RuntimeError("prediction world population drift")
    result = {name: np.concatenate(parts) for name, parts in chunks.items()}
    order = np.argsort(result["global_sequence_index"], kind="stable")
    result = {name: value[order] for name, value in result.items()}
    if len(result["global_sequence_index"]) != sum(EXPECTED_ROWS.values()):
        raise RuntimeError("prediction development row population drift")
    if len(np.unique(result["global_sequence_index"])) != len(result["global_sequence_index"]):
        raise RuntimeError("cross-world prediction identity collision")
    return result


def _probe_score(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    transfer_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="liblinear",
            tol=1e-6,
        ),
    )
    model.fit(fit_x, fit_y)
    fit_score = model.predict_proba(fit_x)[:, 1]
    transfer_score = model.predict_proba(transfer_x)[:, 1]
    iterations = int(np.max(model.named_steps["logisticregression"].n_iter_))
    return fit_score, transfer_score, iterations


def _metric_rows(
    sources: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], int]:
    c07 = reference["split"] == "C07"
    c08 = reference["split"] == "C08"
    rows: list[dict[str, Any]] = []
    probe_fits = 0
    score_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    for source, values in sources.items():
        for event_index, event in enumerate(EVENT_NAMES[1:], 1):
            truth = reference["event_index"] == event_index
            for mode in ("train_free", "independent_head", "explicit_probe"):
                if mode == "train_free":
                    fit_score = values[f"train_free_{event}"][c07]
                    transfer_score = values[f"train_free_{event}"][c08]
                    iterations = 0
                elif mode == "independent_head":
                    fit_score = values["event_probability"][c07, event_index]
                    transfer_score = values["event_probability"][c08, event_index]
                    iterations = 0
                else:
                    fit_score, transfer_score, iterations = _probe_score(
                        values["feature_matrix"][c07],
                        truth[c07],
                        values["feature_matrix"][c08],
                    )
                    probe_fits += 1
                score_cache[(source, mode, event)] = (
                    np.asarray(fit_score, dtype=np.float64),
                    np.asarray(transfer_score, dtype=np.float64),
                    iterations,
                )
                metric = binary_transfer_metrics(
                    truth[c07],
                    fit_score,
                    truth[c08],
                    transfer_score,
                    precision_floor=0.98,
                    transfer_identity=reference["identity"][c08],
                ).to_dict()
                prevalence = float(np.mean(truth[c08]))
                rows.append(
                    {
                        "source": source,
                        "mode": mode,
                        "event": event,
                        "probe_iterations": iterations,
                        "transfer_prevalence": prevalence,
                        "transfer_ap_over_prevalence": (
                            float(metric["transfer_average_precision"]) / prevalence
                        ),
                        **metric,
                    }
                )
    for event_index, event in enumerate(EVENT_NAMES[1:], 1):
        truth = reference["event_index"] == event_index
        for mode in ("train_free", "independent_head", "explicit_probe"):
            fit_score = np.mean(
                np.stack(
                    [score_cache[(source, mode, event)][0] for source in sources]
                ),
                axis=0,
            )
            transfer_score = np.mean(
                np.stack(
                    [score_cache[(source, mode, event)][1] for source in sources]
                ),
                axis=0,
            )
            metric = binary_transfer_metrics(
                truth[c07],
                fit_score,
                truth[c08],
                transfer_score,
                precision_floor=0.98,
                transfer_identity=reference["identity"][c08],
            ).to_dict()
            prevalence = float(np.mean(truth[c08]))
            rows.append(
                {
                    "source": "ensemble",
                    "mode": mode,
                    "event": event,
                    "probe_iterations": max(
                        score_cache[(source, mode, event)][2] for source in sources
                    ),
                    "transfer_prevalence": prevalence,
                    "transfer_ap_over_prevalence": (
                        float(metric["transfer_average_precision"]) / prevalence
                    ),
                    **metric,
                }
            )
    return rows, probe_fits


def _ensemble(seeds: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {
        "global_sequence_index": seeds[0]["global_sequence_index"],
        "event_index": seeds[0]["event_index"],
        "identity": seeds[0]["identity"],
        "parent": seeds[0]["parent"],
        "family": seeds[0]["family"],
        "split": seeds[0]["split"],
    }
    for seed in seeds[1:]:
        for key in result:
            if not np.array_equal(seed[key], result[key]):
                raise RuntimeError(f"cross-seed {key} alignment drift")
    averaged = (
        "event_probability",
        "train_free_junction",
        "train_free_terminal",
        "train_free_turn",
        "train_free_geometry_transition",
    )
    for key in averaged:
        result[key] = np.mean(np.stack([seed[key] for seed in seeds]), axis=0)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    ensemble = [
        row
        for row in rows
        if row["source"] == "ensemble" and row["mode"] in {"independent_head", "explicit_probe"}
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    events = list(EVENT_NAMES[1:])
    x = np.arange(len(events))
    width = 0.35
    for offset, mode in enumerate(("independent_head", "explicit_probe")):
        subset = {row["event"]: row for row in ensemble if row["mode"] == mode}
        axes[0].bar(
            x + (offset - 0.5) * width,
            [subset[event]["transfer_roc_auc"] for event in events],
            width,
            label=mode.replace("_", " "),
        )
        axes[1].bar(
            x + (offset - 0.5) * width,
            [subset[event]["transfer_average_precision"] for event in events],
            width,
            label=mode.replace("_", " "),
        )
    for axis, title, ylabel in (
        (axes[0], "Corrected-Teacher C07→C08 observability", "ROC-AUC"),
        (axes[1], "Rare-event ranking on C08", "average precision"),
    ):
        axis.set_xticks(x, ["junction", "terminal", "turn", "transition"], rotation=20)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
    axes[0].axhline(0.65, color="#b53a3a", linestyle="--", linewidth=1.0)
    figure.suptitle("Does the explicit GSE state contain the semantics needed to build graph nodes?")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--seed-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.seed_dir) != 3:
        raise RuntimeError("exactly three sealed V2R5 seed directories are required")
    if len({path.resolve() for path in args.seed_dir}) != 3:
        raise RuntimeError("the three V2R5 seed directories must be distinct")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = _read_teacher(args.teacher.resolve())
    seeds = [_load_seed(path.resolve(), teacher) for path in args.seed_dir]
    ensemble = _ensemble(seeds)
    sources = {f"seed{index}": seed for index, seed in enumerate(seeds)}
    rows, probe_fits = _metric_rows(sources, ensemble)
    _write_csv(output / "event_observability.csv", rows)
    _write_figure(output / "gse_composer_observability", rows)

    def row(source: str, mode: str, event: str) -> dict[str, Any]:
        return next(
            item
            for item in rows
            if item["source"] == source and item["mode"] == mode and item["event"] == event
        )

    explicit = {event: row("ensemble", "explicit_probe", event) for event in EVENT_NAMES[1:]}
    requirements = {
        "junction_c08_auc_at_least_0p90": explicit["junction"]["transfer_roc_auc"] >= 0.90,
        "terminal_c08_auc_at_least_0p90": explicit["terminal"]["transfer_roc_auc"] >= 0.90,
        "turn_c08_auc_at_least_0p70": explicit["turn"]["transfer_roc_auc"] >= 0.70,
        "transition_c08_auc_at_least_0p65": explicit["geometry_transition"]["transfer_roc_auc"] >= 0.65,
        "turn_c08_ap_at_least_twice_prevalence": explicit["turn"]["transfer_ap_over_prevalence"] >= 2.0,
        "transition_c08_ap_at_least_twice_prevalence": explicit["geometry_transition"]["transfer_ap_over_prevalence"] >= 2.0,
        "at_least_two_seed_transition_auc_at_least_0p60": sum(
            row(f"seed{seed}", "explicit_probe", "geometry_transition")["transfer_roc_auc"] >= 0.60
            for seed in range(3)
        )
        >= 2,
    }
    scientific_pass = all(requirements.values())
    summary = {
        "schema_version": "gse_composer_observability_v1",
        "overall_status": (
            "PASS_GSE_COMPOSER_OBSERVABILITY_V1"
            if scientific_pass
            else "FAIL_GSE_COMPOSER_OBSERVABILITY_V1"
        ),
        "scientific_pass": scientific_pass,
        "requirements": requirements,
        "rows": int(len(ensemble["event_index"])),
        "split_rows": EXPECTED_ROWS,
        "worlds": EXPECTED_WORLDS,
        "event_counts": {key: dict(zip(EVENT_NAMES, value, strict=True)) for key, value in EXPECTED_FRAME_COUNTS.items()},
        "diagnostic_linear_probe_fits": probe_fits,
        "main_model_optimizer_steps": 0,
        "checkpoint_updates": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "explicit_ensemble": explicit,
        "independent_head_ensemble": {
            event: row("ensemble", "independent_head", event) for event in EVENT_NAMES[1:]
        },
        "fit_transfer_gap_policy": (
            "Reported as a diagnostic only: fit AUC is in-sample because the fixed probe is fitted on C07, so its gap to C08 is not a valid pass gate."
        ),
        "interpretation": (
            "PASS permits dual-Composer readiness; FAIL requires RouteGeometryProfile visibility/Teacher proof before any main-method training. "
            "Neither outcome authorizes graph replay."
        ),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
