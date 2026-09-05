#!/usr/bin/env python3
"""Evaluate observable-relation checkpoints on C07 and stop before C08."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import evaluate_primitive_relation_three_seed_v1 as base
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts, _attachment_observability_validity, _attachment_upper_mask,
    add_sweeps, align_for_evaluation, existence_sweep,
    finalize_geometry_totals, geometry_batch_totals, merge_geometry_totals,
    relation_sweeps, select_threshold, temporal_batch_metrics,
    threshold_sweep_counts,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet, safe_attachment_score,
)
from mtare_topo.representation.primitive_relation_observable_schedule import (
    EVALUATION_BATCH_SIZE,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


PASS = "PASS_PRIMITIVE_RELATION_OBSERVABLE_THREE_SEED_C07_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_OBSERVABLE_THREE_SEED_C07_V1"
EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_OBSERVABLE_POSITIVES = 442_936


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _load_model(
    path: Path, seed: int, device: torch.device,
) -> tuple[ObservableSparsePortRelationNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version")
        != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or not checkpoint.get("training_contract", {}).get("frozen_geometry_temporal")
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity")
        != "dual_endpoint_observed"
    ):
        raise RuntimeError("selected observable checkpoint identity drift")
    model = ObservableSparsePortRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _learned_c07(
    model: ObservableSparsePortRelationNet,
    loader: ObservablePrimitiveRelationBatchLoader,
    *, device: torch.device,
) -> dict:
    existence = evidence = None
    rows = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid, batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        aligned = align_for_evaluation(prediction, batch.targets)
        existence = add_sweeps(existence, existence_sweep(prediction, aligned))
        evidence = add_sweeps(evidence, threshold_sweep_counts(
            torch.sigmoid(prediction.endpoint_evidence_logits),
            aligned["endpoint_observed"],
        ))
        rows += len(numpy_batch.base.range_valid)
    if rows != EXPECTED_ROWS or existence is None or evidence is None:
        raise RuntimeError("observable C07 existence/evidence population drift")
    existence_selection = select_threshold(existence)
    existence_threshold = float(existence_selection["threshold"])

    attachment = overlap = safe = None
    geometry = None
    temporal_presence = BinaryCounts()
    temporal_correct = temporal_total = rows = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid, batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        aligned = align_for_evaluation(prediction, batch.targets)
        relations = relation_sweeps(
            prediction, aligned, existence_threshold=existence_threshold,
        )
        attachment = add_sweeps(attachment, relations["attachment"])
        overlap = add_sweeps(overlap, relations["disconnected_overlap"])
        active = torch.sigmoid(prediction.existence_logits) >= existence_threshold
        endpoint_active = active.repeat_interleave(2, dim=1)
        validity = _attachment_observability_validity(aligned)
        upper = _attachment_upper_mask(device)[None]
        eligible = (
            endpoint_active[:, :, None] & endpoint_active[:, None, :]
            & validity & upper
        )
        target = aligned["attachment"].reshape(
            len(active), 64, 64,
        ).bool() & validity & upper
        safe = add_sweeps(safe, threshold_sweep_counts(
            safe_attachment_score(prediction).reshape(len(active), 64, 64),
            target, eligible=eligible,
        ))
        geometry = merge_geometry_totals(
            geometry, geometry_batch_totals(
                prediction, aligned, existence_threshold=existence_threshold,
            ),
        )
        temporal = temporal_batch_metrics(
            prediction, aligned, existence_threshold=existence_threshold,
        )
        temporal_presence += temporal["presence"]
        temporal_correct += int(temporal["correspondence_correct"])
        temporal_total += int(temporal["correspondence_total"])
        rows += len(numpy_batch.base.range_valid)
    if (
        rows != EXPECTED_ROWS or attachment is None or overlap is None
        or safe is None or geometry is None
    ):
        raise RuntimeError("observable C07 metric population drift")
    if safe[0].true_positive + safe[0].false_negative != EXPECTED_OBSERVABLE_POSITIVES:
        raise RuntimeError("observable C07 positive attachment population drift")
    attachment_selection = select_threshold(attachment)
    overlap_selection = select_threshold(overlap)
    safe_selection = select_threshold(
        safe, minimum_precision=base.MINIMUM_SAFE_PRECISION,
    )
    return {
        "rows": rows,
        "primitive_detection": existence_selection,
        "existence_f1_selection": existence_selection,
        "existence_safe_selection": select_threshold(
            existence, minimum_precision=base.MINIMUM_SAFE_PRECISION,
        ),
        "endpoint_evidence_f1_selection": select_threshold(evidence),
        "endpoint_evidence_safe_selection": select_threshold(
            evidence, minimum_precision=base.MINIMUM_SAFE_PRECISION,
        ),
        "attachment": attachment_selection,
        "attachment_f1_selection": attachment_selection,
        "attachment_safe_selection": safe_selection,
        "attachment_safe_score": "p_attachment_times_two_endpoint_evidence_times_one_minus_uncertainty",
        "observable_attachment_target_positives": EXPECTED_OBSERVABLE_POSITIVES,
        "hidden_attachment_targets_excluded": 82_141,
        "disconnected_overlap": overlap_selection,
        "overlap_f1_selection": overlap_selection,
        "overlap_safe_selection": select_threshold(
            overlap, minimum_precision=base.MINIMUM_SAFE_PRECISION,
        ),
        "geometry": finalize_geometry_totals(geometry),
        "temporal_presence": temporal_presence.to_dict(),
        "temporal_correspondence_accuracy": temporal_correct / temporal_total,
        "temporal_correspondence_correct": temporal_correct,
        "temporal_correspondence_total": temporal_total,
    }


def _comparison(method: dict, baseline: dict) -> dict:
    result = base._comparison(method, baseline)
    safe = method["attachment_safe_selection"]
    result["checks"]["safe_attachment_precision_ge_98_nonzero"] = bool(
        safe["available"] and float(safe["precision"]) >= base.MINIMUM_SAFE_PRECISION
        and int(safe["true_positive"]) > 0
    )
    result["pass"] = all(result["checks"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--baseline-c07-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("observable evaluation output exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("observable evaluation requires CUDA")
    device = torch.device("cuda")
    loader = ObservablePrimitiveRelationBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07",
        args.sidecar_root / "c07",
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("observable evaluator C07 loader drift")
    baseline = json.loads(args.baseline_c07_summary.read_text())
    if (
        baseline.get("rows") != EXPECTED_ROWS
        or not baseline.get("scientific_pass")
        or baseline.get("observable_attachment_target_positives")
        != EXPECTED_OBSERVABLE_POSITIVES
    ):
        raise RuntimeError("observable non-learning baseline drift")
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(
            args.models_root / f"seed{seed}/selected.pt", seed, device,
        )
        metrics = _learned_c07(model, loader, device=device)
        comparison = _comparison(metrics, baseline)
        record = {
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics, "comparison": comparison,
        }
        seeds.append(record)
        _write_json(output / f"c07_seed{seed}.json", record)
        print(json.dumps({
            "stage": "c07", "seed": seed, "comparison": comparison,
            "safe": metrics["attachment_safe_selection"],
        }), flush=True)
    passing = sum(bool(value["comparison"]["pass"]) for value in seeds)
    scientific_pass = passing >= 2
    summary = {
        "schema_version": "primitive_relation_observable_three_seed_evaluation_v1",
        "overall_status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "c07": {
            "baseline": baseline, "seeds": seeds,
            "passing_seeds": passing, "scientific_pass": scientific_pass,
        },
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "decision": (
            "ALLOW_C08_OBSERVABILITY_SIDECAR_AND_ZERO_ADAPTATION_TRANSFER"
            if scientific_pass else "STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH"
        ),
        "duration_seconds": time.monotonic() - started,
    }
    _write_json(output / "summary.json", summary)
    base._plot_result(summary, output / "primitive_relation_observable_comparison")
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
