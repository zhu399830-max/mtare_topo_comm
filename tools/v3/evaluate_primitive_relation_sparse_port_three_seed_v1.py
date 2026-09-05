#!/usr/bin/env python3
"""Evaluate sparse-port checkpoints on C07, then one-shot C08 if qualified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import evaluate_primitive_relation_three_seed_v1 as base
from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts,
    _attachment_upper_mask,
    add_sweeps,
    align_for_evaluation,
    binary_counts,
    finalize_geometry_totals,
    geometry_batch_totals,
    merge_geometry_totals,
    relation_counts,
    select_threshold,
    temporal_batch_metrics,
    threshold_sweep_counts,
)
from mtare_topo.representation.primitive_relation_model import MAXIMUM_SLOTS
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_sparse_port_training import (
    EVALUATION_BATCH_SIZE,
    numpy_batch_to_torch,
)


PASS = "PASS_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_V1"
FAIL_C07 = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_C07"
FAIL_C08 = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_C08"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _load_model(
    path: Path,
    seed: int,
    device: torch.device,
) -> tuple[SparsePortRelationNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version")
        != "primitive_relation_sparse_port_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
    ):
        raise RuntimeError("selected sparse-port checkpoint identity drift")
    model = SparsePortRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _risk_adjusted_attachment(
    prediction,
    aligned: dict[str, torch.Tensor],
    *,
    existence_threshold: float,
) -> tuple[BinaryCounts, ...]:
    active = torch.sigmoid(prediction.existence_logits) >= float(existence_threshold)
    endpoint_active = active.repeat_interleave(2, dim=1)
    upper = _attachment_upper_mask(active.device)
    eligible = (
        endpoint_active[:, :, None]
        & endpoint_active[:, None, :]
        & upper[None]
    )
    target = aligned["attachment"].reshape(
        len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    ).bool() & upper[None]
    probability = torch.sigmoid(prediction.endpoint_attachment_logits).reshape(
        len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    )
    uncertainty = prediction.endpoint_attachment_uncertainty.reshape_as(probability)
    safe_score = probability * (1.0 - uncertainty)
    return threshold_sweep_counts(safe_score, target, eligible=eligible)


@torch.no_grad()
def _learned_c07(
    model: SparsePortRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
) -> dict:
    # The first two passes are byte-for-byte the same primary metrics as V1,
    # preserving fair F1/geometry comparison.  The third pass evaluates the
    # pre-registered V2 risk-adjusted safe relation score.
    metrics = base._learned_c07(model, loader, device=device)
    existence_threshold = float(metrics["existence_f1_selection"]["threshold"])
    safe = None
    rows = 0
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
        batch = numpy_batch_to_torch(numpy_batch, device=device)
        prediction = model(
            batch.range_valid,
            batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        aligned = align_for_evaluation(prediction, batch.targets)
        safe = add_sweeps(
            safe,
            _risk_adjusted_attachment(
                prediction, aligned, existence_threshold=existence_threshold,
            ),
        )
        rows += len(numpy_batch.range_valid)
    if rows != base.EXPECTED_ROWS["c07"] or safe is None:
        raise RuntimeError("sparse-port C07 safe-score population drift")
    metrics["attachment_safe_selection_raw_probability"] = metrics[
        "attachment_safe_selection"
    ]
    metrics["attachment_safe_selection"] = select_threshold(
        safe, minimum_precision=base.MINIMUM_SAFE_PRECISION,
    )
    metrics["attachment_safe_score"] = "p_attachment_times_one_minus_uncertainty"
    return metrics


def _comparison(method: dict, baseline: dict) -> dict:
    result = base._comparison(method, baseline)
    safe = method["attachment_safe_selection"]
    result["checks"]["safe_attachment_precision_ge_98_nonzero"] = bool(
        safe["available"]
        and float(safe["precision"]) >= base.MINIMUM_SAFE_PRECISION
        and int(safe["true_positive"]) > 0
    )
    result["pass"] = all(result["checks"].values())
    return result


def _compact_prediction(
    prediction,
    numpy_batch,
    *,
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    result = base._compact_prediction(
        prediction, numpy_batch, thresholds=thresholds,
    )
    existence = torch.sigmoid(prediction.existence_logits)
    active = existence >= thresholds["existence"]
    risk_score = (
        torch.sigmoid(prediction.endpoint_attachment_logits)
        * (1.0 - prediction.endpoint_attachment_uncertainty)
    )
    attachment_safe = risk_score >= thresholds["attachment_safe"]
    attachment_safe &= (
        active[:, :, None, None, None]
        & active[:, None, None, :, None]
    )
    result["attachment_safe_packed"] = np.packbits(
        attachment_safe.cpu().numpy().reshape(len(active), -1),
        axis=1,
        bitorder="little",
    )
    result["attachment_uncertainty"] = (
        prediction.endpoint_attachment_uncertainty.cpu().numpy().astype(np.float16)
    )
    result["overlap_uncertainty"] = (
        prediction.disconnected_overlap_uncertainty.cpu().numpy().astype(np.float16)
    )
    return result


@torch.no_grad()
def _learned_transfer(
    model: SparsePortRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
    calibration: dict,
    output_root: Path,
) -> dict:
    thresholds = {
        "existence": float(calibration["existence_f1_selection"]["threshold"]),
        "attachment": float(calibration["attachment_f1_selection"]["threshold"]),
        "overlap": float(calibration["overlap_f1_selection"]["threshold"]),
        "attachment_safe": float(calibration["attachment_safe_selection"]["threshold"]),
    }
    primitive = BinaryCounts(); attachment = BinaryCounts(); overlap = BinaryCounts()
    safe_attachment = BinaryCounts(); temporal_presence = BinaryCounts()
    temporal_correct = temporal_total = rows = 0
    geometry = None
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "thresholds.json", thresholds)
    upper = _attachment_upper_mask(device)
    for name in loader.task_names:
        parts: dict[str, list[np.ndarray]] = {}
        length = loader._lengths[name]
        for start in range(0, length, EVALUATION_BATCH_SIZE):
            indices = np.arange(
                start, min(start + EVALUATION_BATCH_SIZE, length), dtype=np.int64,
            )
            numpy_batch = loader._read(name, indices)
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            active = torch.sigmoid(prediction.existence_logits) >= thresholds["existence"]
            primitive += BinaryCounts(
                int((active & aligned["mask"]).sum()),
                int((active & ~aligned["mask"]).sum()),
                int((~active & aligned["mask"]).sum()),
            )
            relations = relation_counts(
                prediction,
                aligned,
                existence_threshold=thresholds["existence"],
                attachment_threshold=thresholds["attachment"],
                overlap_threshold=thresholds["overlap"],
            )
            attachment += relations["attachment"]
            overlap += relations["disconnected_overlap"]
            endpoint_active = active.repeat_interleave(2, dim=1)
            eligible = (
                endpoint_active[:, :, None]
                & endpoint_active[:, None, :]
                & upper[None]
            )
            target = aligned["attachment"].reshape(
                len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
            ).bool() & upper[None]
            safe_score = (
                torch.sigmoid(prediction.endpoint_attachment_logits)
                * (1.0 - prediction.endpoint_attachment_uncertainty)
            ).reshape(len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS)
            safe_attachment += binary_counts(
                (safe_score >= thresholds["attachment_safe"]) & eligible,
                target,
            )
            geometry = merge_geometry_totals(
                geometry,
                geometry_batch_totals(
                    prediction, aligned,
                    existence_threshold=thresholds["existence"],
                ),
            )
            temporal = temporal_batch_metrics(
                prediction, aligned,
                existence_threshold=thresholds["existence"],
            )
            temporal_presence += temporal["presence"]
            temporal_correct += int(temporal["correspondence_correct"])
            temporal_total += int(temporal["correspondence_total"])
            for key, value in _compact_prediction(
                prediction, numpy_batch, thresholds=thresholds,
            ).items():
                parts.setdefault(key, []).append(value)
            rows += len(indices)
        np.savez_compressed(
            output_root / f"{name.removesuffix('.zarr')}.npz",
            **{
                key: np.concatenate(values, axis=0)
                for key, values in parts.items()
            },
        )
    if rows != base.EXPECTED_ROWS["c08"] or geometry is None:
        raise RuntimeError("sparse-port C08 transfer population drift")
    return {
        "rows": rows,
        "thresholds": thresholds,
        "primitive_detection": primitive.to_dict(),
        "attachment": attachment.to_dict(),
        "attachment_safe": safe_attachment.to_dict(),
        "attachment_safe_score": "p_attachment_times_one_minus_uncertainty",
        "disconnected_overlap": overlap.to_dict(),
        "geometry": finalize_geometry_totals(geometry),
        "temporal_presence": temporal_presence.to_dict(),
        "temporal_correspondence_accuracy": temporal_correct / temporal_total,
        "temporal_correspondence_correct": temporal_correct,
        "temporal_correspondence_total": temporal_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--baseline-c07-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("sparse-port evaluation output exists; overwrite forbidden")
    output.mkdir(parents=True)
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("sparse-port evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    c07_loader = PrimitiveRelationBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07",
    )
    baseline_c07 = json.loads(args.baseline_c07_summary.read_text())
    if (
        baseline_c07.get("rows") != base.EXPECTED_ROWS["c07"]
        or not baseline_c07.get("scientific_pass")
    ):
        raise RuntimeError("frozen non-learning C07 baseline drift")
    c07_seed, models = [], []
    for seed in range(3):
        model, checkpoint = _load_model(
            args.models_root / f"seed{seed}/selected.pt", seed, device,
        )
        metrics = _learned_c07(model, c07_loader, device=device)
        comparison = _comparison(metrics, baseline_c07)
        record = {
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics, "comparison": comparison,
        }
        c07_seed.append(record); models.append(model)
        _write_json(output / f"c07_seed{seed}.json", record)
        print(json.dumps({"stage": "c07", "seed": seed, "comparison": comparison}), flush=True)
    c07_pass_count = sum(bool(value["comparison"]["pass"]) for value in c07_seed)
    c07_scientific_pass = c07_pass_count >= 2
    preliminary = {
        "schema_version": "primitive_relation_sparse_port_three_seed_evaluation_v1",
        "c07": {
            "baseline": baseline_c07, "seeds": c07_seed,
            "passing_seeds": c07_pass_count,
            "scientific_pass": c07_scientific_pass,
        },
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
    }
    _write_json(output / "c07_decision.json", preliminary)
    if not c07_scientific_pass:
        preliminary.update({
            "overall_status": FAIL_C07, "scientific_pass": False,
            "decision": "STOP_SPARSE_PORT_BEFORE_C08_AND_GRAPH",
            "duration_seconds": time.monotonic() - started,
        })
        _write_json(output / "summary.json", preliminary)
        base._plot_result(preliminary, output / "primitive_relation_sparse_port_comparison")
        return 2

    # C08 is not instantiated until the complete C07 geometry/relation/safe
    # gate has passed for at least two independent seeds.
    c08_loader = PrimitiveRelationBatchLoader(
        args.sensor_root / "c08", args.teacher_root / "c08",
    )
    baseline_c08 = base._nonlearning_partition(c08_loader, split="c08")
    _write_json(output / "baseline_c08.json", baseline_c08)
    c08_seed = []
    predictions_root = output / "c08_predictions"
    predictions_root.mkdir()
    for seed, model in enumerate(models):
        metrics = _learned_transfer(
            model, c08_loader, device=device,
            calibration=c07_seed[seed]["metrics"],
            output_root=predictions_root / f"seed{seed}",
        )
        comparison = base._comparison(metrics, baseline_c08)
        safe = metrics["attachment_safe"]
        comparison["checks"]["safe_attachment_precision_ge_98_nonzero"] = bool(
            float(safe["precision"]) >= base.MINIMUM_SAFE_PRECISION
            and int(safe["true_positive"]) > 0
        )
        comparison["pass"] = all(comparison["checks"].values())
        record = {"seed": seed, "metrics": metrics, "comparison": comparison}
        c08_seed.append(record)
        _write_json(output / f"c08_seed{seed}.json", record)
        print(json.dumps({"stage": "c08", "seed": seed, "comparison": comparison}), flush=True)
    c08_pass_count = sum(bool(value["comparison"]["pass"]) for value in c08_seed)
    scientific_pass = c08_pass_count >= 2
    summary = {
        "schema_version": "primitive_relation_sparse_port_three_seed_evaluation_v1",
        "overall_status": PASS if scientific_pass else FAIL_C08,
        "scientific_pass": scientific_pass,
        "c07": {
            "baseline": baseline_c07, "seeds": c07_seed,
            "passing_seeds": c07_pass_count, "scientific_pass": True,
        },
        "c08": {
            "baseline": baseline_c08, "seeds": c08_seed,
            "passing_seeds": c08_pass_count, "scientific_pass": scientific_pass,
        },
        "c08_rows_read": base.EXPECTED_ROWS["c08"] * 4,
        "c09_c10_worlds_read": 0, "graph_replays": 0,
        "decision": (
            "ALLOW_P3_OFFLINE_STRUCTURAL_GRAPH"
            if scientific_pass else "STOP_SPARSE_PORT_BEFORE_GRAPH"
        ),
        "duration_seconds": time.monotonic() - started,
    }
    _write_json(output / "summary.json", summary)
    base._plot_result(summary, output / "primitive_relation_sparse_port_comparison")
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
