#!/usr/bin/env python3
"""Select C07 thresholds and perform one-shot C08 primitive-relation transfer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts,
    THRESHOLD_GRID,
    add_sweeps,
    align_for_evaluation,
    existence_sweep,
    finalize_geometry_totals,
    geometry_batch_totals,
    merge_geometry_totals,
    nonlearning_batch_to_torch,
    relation_counts,
    relation_sweeps,
    select_threshold,
    temporal_batch_metrics,
)
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationNet
from mtare_topo.representation.primitive_relation_training import (
    EVALUATION_BATCH_SIZE,
    numpy_batch_to_torch,
)
from mtare_topo.semantics.primitive_relation_nonlearning import RobustPrimitiveRelationBaseline


EXPECTED_ROWS = {"c07":64_644, "c08":73_182}
EXPECTED_TASKS = 30
MINIMUM_GEOMETRY_IMPROVEMENT = 0.10
MINIMUM_SURFACE_IMPROVEMENT = 0.10
MINIMUM_ATTACHMENT_F1_GAIN = 0.05
MINIMUM_SAFE_PRECISION = 0.98
GEOMETRY_KEYS = (
    "axis_control_absolute_error_m_mean",
    "width_absolute_error_m_mean",
    "height_absolute_error_m_mean",
    "shape_exponent_absolute_error_mean",
    "slope_absolute_error_deg_mean",
    "curvature_absolute_error_per_m_mean",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add(first: BinaryCounts, second: BinaryCounts) -> BinaryCounts:
    return first + second


def _load_model(path: Path, seed: int, device: torch.device) -> tuple[PrimitiveRelationNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "primitive_relation_checkpoint_v1" or int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError("selected primitive checkpoint identity drift")
    model = PrimitiveRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _learned_c07(
    model: PrimitiveRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
) -> dict:
    existence = None
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
        existence = add_sweeps(existence, existence_sweep(prediction, aligned))
        rows += len(numpy_batch.range_valid)
    if rows != EXPECTED_ROWS["c07"] or existence is None:
        raise RuntimeError("learned C07 existence pass population drift")
    existence_selection = select_threshold(existence)
    safe_existence = select_threshold(existence, minimum_precision=MINIMUM_SAFE_PRECISION)
    threshold = float(existence_selection["threshold"])

    attachment = overlap = None
    geometry = None
    temporal_presence = BinaryCounts(); temporal_correct = temporal_total = 0
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
        relations = relation_sweeps(
            prediction, aligned, existence_threshold=threshold,
        )
        attachment = add_sweeps(attachment, relations["attachment"])
        overlap = add_sweeps(overlap, relations["disconnected_overlap"])
        geometry = merge_geometry_totals(
            geometry,
            geometry_batch_totals(
                prediction, aligned, existence_threshold=threshold,
            ),
        )
        temporal = temporal_batch_metrics(
            prediction, aligned, existence_threshold=threshold,
        )
        temporal_presence = _add(temporal_presence, temporal["presence"])
        temporal_correct += int(temporal["correspondence_correct"])
        temporal_total += int(temporal["correspondence_total"])
        rows += len(numpy_batch.range_valid)
    if rows != EXPECTED_ROWS["c07"] or attachment is None or overlap is None or geometry is None:
        raise RuntimeError("learned C07 metric pass population drift")
    attachment_selection = select_threshold(attachment)
    overlap_selection = select_threshold(overlap)
    return {
        "rows":rows,
        "primitive_detection":existence_selection,
        "existence_f1_selection":existence_selection,
        "existence_safe_selection":safe_existence,
        "attachment":attachment_selection,
        "attachment_f1_selection":attachment_selection,
        "attachment_safe_selection":select_threshold(attachment, minimum_precision=MINIMUM_SAFE_PRECISION),
        "disconnected_overlap":overlap_selection,
        "overlap_f1_selection":overlap_selection,
        "overlap_safe_selection":select_threshold(overlap, minimum_precision=MINIMUM_SAFE_PRECISION),
        "geometry":finalize_geometry_totals(geometry),
        "temporal_presence":temporal_presence.to_dict(),
        "temporal_correspondence_accuracy":temporal_correct / temporal_total,
        "temporal_correspondence_correct":temporal_correct,
        "temporal_correspondence_total":temporal_total,
    }


def _compact_prediction(
    prediction,
    numpy_batch,
    *,
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    existence = torch.sigmoid(prediction.existence_logits)
    active = existence >= thresholds["existence"]
    attachment_probability = torch.sigmoid(prediction.endpoint_attachment_logits)
    overlap_probability = torch.sigmoid(prediction.disconnected_overlap_logits)
    attachment_f1 = attachment_probability >= thresholds["attachment"]
    attachment_f1 &= active[:, :, None, None, None] & active[:, None, None, :, None]
    overlap_f1 = overlap_probability >= thresholds["overlap"]
    overlap_f1 &= active[:, :, None] & active[:, None, :]
    safe_threshold = thresholds.get("attachment_safe", 1.0)
    attachment_safe = attachment_probability >= safe_threshold
    attachment_safe &= active[:, :, None, None, None] & active[:, None, None, :, None]
    temporal_destination = prediction.temporal_correspondence_logits.argmax(-1)
    temporal_presence = torch.sigmoid(prediction.temporal_presence_logits) >= 0.5
    return {
        "source_global_sequence_index":numpy_batch.source_global_sequence_index,
        "variant_global_sequence_index":numpy_batch.variant_global_sequence_index,
        "existence_probability":existence.cpu().numpy().astype(np.float16),
        "active_mask":active.cpu().numpy().astype(np.uint8),
        "axis_control_current_sensor_m":prediction.axis_control_current_sensor_m.cpu().numpy().astype(np.float16),
        "endpoint_half_axes_m":prediction.endpoint_half_axes_m.cpu().numpy().astype(np.float16),
        "endpoint_shape_exponent":prediction.endpoint_shape_exponent.cpu().numpy().astype(np.float16),
        "endpoint_descriptor":prediction.endpoint_descriptor.cpu().numpy().astype(np.float16),
        "geometry_uncertainty":prediction.geometry_uncertainty.cpu().numpy().astype(np.float16),
        "attachment_f1_packed":np.packbits(attachment_f1.cpu().numpy().reshape(len(active), -1), axis=1, bitorder="little"),
        "attachment_safe_packed":np.packbits(attachment_safe.cpu().numpy().reshape(len(active), -1), axis=1, bitorder="little"),
        "overlap_f1_packed":np.packbits(overlap_f1.cpu().numpy().reshape(len(active), -1), axis=1, bitorder="little"),
        "temporal_destination":temporal_destination.cpu().numpy().astype(np.int8),
        "temporal_presence_packed":np.packbits(temporal_presence.cpu().numpy().reshape(len(active), -1), axis=1, bitorder="little"),
    }


@torch.no_grad()
def _learned_transfer(
    model: PrimitiveRelationNet,
    loader: PrimitiveRelationBatchLoader,
    *,
    device: torch.device,
    calibration: dict,
    output_root: Path,
) -> dict:
    thresholds = {
        "existence":float(calibration["existence_f1_selection"]["threshold"]),
        "attachment":float(calibration["attachment_f1_selection"]["threshold"]),
        "overlap":float(calibration["overlap_f1_selection"]["threshold"]),
        "attachment_safe":float(calibration["attachment_safe_selection"]["threshold"])
        if calibration["attachment_safe_selection"]["available"] else 1.0,
    }
    primitive = BinaryCounts(); attachment = BinaryCounts(); overlap = BinaryCounts()
    temporal_presence = BinaryCounts(); temporal_correct = temporal_total = 0
    geometry = None; rows = 0
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "thresholds.json", thresholds)
    for name in loader.task_names:
        parts: dict[str, list[np.ndarray]] = {}
        length = loader._lengths[name]
        for start in range(0, length, EVALUATION_BATCH_SIZE):
            indices = np.arange(start, min(start + EVALUATION_BATCH_SIZE, length), dtype=np.int64)
            numpy_batch = loader._read(name, indices)
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            active = torch.sigmoid(prediction.existence_logits) >= thresholds["existence"]
            primitive = _add(primitive, BinaryCounts(
                int((active & aligned["mask"]).sum()),
                int((active & ~aligned["mask"]).sum()),
                int((~active & aligned["mask"]).sum()),
            ))
            relations = relation_counts(
                prediction, aligned, existence_threshold=thresholds["existence"],
                attachment_threshold=thresholds["attachment"],
                overlap_threshold=thresholds["overlap"],
            )
            attachment = _add(attachment, relations["attachment"])
            overlap = _add(overlap, relations["disconnected_overlap"])
            geometry = merge_geometry_totals(
                geometry,
                geometry_batch_totals(
                    prediction, aligned, existence_threshold=thresholds["existence"],
                ),
            )
            temporal = temporal_batch_metrics(
                prediction, aligned, existence_threshold=thresholds["existence"],
            )
            temporal_presence = _add(temporal_presence, temporal["presence"])
            temporal_correct += int(temporal["correspondence_correct"])
            temporal_total += int(temporal["correspondence_total"])
            for key, value in _compact_prediction(
                prediction, numpy_batch, thresholds=thresholds,
            ).items():
                parts.setdefault(key, []).append(value)
            rows += len(indices)
        np.savez_compressed(
            output_root / f"{name.removesuffix('.zarr')}.npz",
            **{key:np.concatenate(values, axis=0) for key,values in parts.items()},
        )
    if rows != EXPECTED_ROWS["c08"] or geometry is None:
        raise RuntimeError("learned C08 transfer population drift")
    return {
        "rows":rows,"thresholds":thresholds,
        "primitive_detection":primitive.to_dict(),
        "attachment":attachment.to_dict(),
        "disconnected_overlap":overlap.to_dict(),
        "geometry":finalize_geometry_totals(geometry),
        "temporal_presence":temporal_presence.to_dict(),
        "temporal_correspondence_accuracy":temporal_correct/temporal_total,
        "temporal_correspondence_correct":temporal_correct,
        "temporal_correspondence_total":temporal_total,
    }


def _nonlearning_partition(
    loader: PrimitiveRelationBatchLoader,
    *,
    split: str,
) -> dict:
    method = RobustPrimitiveRelationBaseline()
    primitive = BinaryCounts(); attachment = BinaryCounts(); overlap = BinaryCounts()
    temporal_presence = BinaryCounts(); temporal_correct = temporal_total = 0
    geometry = None; rows = 0
    threshold_index = int(np.argmin(np.abs(THRESHOLD_GRID - 0.5)))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name in loader.task_names:
            length = loader._lengths[name]
            for start in range(0, length, 32):
                indices = np.arange(start, min(start + 32, length), dtype=np.int64)
                numpy_batch = loader._read(name, indices)

                def predict(index: int):
                    return method.predict(
                        numpy_batch.range_valid[index],
                        numpy_batch.relative_translation_current_sensor_m[index],
                        numpy_batch.relative_yaw_current_sensor_deg[index],
                    )

                hard = list(executor.map(predict, range(len(indices))))
                prediction = nonlearning_batch_to_torch(hard, device=torch.device("cpu"))
                batch = numpy_batch_to_torch(numpy_batch, device=torch.device("cpu"))
                aligned = align_for_evaluation(prediction, batch.targets)
                primitive = _add(primitive, existence_sweep(prediction, aligned)[threshold_index])
                relations = relation_counts(
                    prediction, aligned, existence_threshold=0.5,
                    attachment_threshold=0.5, overlap_threshold=0.5,
                )
                attachment = _add(attachment, relations["attachment"])
                overlap = _add(overlap, relations["disconnected_overlap"])
                geometry = merge_geometry_totals(
                    geometry,
                    geometry_batch_totals(prediction, aligned, existence_threshold=0.5),
                )
                temporal = temporal_batch_metrics(prediction, aligned, existence_threshold=0.5)
                temporal_presence = _add(temporal_presence, temporal["presence"])
                temporal_correct += int(temporal["correspondence_correct"])
                temporal_total += int(temporal["correspondence_total"])
                rows += len(indices)
    if rows != EXPECTED_ROWS[split] or geometry is None:
        raise RuntimeError(f"non-learning {split} population drift")
    return {
        "rows":rows,"primitive_detection":primitive.to_dict(),
        "attachment":attachment.to_dict(),"disconnected_overlap":overlap.to_dict(),
        "geometry":finalize_geometry_totals(geometry),
        "temporal_presence":temporal_presence.to_dict(),
        "temporal_correspondence_accuracy":temporal_correct/temporal_total,
    }


def _comparison(method: dict, baseline: dict) -> dict:
    geometry_improvements = {
        name:1.0 - float(method["geometry"][name]) / float(baseline["geometry"][name])
        for name in GEOMETRY_KEYS
    }
    result = {
        "surface_improvement":1.0 - float(method["geometry"]["surface_chamfer_m_mean"]) / float(baseline["geometry"]["surface_chamfer_m_mean"]),
        "geometry_improvements":geometry_improvements,
        "geometry_macro_improvement":float(np.mean(tuple(geometry_improvements.values()))),
        "attachment_f1_gain":float(method["attachment"]["f1"])-float(baseline["attachment"]["f1"]),
        "primitive_f1_gain":float(method["primitive_detection"]["f1"])-float(baseline["primitive_detection"]["f1"]),
        "coverage_gain":float(method["geometry"]["target_coverage"])-float(baseline["geometry"]["target_coverage"]),
    }
    result["checks"] = {
        "surface_improvement_ge_10pct":result["surface_improvement"] >= MINIMUM_SURFACE_IMPROVEMENT,
        "geometry_macro_improvement_ge_10pct":result["geometry_macro_improvement"] >= MINIMUM_GEOMETRY_IMPROVEMENT,
        "attachment_f1_gain_ge_5pt":result["attachment_f1_gain"] >= MINIMUM_ATTACHMENT_F1_GAIN,
        "primitive_f1_not_below_baseline":result["primitive_f1_gain"] >= 0.0,
        "target_coverage_not_below_baseline":result["coverage_gain"] >= 0.0,
    }
    result["pass"] = all(result["checks"].values())
    return result


def _plot_result(summary: dict, destination: Path) -> None:
    splits = ["c07"] + (["c08"] if "c08" in summary else [])
    figure, axes = plt.subplots(1, len(splits), figsize=(6.0 * len(splits), 4.4), squeeze=False)
    for column, split in enumerate(splits):
        axis = axes[0, column]
        block = summary[split]
        labels = ["baseline"] + [f"seed {value['seed']}" for value in block["seeds"]]
        surface = [block["baseline"]["geometry"]["surface_chamfer_m_mean"]] + [
            value["metrics"]["geometry"]["surface_chamfer_m_mean"] for value in block["seeds"]
        ]
        attachment = [block["baseline"]["attachment"]["f1"]] + [
            value["metrics"]["attachment"]["f1"]
            for value in block["seeds"]
        ]
        x = np.arange(len(labels)); width = 0.38
        left = axis.bar(x-width/2, surface, width, label="surface Chamfer (m)", color="#2563eb")
        second = axis.twinx()
        right = second.bar(x+width/2, attachment, width, label="attachment F1", color="#ea580c")
        axis.set_xticks(x, labels, rotation=15); axis.set_ylabel("surface Chamfer (m)")
        second.set_ylabel("attachment F1"); second.set_ylim(0.0, 1.0)
        axis.set_title(f"{split.upper()} primitive relation recovery")
        axis.legend([left, right], ["surface Chamfer", "attachment F1"], loc="upper right")
    figure.suptitle("Learned construction-supervised primitives vs same-input robust fitting")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination.with_suffix("." + suffix), dpi=180)
    plt.close(figure)


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
        raise RuntimeError("primitive-relation evaluation output exists; overwrite forbidden")
    output.mkdir(parents=True)
    started = time.monotonic()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("learned primitive evaluation requires CUDA")
    c07_loader = PrimitiveRelationBatchLoader(args.sensor_root/"c07", args.teacher_root/"c07")
    baseline_c07 = json.loads(args.baseline_c07_summary.read_text())
    if baseline_c07.get("rows") != EXPECTED_ROWS["c07"] or not baseline_c07.get("scientific_pass"):
        raise RuntimeError("frozen non-learning C07 baseline drift")
    c07_seed = []
    models = []
    for seed in range(3):
        model, checkpoint = _load_model(args.models_root/f"seed{seed}/selected.pt", seed, device)
        metrics = _learned_c07(model, c07_loader, device=device)
        comparison = _comparison(metrics, baseline_c07)
        record = {"seed":seed,"selected_epoch":int(checkpoint["epoch"]),"metrics":metrics,"comparison":comparison}
        c07_seed.append(record); models.append(model)
        _write_json(output/f"c07_seed{seed}.json", record)
        print(json.dumps({"stage":"c07","seed":seed,"comparison":comparison}), flush=True)
    c07_pass_count = sum(bool(value["comparison"]["pass"]) for value in c07_seed)
    c07_scientific_pass = c07_pass_count >= 2
    preliminary = {
        "schema_version":"primitive_relation_three_seed_evaluation_v1",
        "c07":{"baseline":baseline_c07,"seeds":c07_seed,"passing_seeds":c07_pass_count,"scientific_pass":c07_scientific_pass},
        "c08_rows_read":0,
    }
    _write_json(output/"c07_decision.json", preliminary)
    if not c07_scientific_pass:
        preliminary.update({
            "overall_status":"FAIL_PRIMITIVE_RELATION_THREE_SEED_C07",
            "scientific_pass":False,
            "decision":"STOP_BEFORE_C08_AND_GRAPH",
            "duration_seconds":time.monotonic()-started,
        })
        _write_json(output/"summary.json", preliminary)
        _plot_result(preliminary, output/"primitive_relation_comparison")
        return 2

    # Do not even open or validate a C08 shard until the complete C07 gate has
    # passed in at least two independently trained seeds.
    c08_loader = PrimitiveRelationBatchLoader(
        args.sensor_root/"c08", args.teacher_root/"c08",
    )
    baseline_c08 = _nonlearning_partition(c08_loader, split="c08")
    _write_json(output/"baseline_c08.json", baseline_c08)
    c08_seed = []
    predictions_root = output/"c08_predictions"
    predictions_root.mkdir()
    for seed, model in enumerate(models):
        metrics = _learned_transfer(
            model, c08_loader, device=device,
            calibration=c07_seed[seed]["metrics"],
            output_root=predictions_root/f"seed{seed}",
        )
        comparison = _comparison(metrics, baseline_c08)
        record = {"seed":seed,"metrics":metrics,"comparison":comparison}
        c08_seed.append(record)
        _write_json(output/f"c08_seed{seed}.json", record)
        print(json.dumps({"stage":"c08","seed":seed,"comparison":comparison}), flush=True)
    c08_pass_count = sum(bool(value["comparison"]["pass"]) for value in c08_seed)
    scientific_pass = c08_pass_count >= 2
    summary = {
        "schema_version":"primitive_relation_three_seed_evaluation_v1",
        "overall_status":"PASS_PRIMITIVE_RELATION_THREE_SEED_V1" if scientific_pass else "FAIL_PRIMITIVE_RELATION_THREE_SEED_C08",
        "scientific_pass":scientific_pass,
        "c07":{"baseline":baseline_c07,"seeds":c07_seed,"passing_seeds":c07_pass_count,"scientific_pass":True},
        "c08":{"baseline":baseline_c08,"seeds":c08_seed,"passing_seeds":c08_pass_count,"scientific_pass":scientific_pass},
        "c08_rows_read":EXPECTED_ROWS["c08"]*4,
        "c09_c10_worlds_read":0,"graph_replays":0,
        "decision":"ALLOW_P3_OFFLINE_STRUCTURAL_GRAPH" if scientific_pass else "STOP_PRIMITIVE_RELATION_METHOD_BEFORE_GRAPH",
        "duration_seconds":time.monotonic()-started,
    }
    _write_json(output/"summary.json", summary)
    _plot_result(summary, output/"primitive_relation_comparison")
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
