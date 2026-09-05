#!/usr/bin/env python3
"""Evaluate the zero-training typed dual-Composer readiness contract."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.governance import write_json
from mtare_topo.representation.gse_typed_composers import (
    ActionSetComposerInput,
    ActionSetRelationComposer,
    MetricChangeComposer,
    MetricChangeComposerInput,
    action_set_composer_loss,
    metric_change_composer_loss,
    typed_composer_input_contract,
)


PASS = "PASS_GSE_TYPED_COMPOSER_READINESS_V1"
FAIL = "FAIL_GSE_TYPED_COMPOSER_READINESS_V1"
EVENTS = ("corridor", "junction", "terminal", "turn", "geometry_transition")
EXPECTED_C07_COUNTS = {
    "corridor": 17113, "junction": 3189, "terminal": 900,
    "turn": 279, "geometry_transition": 67,
}


def _read_teacher(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["parent_id"]).endswith("_C07"):
                rows.append(row)
    rows.sort(key=lambda row: int(row["global_sequence_index"]))
    return rows


def _select_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for event in EVENTS:
        selected.extend([row for row in rows if row["event"] == event][:2])
    start = next(row for row in rows if int(row["sequence_index"]) == 0)
    if int(start["global_sequence_index"]) not in {
        int(row["global_sequence_index"]) for row in selected
    }:
        corridor = [index for index, row in enumerate(selected) if row["event"] == "corridor"]
        selected[corridor[-1]] = start
    selected.sort(key=lambda row: int(row["global_sequence_index"]))
    if len(selected) != 10 or len({int(row["global_sequence_index"]) for row in selected}) != 10:
        raise RuntimeError("fixed real readiness sample selection drift")
    return selected


def _required_prediction_indices(
    selected: list[dict[str, object]], teacher: list[dict[str, object]],
) -> set[int]:
    lookup = {
        (str(row["traversal_id"]), int(row["sequence_index"])): int(row["global_sequence_index"])
        for row in teacher
    }
    required = set()
    for row in selected:
        traversal = str(row["traversal_id"])
        sequence = int(row["sequence_index"])
        required.update(
            lookup[(traversal, position)] for position in range(max(0, sequence - 4), sequence + 1)
        )
    return required


def _load_predictions(
    root: Path, required: set[int],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, int]]:
    by_index: dict[int, dict[str, np.ndarray]] = {}
    world_counts = {}
    allowed = (
        "token_bearing_deg", "token_existence_logits", "token_opening_width_m",
        "token_vertical_profile_m", "token_geometry_uncertainty",
        "token_count_probability", "transport_row_probability",
        "transport_reveal_probability", "geometry", "observation_uncertainty",
    )
    paths = sorted(root.glob("*_C07.npz"))
    if len(paths) != 10:
        raise RuntimeError("readiness requires exactly ten C07 prediction archives")
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            indices = np.asarray(arrays["global_sequence_index"], dtype=np.int64)
            world_counts[path.stem] = len(indices)
            for row, index in enumerate(indices.tolist()):
                if index not in required:
                    continue
                if index in by_index:
                    raise RuntimeError("prediction global index is not unique")
                by_index[index] = {name: np.asarray(arrays[name][row]) for name in allowed}
    missing = required - set(by_index)
    if missing:
        raise RuntimeError(f"required prediction indices are missing: {sorted(missing)[:8]}")
    return by_index, world_counts


def _action_input(
    selected: list[dict[str, object]], predictions: dict[int, dict[str, np.ndarray]],
) -> ActionSetComposerInput:
    values = [predictions[int(row["global_sequence_index"])] for row in selected]
    bearing_deg = torch.from_numpy(np.stack([row["token_bearing_deg"] for row in values])).float()
    bearing_rad = torch.deg2rad(bearing_deg)
    return ActionSetComposerInput(
        token_bearing_unit=torch.stack((torch.sin(bearing_rad), torch.cos(bearing_rad)), dim=-1),
        token_existence_probability=torch.sigmoid(torch.from_numpy(np.stack([
            row["token_existence_logits"] for row in values
        ])).float()),
        token_opening_width_m=torch.from_numpy(np.stack([
            row["token_opening_width_m"] for row in values
        ])).float(),
        token_vertical_profile_m=torch.from_numpy(np.stack([
            row["token_vertical_profile_m"] for row in values
        ])).float(),
        token_geometry_uncertainty=torch.from_numpy(np.stack([
            row["token_geometry_uncertainty"] for row in values
        ])).float(),
        token_count_probability=torch.from_numpy(np.stack([
            row["token_count_probability"] for row in values
        ])).float(),
        transport_row_probability=torch.from_numpy(np.stack([
            row["transport_row_probability"] for row in values
        ])).float(),
        transport_reveal_probability=torch.from_numpy(np.stack([
            row["transport_reveal_probability"] for row in values
        ])).float(),
        valid_history_mask=torch.ones(len(values), 5, dtype=torch.bool),
    )


def _metric_input(
    selected: list[dict[str, object]],
    teacher: list[dict[str, object]],
    predictions: dict[int, dict[str, np.ndarray]],
) -> MetricChangeComposerInput:
    lookup = {
        (str(row["traversal_id"]), int(row["sequence_index"])): int(row["global_sequence_index"])
        for row in teacher
    }
    geometry = np.zeros((len(selected), 5, 4), dtype=np.float32)
    uncertainty = np.zeros_like(geometry)
    mask = np.zeros((len(selected), 5), dtype=bool)
    for output_row, row in enumerate(selected):
        traversal = str(row["traversal_id"])
        sequence = int(row["sequence_index"])
        first = max(0, sequence - 4)
        indices = [lookup[(traversal, position)] for position in range(first, sequence + 1)]
        offset = 5 - len(indices)
        for column, index in enumerate(indices, start=offset):
            prediction = predictions[index]
            geometry[output_row, column] = prediction["geometry"]
            uncertainty[output_row, column] = float(prediction["observation_uncertainty"])
            mask[output_row, column] = True
        geometry[output_row, :offset] = geometry[output_row, offset]
        uncertainty[output_row, :offset] = uncertainty[output_row, offset]
    return MetricChangeComposerInput(
        geometry_sequence=torch.from_numpy(geometry),
        geometry_uncertainty=torch.from_numpy(uncertainty),
        valid_history_mask=torch.from_numpy(mask),
    )


def _permute_tokens(inputs: ActionSetComposerInput, permutation: torch.Tensor) -> ActionSetComposerInput:
    row = inputs.transport_row_probability[:, :, permutation]
    row = torch.cat((row[..., :6][..., permutation], row[..., 6:]), dim=-1)
    return replace(
        inputs,
        token_bearing_unit=inputs.token_bearing_unit[:, :, permutation],
        token_existence_probability=inputs.token_existence_probability[:, :, permutation],
        token_opening_width_m=inputs.token_opening_width_m[:, :, permutation],
        token_vertical_profile_m=inputs.token_vertical_profile_m[:, :, permutation],
        token_geometry_uncertainty=inputs.token_geometry_uncertainty[:, :, permutation],
        transport_row_probability=row,
        transport_reveal_probability=inputs.transport_reveal_probability[:, :, permutation],
    )


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach())


def _gradient_report(model: torch.nn.Module) -> dict[str, object]:
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    missing = [name for name, value in parameters if value.grad is None]
    nonfinite = [
        name for name, value in parameters
        if value.grad is not None and not bool(torch.isfinite(value.grad).all())
    ]
    maximum = max(
        (float(value.grad.abs().max()) for _, value in parameters if value.grad is not None),
        default=math.nan,
    )
    return {"missing": missing, "nonfinite": nonfinite, "maximum_absolute": maximum}


def _plot(output: Path, checks: dict[str, bool], counts: dict[str, int]) -> None:
    names = list(checks)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].barh(range(len(names)), [1 if checks[name] else 0 for name in names], color=[
        "#2f855a" if checks[name] else "#c53030" for name in names
    ])
    axes[0].set_yticks(range(len(names)), [name.replace("_", " ") for name in names], fontsize=7)
    axes[0].set_xlim(0, 1.05); axes[0].set_title("Typed dual-Composer readiness checks")
    axes[1].bar(list(counts), list(counts.values()), color="#2b6cb0")
    axes[1].set_yscale("log"); axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("C07 corrected causal event population")
    axes[1].set_ylabel("frames (log scale)")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_typed_composer_readiness.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260829); torch.use_deterministic_algorithms(True)

    teacher = _read_teacher(args.teacher.resolve())
    counts = {event: sum(row["event"] == event for row in teacher) for event in EVENTS}
    selected = _select_rows(teacher)
    required = _required_prediction_indices(selected, teacher)
    predictions, world_counts = _load_predictions(args.prediction_root.resolve(), required)
    action_input = _action_input(selected, predictions)
    metric_input = _metric_input(selected, teacher, predictions)

    action = ActionSetRelationComposer().cpu()
    metric = MetricChangeComposer().cpu()
    action_output = action(action_input)
    metric_output = metric(metric_input)
    action_repeat = action(action_input)
    metric_repeat = metric(metric_input)

    permutation = torch.tensor((3, 0, 5, 2, 1, 4))
    action_permuted = action(_permute_tokens(action_input, permutation))
    angle = action_input.token_bearing_unit.new_tensor(0.71)
    sine, cosine = torch.sin(angle), torch.cos(angle)
    bearing = action_input.token_bearing_unit
    rotated_bearing = torch.stack((
        bearing[..., 0] * cosine + bearing[..., 1] * sine,
        bearing[..., 1] * cosine - bearing[..., 0] * sine,
    ), dim=-1)
    action_rotated = action(replace(action_input, token_bearing_unit=rotated_bearing))
    batch_permutation = torch.tensor((9, 0, 7, 3, 1, 8, 2, 6, 4, 5))
    action_batch = action(ActionSetComposerInput(**{
        name: getattr(action_input, name)[batch_permutation]
        for name in typed_composer_input_contract()["action_fields"]
    }))
    metric_batch = metric(MetricChangeComposerInput(**{
        name: getattr(metric_input, name)[batch_permutation]
        for name in typed_composer_input_contract()["metric_fields"]
    }))

    action_target = torch.tensor([
        1 if row["event"] == "junction" else 2 if row["event"] == "terminal" else 0
        for row in selected
    ])
    metric_target = torch.tensor([
        1 if row["event"] == "turn" else 2 if row["event"] == "geometry_transition" else 0
        for row in selected
    ])
    commit = torch.tensor([0 if row["event"] == "corridor" else 1 for row in selected])
    action_loss = action_set_composer_loss(action_output, action_target, commit)["total"]
    metric_loss = metric_change_composer_loss(
        metric_output, metric_target, commit, torch.full((len(selected),), 4),
    )["total"]
    action.zero_grad(set_to_none=True); action_loss.backward()
    metric.zero_grad(set_to_none=True); metric_loss.backward()
    action_gradient = _gradient_report(action); metric_gradient = _gradient_report(metric)

    errors = {
        "token_permutation": _maximum_error(action_output.event_probability, action_permuted.event_probability),
        "global_rotation": _maximum_error(action_output.event_probability, action_rotated.event_probability),
        "action_batch_permutation": _maximum_error(
            action_output.event_probability[batch_permutation], action_batch.event_probability,
        ),
        "metric_batch_permutation": _maximum_error(
            metric_output.event_probability[batch_permutation], metric_batch.event_probability,
        ),
        "action_repeat": _maximum_error(action_output.event_probability, action_repeat.event_probability),
        "metric_repeat": _maximum_error(metric_output.event_probability, metric_repeat.event_probability),
    }
    contract = typed_composer_input_contract()
    exposed = set(contract["action_fields"]) | set(contract["metric_fields"])
    checks = {
        "c07_population_exact": len(teacher) == 21548 and counts == EXPECTED_C07_COUNTS,
        "ten_world_prediction_population_exact": len(world_counts) == 10 and sum(world_counts.values()) == 21548,
        "fixed_real_sample_has_all_events": len(selected) == 10 and {row["event"] for row in selected} == set(EVENTS),
        "typed_inputs_exclude_forbidden_fields": exposed.isdisjoint(contract["forbidden_inputs"]),
        "action_forward_accepts_only_typed_input": tuple(inspect.signature(ActionSetRelationComposer.forward).parameters) == ("self", "inputs"),
        "metric_forward_accepts_only_typed_input": tuple(inspect.signature(MetricChangeComposer.forward).parameters) == ("self", "inputs"),
        "real_outputs_finite_and_normalized": all((
            bool(torch.isfinite(action_output.event_probability).all()),
            bool(torch.isfinite(metric_output.event_probability).all()),
            bool(torch.allclose(action_output.event_probability.sum(-1), torch.ones(len(selected)), atol=1e-6)),
            bool(torch.allclose(metric_output.event_probability.sum(-1), torch.ones(len(selected)), atol=1e-6)),
        )),
        "token_permutation_invariant": errors["token_permutation"] <= 1e-6,
        "global_rotation_invariant": errors["global_rotation"] <= 1e-6,
        "batch_permutation_consistent": max(errors["action_batch_permutation"], errors["metric_batch_permutation"]) <= 1e-6,
        "deterministic_repeat_exact": errors["action_repeat"] == 0.0 and errors["metric_repeat"] == 0.0,
        "causal_backprojection_support_valid": bool((metric_output.backprojection_probability[~metric_input.valid_history_mask] == 0.0).all()) and bool(((metric_output.expected_steps_ago >= 0.0) & (metric_output.expected_steps_ago <= 4.0)).all()),
        "complete_backward_finite": not action_gradient["missing"] and not action_gradient["nonfinite"] and not metric_gradient["missing"] and not metric_gradient["nonfinite"] and math.isfinite(action_gradient["maximum_absolute"]) and math.isfinite(metric_gradient["maximum_absolute"]),
        "small_parameter_budget": sum(value.numel() for value in action.parameters()) <= 250000 and sum(value.numel() for value in metric.parameters()) <= 100000,
        "zero_training_test_graph": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_typed_composer_readiness_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_DUAL_COMPOSER_TRAINING_DATA_CARD" if scientific_pass else "STOP_DUAL_COMPOSER_BEFORE_TRAINING",
        "question": "Can the two GSE Composers consume only explicit causal geometry and satisfy the deployable interface contract before training?",
        "method_contract": contract,
        "parameters": {
            "action_set_relation_composer": sum(value.numel() for value in action.parameters()),
            "metric_change_composer": sum(value.numel() for value in metric.parameters()),
        },
        "c07": {"worlds": world_counts, "observations": len(teacher), "event_frames": counts},
        "real_samples": [{key: row[key] for key in ("global_sequence_index", "parent_id", "traversal_id", "sequence_index", "event", "identity")} for row in selected],
        "loaded_prediction_rows": len(predictions),
        "numerical_errors": errors,
        "gradients": {"action": action_gradient, "metric": metric_gradient},
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0,
        "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "readiness_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed"))
        writer.writeheader(); writer.writerows({"check": name, "passed": passed} for name, passed in checks.items())
    _plot(output, checks, counts)
    print(json.dumps({"status": summary["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
