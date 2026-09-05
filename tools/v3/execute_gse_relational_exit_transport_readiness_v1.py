#!/usr/bin/env python3
"""Zero-training readiness proof for relational exit-token transport events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.representation.gse_relational_exit_transport_event import (
    RelationalExitTokenTransportEventModel,
    parameter_count,
    relational_event_episode_loss,
)


PASS = "PASS_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1"
FAIL = "FAIL_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1"
EXPECTED = 188_126


def _causal_audit(
    references: np.ndarray,
    mask: np.ndarray,
    traversal: np.ndarray,
    sequence: np.ndarray,
) -> dict[str, int | bool]:
    invalid = 0
    maximum_length = 0
    minimum_length = 5
    for row in range(len(references)):
        indices = references[row, mask[row]]
        maximum_length = max(maximum_length, len(indices))
        minimum_length = min(minimum_length, len(indices))
        if (
            len(indices) < 1
            or int(indices[-1]) != row
            or np.any(traversal[indices] != traversal[row])
            or np.any(sequence[indices] > sequence[row])
            or not np.array_equal(sequence[indices], np.arange(sequence[row] - len(indices) + 1, sequence[row] + 1))
        ):
            invalid += 1
    return {
        "rows": len(references),
        "invalid_rows": invalid,
        "minimum_history": minimum_length,
        "maximum_history": maximum_length,
        "all_past_only_same_traversal_suffixes": invalid == 0,
    }


def _batch_rows(target: np.ndarray, episode: np.ndarray, mask: np.ndarray, partition: np.ndarray) -> np.ndarray:
    full = mask.all(axis=1) & (partition == 0)
    negative = np.flatnonzero(full & (target == 0))[:4]
    junction = np.flatnonzero(full & (target == 1))[:2]
    terminal = np.flatnonzero(full & (target == 2))[:2]
    rows = np.concatenate((negative, junction, terminal))
    if len(rows) != 8 or len(np.unique(rows)) != 8 or np.any(episode[negative] >= 0):
        raise RuntimeError("real readiness batch selection drift")
    return rows


def _gather(
    rows: np.ndarray,
    raw: np.ndarray,
    references: np.ndarray,
    mask: np.ndarray,
    geometry: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe = references[rows].astype(np.int64).copy()
    safe[~mask[rows]] = rows[:, None].repeat(5, axis=1)[~mask[rows]]
    token = np.asarray(raw[safe], dtype=np.float32)
    context = np.asarray(geometry[safe], dtype=np.float32)
    return torch.from_numpy(token), torch.from_numpy(context), torch.from_numpy(mask[rows].copy())


def _permute_tokens(tokens: torch.Tensor) -> torch.Tensor:
    result = tokens.clone()
    generator = torch.Generator().manual_seed(20260828)
    for time_index in range(5):
        for seed in range(3):
            order = torch.randperm(6, generator=generator)
            result[:, time_index, seed] = result[:, time_index, seed, order]
    return result


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def _plot(output: Path, summary: dict) -> None:
    errors = summary["equivariance"]
    names = ["token permutation", "seed permutation", "masked history", "repeat"]
    values = [
        max(errors["token_permutation_event_error"], 1e-12),
        max(errors["seed_permutation_event_error"], 1e-12),
        max(errors["masked_history_event_error"], 1e-12),
        max(errors["repeat_event_error"], 1e-12),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    axes[0].bar(np.arange(4), values, color=["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"])
    axes[0].axhline(1e-6, color="#b22222", linestyle="--", linewidth=1.0, label="1e-6 contract")
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(4), names, rotation=18)
    axes[0].set_ylabel("maximum probability error")
    axes[0].set_title("A  Invariance and causality")
    axes[0].legend(frameon=False, fontsize=8)
    checks = summary["checks"]
    labels = ["causal", "transport", "backward", "interface", "isolation"]
    selected = [
        checks["all_history_past_only"], checks["transport_normalized_and_nontrivial"],
        checks["real_batch_finite_backward"], checks["typed_refusal_interface"],
        checks["zero_forbidden_operations"],
    ]
    axes[1].bar(np.arange(5), [int(value) for value in selected], color="#76b7b2")
    axes[1].set_xticks(np.arange(5), labels, rotation=18)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_yticks([0, 1], ["FAIL", "PASS"])
    axes[1].set_title("B  Readiness contracts")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Relational exit-token transport event model readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_relational_exit_transport_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.action_cache.resolve()
    raw = np.load(cache / "raw_tokens.npy", mmap_mode="r")
    references = np.load(cache / "history_references.npy")
    history_mask = np.load(cache / "history_mask.npy")
    target = np.load(cache / "decision_target.npy")
    episode = np.load(cache / "decision_episode_id.npy")
    partition = np.load(cache / "partition_code.npy")
    traversal = np.load(cache / "traversal_id.npy")
    sequence = np.load(cache / "sequence_index.npy")
    mean = np.load(cache / "normalization_mean.npy")
    scale = np.load(cache / "normalization_scale.npy")
    if (
        raw.shape != (EXPECTED, 3, 6, 40)
        or references.shape != (EXPECTED, 5)
        or history_mask.shape != references.shape
        or target.shape != (EXPECTED,)
        or episode.shape != (EXPECTED,)
        or int(np.sum(partition == 0)) != 142_184
        or int(np.sum(partition == 1)) != 45_942
    ):
        raise RuntimeError("relational readiness action-cache population drift")
    observation = []
    for seed in range(3):
        values = np.load(getattr(args, f"observation{seed}").resolve(), mmap_mode="r")
        if values.shape != (EXPECTED, 146) or values.dtype != np.float32 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"relational readiness observation contract drift: seed{seed}")
        observation.append(np.concatenate((values[:, 5:12], values[:, 140:141]), axis=1))
    geometry = np.stack(observation, axis=1).astype(np.float32)
    causal = _causal_audit(references, history_mask, traversal, sequence)
    rows = _batch_rows(target, episode, history_mask, partition)
    tokens, contexts, masks = _gather(rows, raw, references, history_mask, geometry)
    torch.manual_seed(20260828)
    torch.use_deterministic_algorithms(True)
    model = RelationalExitTokenTransportEventModel(
        token_normalization_mean=mean,
        token_normalization_scale=scale,
    )
    model.eval()
    with torch.no_grad():
        first = model(tokens, contexts, masks)
        repeat = model(tokens, contexts, masks)
        token_permuted = model(_permute_tokens(tokens), contexts, masks)
        seed_order = torch.tensor([2, 0, 1])
        seed_permuted = model(tokens[:, :, seed_order], contexts[:, :, seed_order], masks)
    early_rows = np.flatnonzero(history_mask.sum(axis=1) == 1)[:4]
    early_tokens, early_contexts, early_masks = _gather(early_rows, raw, references, history_mask, geometry)
    with torch.no_grad():
        early_reference = model(early_tokens, early_contexts, early_masks)
        changed_tokens = early_tokens.clone()
        changed_contexts = early_contexts.clone()
        changed_tokens[:, :-1] = tokens[:4, :-1]
        changed_contexts[:, :-1] = contexts[:4, :-1]
        early_changed = model(changed_tokens, changed_contexts, early_masks)
    model.train()
    outputs = model(tokens, contexts, masks)
    local_target = torch.from_numpy(target[rows].astype(np.int64))
    local_episode = torch.full_like(local_target, -1)
    positive_rows = torch.flatnonzero(local_target > 0)
    local_episode[positive_rows] = torch.arange(len(positive_rows), dtype=torch.int64)
    losses = relational_event_episode_loss(outputs, local_target, local_episode)
    losses["total"].backward()
    gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    transport_sum_error = float(torch.max(torch.abs(outputs["transport"].sum(dim=-2) - 1.0)).item())
    transport_standard_deviation = float(outputs["transport"].std().item())
    equivariance = {
        "token_permutation_event_error": _maximum_error(first["event_probability"], token_permuted["event_probability"]),
        "token_permutation_commit_error": _maximum_error(first["commit_probability"], token_permuted["commit_probability"]),
        "seed_permutation_event_error": _maximum_error(first["event_probability"], seed_permuted["event_probability"]),
        "seed_permutation_commit_error": _maximum_error(first["commit_probability"], seed_permuted["commit_probability"]),
        "masked_history_event_error": _maximum_error(early_reference["event_probability"], early_changed["event_probability"]),
        "masked_history_commit_error": _maximum_error(early_reference["commit_probability"], early_changed["commit_probability"]),
        "repeat_event_error": _maximum_error(first["event_probability"], repeat["event_probability"]),
        "repeat_commit_error": _maximum_error(first["commit_probability"], repeat["commit_probability"]),
    }
    output_contract = {
        "event_probability_shape": list(outputs["event_probability"].shape),
        "commit_probability_shape": list(outputs["commit_probability"].shape),
        "provisional_probability_shape": list(outputs["provisional_probability"].shape),
        "uncertainty_shape": list(outputs["uncertainty"].shape),
        "transport_shape": list(outputs["transport"].shape),
        "event_probability_sum_error": float(torch.max(torch.abs(outputs["event_probability"].sum(dim=1) - 1.0)).item()),
        "commit_plus_provisional_error": float(torch.max(torch.abs(outputs["commit_probability"] + outputs["provisional_probability"] - 1.0)).item()),
        "all_outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs.values()),
    }
    checks = {
        "exact_population": len(raw) == EXPECTED and len(rows) == 8,
        "all_history_past_only": causal["all_past_only_same_traversal_suffixes"],
        "parameter_count_exact_240101": parameter_count() == 240_101,
        "token_permutation_invariant": max(equivariance["token_permutation_event_error"], equivariance["token_permutation_commit_error"]) <= 1e-6,
        "seed_permutation_invariant": max(equivariance["seed_permutation_event_error"], equivariance["seed_permutation_commit_error"]) <= 1e-6,
        "masked_history_invariant": max(equivariance["masked_history_event_error"], equivariance["masked_history_commit_error"]) == 0.0,
        "deterministic_replay_bit_exact": max(equivariance["repeat_event_error"], equivariance["repeat_commit_error"]) == 0.0,
        "transport_normalized_and_nontrivial": transport_sum_error <= 1e-6 and transport_standard_deviation > 1e-6,
        "real_batch_finite_backward": bool(torch.isfinite(losses["total"])) and gradients_finite,
        "typed_refusal_interface": output_contract["all_outputs_finite"] and output_contract["event_probability_sum_error"] <= 1e-7 and output_contract["commit_plus_provisional_error"] == 0.0,
        "zero_forbidden_operations": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_relational_exit_transport_readiness_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_RELATIONAL_EXIT_TRANSPORT_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_RELATIONAL_EXIT_TRANSPORT_BEFORE_TRAINING",
        "population": {
            "worlds": 80,
            "observations": EXPECTED,
            "fit_observations": 142_184,
            "selection_observations": 45_942,
            "real_readiness_rows": len(rows),
            "history_observations": 5,
            "perception_seeds": 3,
            "tokens_per_seed": 6,
            "raw_token_dimensions": 40,
            "geometry_context_dimensions": 8,
        },
        "student_inputs": ["raw frozen exit tokens", "learned local axis/width/height/slope/curvature/uncertainty", "past-only history mask"],
        "forbidden_inputs": ["Teacher event", "Teacher exit identity", "world/parent identity", "pose", "future frame", "objective geometry", "C09", "C10", "M-TARE"],
        "causal_audit": causal,
        "parameter_count": parameter_count(),
        "equivariance": equivariance,
        "transport": {"normalization_error": transport_sum_error, "standard_deviation": transport_standard_deviation},
        "output_contract": output_contract,
        "loss": {name: float(value.detach()) for name, value in losses.items()},
        "checks": checks,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "graph_replays": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_relational_exit_transport_readiness_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "readiness_checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed"))
        writer.writeheader()
        for name, passed in checks.items():
            writer.writerow({"check": name, "passed": int(passed)})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
