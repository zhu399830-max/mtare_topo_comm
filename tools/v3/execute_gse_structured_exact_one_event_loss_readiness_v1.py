#!/usr/bin/env python3
"""Zero-training readiness for structured exactly-one event likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.gse_causal_episode_sampler import EpisodePreservingBatchSampler
from mtare_topo.data.gse_relational_exit_transport_cache import RelationalExitTransportDataset
from mtare_topo.representation.gse_relational_exit_transport_event import (
    RelationalExitTokenTransportEventModel,
    parameter_count,
    relational_event_episode_loss,
)
from mtare_topo.representation.gse_structured_exact_one_event import (
    structured_exact_one_event_loss,
)


PASS = "PASS_GSE_STRUCTURED_EXACT_ONE_EVENT_LOSS_READINESS_V1"
FAIL = "FAIL_GSE_STRUCTURED_EXACT_ONE_EVENT_LOSS_READINESS_V1"


def _synthetic_outputs(commit: list[float], event: list[int]) -> dict[str, torch.Tensor]:
    probability = torch.full((len(commit), 3), 0.01, dtype=torch.float64)
    for row, value in enumerate(event):
        probability[row, value] = 0.98
    return {
        "commit_logit": torch.logit(torch.tensor(commit, dtype=torch.float64), eps=1e-12).requires_grad_(),
        "event_logits": torch.log(probability).requires_grad_(),
    }


def _synthetic_loss(commit: list[float], event: list[int]) -> float:
    return float(structured_exact_one_event_loss(
        _synthetic_outputs(commit, event),
        torch.tensor([0, 1, 1, 1]),
        torch.tensor([-1, 0, 0, 0]),
    )["total"].detach())


def _gradient_attribution() -> dict[str, float]:
    target = torch.tensor([0, 1, 1, 1])
    episode = torch.tensor([-1, 0, 0, 0])
    event_logits = torch.tensor([[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, 5.0, -5.0]])
    old_commit = torch.tensor([-5.0, 5.0, 3.0, -5.0], requires_grad=True)
    old = relational_event_episode_loss(
        {"event_logits": event_logits.clone().requires_grad_(), "commit_logit": old_commit},
        target, episode,
    )["total"]
    old.backward()
    new_commit = torch.tensor([-5.0, 5.0, 3.0, -5.0], requires_grad=True)
    new = structured_exact_one_event_loss(
        {"event_logits": event_logits.clone().requires_grad_(), "commit_logit": new_commit},
        target, episode,
    )["total"]
    new.backward()
    return {
        "old_loss": float(old.detach()), "new_loss": float(new.detach()),
        "old_second_positive_peak_gradient": float(old_commit.grad[2]),
        "new_second_positive_peak_gradient": float(new_commit.grad[2]),
    }


def _plot(output: Path, result: dict) -> None:
    synthetic = result["synthetic"]
    gradient = result["gradient_attribution"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    labels = ["One correct", "No peak", "Two peaks", "Wrong peak"]
    values = [synthetic[key] for key in ("one_correct_peak", "zero_peak", "two_correct_peaks", "wrong_class_peak")]
    axes[0].bar(labels, values, color=["#59a14f", "#bab0ab", "#f28e2b", "#e15759"])
    axes[0].set_ylabel("Exact-one negative log-likelihood")
    axes[0].set_title("A  Structured episode preference")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(["Old max-MIL", "Exact-one"], [abs(gradient["old_second_positive_peak_gradient"]), abs(gradient["new_second_positive_peak_gradient"])], color=["#9c755f", "#4e79a7"])
    axes[1].set_ylabel("|gradient on second commit peak|")
    axes[1].set_title("B  Duplicate-peak supervision")
    for axis in axes:
        axis.grid(axis="y", alpha=0.23)
        axis.set_axisbelow(True)
    figure.suptitle("Structured exact-one likelihood directly supervises one graph-node commit")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_structured_exact_one_event_loss_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--observation", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("exact-one readiness output exists; overwrite is forbidden")
    if len(args.observation) != 3:
        raise RuntimeError("exact-one readiness requires three frozen geometry observations")
    args.output_dir.mkdir(parents=True)
    partition = np.load(args.cache_dir / "partition_code.npy")
    target = np.load(args.cache_dir / "decision_target.npy")
    episode = np.load(args.cache_dir / "decision_episode_id.npy")
    references = np.load(args.cache_dir / "history_references.npy")
    history_mask = np.load(args.cache_dir / "history_mask.npy")
    fit_rows = np.flatnonzero(partition == 0)
    selection_rows = np.flatnonzero(partition == 1)
    if (
        len(partition) != 188_126 or len(fit_rows) != 142_184 or len(selection_rows) != 45_942
        or references.shape != (188_126, 5) or history_mask.shape != (188_126, 5)
        or int(np.sum(episode >= 0)) == 0
    ):
        raise RuntimeError("exact-one readiness population drift")
    synthetic = {
        "one_correct_peak": _synthetic_loss([0.01, 0.99, 0.01, 0.01], [0, 1, 1, 1]),
        "zero_peak": _synthetic_loss([0.01, 0.01, 0.01, 0.01], [0, 1, 1, 1]),
        "two_correct_peaks": _synthetic_loss([0.01, 0.99, 0.99, 0.01], [0, 1, 1, 1]),
        "wrong_class_peak": _synthetic_loss([0.01, 0.99, 0.01, 0.01], [0, 2, 1, 1]),
    }
    gradient = _gradient_attribution()
    permutation_output = _synthetic_outputs([0.01, 0.9, 0.2, 0.1], [0, 1, 1, 1])
    permutation_target = torch.tensor([0, 1, 1, 1])
    permutation_episode = torch.tensor([-1, 0, 0, 0])
    first = structured_exact_one_event_loss(permutation_output, permutation_target, permutation_episode)["total"]
    order = torch.tensor([0, 3, 1, 2])
    second = structured_exact_one_event_loss(
        {key: value[order] for key, value in permutation_output.items()},
        permutation_target[order], permutation_episode[order],
    )["total"]
    permutation_error = float(torch.abs(first - second).detach())
    extreme = {
        "commit_logit": torch.tensor([-40.0, 40.0, -40.0, -40.0], requires_grad=True),
        "event_logits": torch.tensor([[40.0, -40.0, -40.0], [-40.0, 40.0, -40.0], [0.0, 0.0, 0.0], [40.0, -40.0, -40.0]], requires_grad=True),
    }
    extreme_loss = structured_exact_one_event_loss(
        extreme, torch.tensor([0, 1, 1, 1]), torch.tensor([-1, 0, 0, 0])
    )["total"]
    extreme_loss.backward()
    extreme_gradients_finite = all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in extreme.values())

    dataset = RelationalExitTransportDataset(args.cache_dir, args.observation, fit_rows)
    sampler = EpisodePreservingBatchSampler(dataset.episode_id, batch_size=128, seed=20260829)
    batch_indices = next(iter(sampler))
    batch_episode = dataset.episode_id[batch_indices]
    complete_episode = all(
        int(np.sum(batch_episode == identity)) == int(np.sum(dataset.episode_id == identity))
        for identity in np.unique(batch_episode[batch_episode >= 0])
    )
    samples = [dataset[index] for index in batch_indices]
    tokens = torch.from_numpy(np.stack([sample["tokens"] for sample in samples]))
    geometry = torch.from_numpy(np.stack([sample["geometry_context"] for sample in samples]))
    mask = torch.from_numpy(np.stack([sample["history_mask"] for sample in samples]))
    batch_target = torch.from_numpy(np.asarray([sample["decision_target"] for sample in samples]))
    batch_episode_tensor = torch.from_numpy(np.asarray([sample["episode_id"] for sample in samples]))
    torch.manual_seed(20260829)
    model = RelationalExitTokenTransportEventModel(
        token_normalization_mean=np.load(args.cache_dir / "normalization_mean.npy"),
        token_normalization_scale=np.load(args.cache_dir / "normalization_scale.npy"),
    )
    predicted = model(tokens, geometry, mask)
    real = structured_exact_one_event_loss(predicted, batch_target, batch_episode_tensor)
    real["total"].backward()
    real_gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    real_metrics = {
        "rows": len(batch_indices),
        "positive_rows": int(np.sum(batch_episode >= 0)),
        "negative_rows": int(np.sum(batch_episode < 0)),
        "complete_positive_episodes": int(len(np.unique(batch_episode[batch_episode >= 0]))),
        "complete_episode_membership": complete_episode,
        "loss": {key: float(value.detach()) for key, value in real.items()},
        "gradients_finite": real_gradients_finite,
    }
    gates = {
        "population_exact": len(partition) == 188_126 and len(fit_rows) == 142_184 and len(selection_rows) == 45_942,
        "one_peak_beats_zero": synthetic["one_correct_peak"] < synthetic["zero_peak"],
        "one_peak_beats_two": synthetic["one_correct_peak"] < synthetic["two_correct_peaks"],
        "one_peak_beats_wrong": synthetic["one_correct_peak"] < synthetic["wrong_class_peak"],
        "old_second_peak_gradient_zero": abs(gradient["old_second_positive_peak_gradient"]) <= 1e-12,
        "exact_one_second_peak_gradient_nonzero": gradient["new_second_positive_peak_gradient"] > 1e-3,
        "episode_permutation_error_at_most_1e-6": permutation_error <= 1e-6,
        "extreme_loss_and_gradients_finite": bool(torch.isfinite(extreme_loss)) and extreme_gradients_finite,
        "real_batch_complete": complete_episode and real_metrics["complete_positive_episodes"] > 0 and real_metrics["positive_rows"] > 0 and real_metrics["negative_rows"] > 0,
        "real_loss_and_gradients_finite": bool(torch.isfinite(real["total"])) and real_gradients_finite,
        "parameter_count_unchanged": parameter_count() == 240_101,
        "zero_training_test_graph": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": "gse_structured_exact_one_event_loss_readiness_v1",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "decision": "ALLOW_STRUCTURED_EXACT_ONE_EVENT_THREE_SEED_DATA_CARD" if passed else "STOP_STRUCTURED_EXACT_ONE_EVENT_ROUTE",
        "population": {"observations": len(partition), "fit_observations": len(fit_rows), "selection_observations": len(selection_rows)},
        "parameters": parameter_count(),
        "synthetic": synthetic,
        "gradient_attribution": gradient,
        "episode_permutation_error": permutation_error,
        "extreme_loss": float(extreme_loss.detach()),
        "extreme_gradients_finite": extreme_gradients_finite,
        "real_batch": real_metrics,
        "gates": gates,
        "optimizer_steps": 0, "trained_model_inference_frames": 0, "untrained_readiness_forward_rows": len(batch_indices),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "figure_source.json").write_text(json.dumps({"schema_version": "gse_structured_exact_one_event_loss_figure_source_v1", "summary": result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(args.output_dir, result)
    print(json.dumps({"status": result["status"], "decision": result["decision"], "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
