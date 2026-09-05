from __future__ import annotations

import torch

from mtare_topo.representation.gse_structured_exact_one_event import (
    structured_exact_one_event_loss,
)


def _outputs(commit: list[float], event: list[int]) -> dict[str, torch.Tensor]:
    probability = torch.full((len(commit), 3), 0.01)
    for row, value in enumerate(event):
        probability[row, value] = 0.98
    return {
        "commit_logit": torch.logit(torch.tensor(commit), eps=1e-6).requires_grad_(),
        "event_logits": torch.log(probability).requires_grad_(),
    }


def _loss(commit: list[float], event: list[int]) -> torch.Tensor:
    output = _outputs(commit, event)
    return structured_exact_one_event_loss(
        output,
        torch.tensor([0, 1, 1, 1]),
        torch.tensor([-1, 0, 0, 0]),
    )["total"]


def test_one_correct_peak_beats_zero_two_and_wrong_peaks() -> None:
    one = _loss([0.01, 0.99, 0.01, 0.01], [0, 1, 1, 1])
    zero = _loss([0.01, 0.01, 0.01, 0.01], [0, 1, 1, 1])
    two = _loss([0.01, 0.99, 0.99, 0.01], [0, 1, 1, 1])
    wrong = _loss([0.01, 0.99, 0.01, 0.01], [0, 2, 1, 1])
    assert one < zero
    assert one < two
    assert one < wrong


def test_episode_row_permutation_is_invariant() -> None:
    output = _outputs([0.01, 0.9, 0.2, 0.1], [0, 1, 1, 1])
    target = torch.tensor([0, 1, 1, 1])
    episode = torch.tensor([-1, 0, 0, 0])
    first = structured_exact_one_event_loss(output, target, episode)["total"]
    order = torch.tensor([0, 3, 1, 2])
    moved = {key: value[order] for key, value in output.items()}
    second = structured_exact_one_event_loss(moved, target[order], episode[order])["total"]
    assert torch.allclose(first, second, atol=1e-6)


def test_extreme_logits_have_finite_backward() -> None:
    output = {
        "commit_logit": torch.tensor([-40.0, 40.0, -40.0, -40.0], requires_grad=True),
        "event_logits": torch.tensor([[40.0, -40.0, -40.0], [-40.0, 40.0, -40.0], [0.0, 0.0, 0.0], [40.0, -40.0, -40.0]], requires_grad=True),
    }
    loss = structured_exact_one_event_loss(
        output, torch.tensor([0, 1, 1, 1]), torch.tensor([-1, 0, 0, 0])
    )["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in output.values())
