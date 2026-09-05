from __future__ import annotations

import numpy as np
import torch

from mtare_topo.representation.gse_action_set_node import (
    ActionSetNodeDetector,
    CAUSAL_ACTION_FEATURE_DIM,
    causal_action_set_features,
    ensemble_action_set_signature,
    stack_raw_action_tokens,
    fit_action_token_normalization,
    raw_action_tokens_one_seed,
    action_set_episode_loss,
)


def _outputs(rows=4):
    confidence = np.tile(np.linspace(.1, .9, 6), (rows, 1)).astype(np.float32)
    angle = np.linspace(-2.0, 2.0, 6)
    heading = np.tile(np.stack((np.sin(angle), np.cos(angle)), axis=1), (rows, 1, 1)).astype(np.float32)
    return {
        "exit_confidence": confidence,
        "exit_heading_unit": heading,
        "exit_opening_width_m": np.tile(np.linspace(2, 4, 6), (rows, 1)).astype(np.float32),
        "exit_vertical_profile": np.ones((rows, 6, 4), dtype=np.float32),
        "exit_descriptor": np.ones((rows, 6, 32), dtype=np.float32),
    }


def test_action_set_signature_is_token_permutation_invariant() -> None:
    outputs = {seed: _outputs() for seed in range(3)}
    expected = ensemble_action_set_signature(outputs)
    permutation = [3, 0, 5, 2, 1, 4]
    shuffled = {
        seed: {name: value[:, permutation] for name, value in sample.items()}
        for seed, sample in outputs.items()
    }
    np.testing.assert_allclose(ensemble_action_set_signature(shuffled), expected, atol=1e-6)


def test_causal_action_features_do_not_cross_traversal() -> None:
    signature = ensemble_action_set_signature({seed: _outputs() for seed in range(3)})
    features = causal_action_set_features(signature, ["a", "a", "b", "b"], [0, 1, 0, 1])
    assert features.shape == (4, CAUSAL_ACTION_FEATURE_DIM)
    assert features[0, -1] == .2
    assert features[1, -1] == .4
    assert features[2, -1] == .2


def test_learned_action_set_detector_is_permutation_invariant() -> None:
    torch.manual_seed(0)
    model = ActionSetNodeDetector().eval()
    raw = stack_raw_action_tokens({
        seed: {"global_sequence_index": np.arange(4), **_outputs()} for seed in range(3)
    })
    tokens = torch.from_numpy(np.tile(raw[:1, None], (1, 5, 1, 1, 1)))
    mask = torch.ones(1, 5, dtype=torch.bool)
    expected = model(tokens, mask)["decision_probability"]
    shuffled = tokens[:, :, :, [3, 0, 5, 2, 1, 4]]
    actual = model(shuffled, mask)["decision_probability"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert expected.shape == (1, 3)
    torch.testing.assert_close(expected.sum(dim=1), torch.ones(1))


def test_action_set_detector_accepts_left_padded_past_only_history() -> None:
    model = ActionSetNodeDetector()
    tokens = torch.randn(2, 5, 3, 6, 40)
    tokens[..., 0].sigmoid_()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, :3] = False
    output = model(tokens, mask)
    assert output["decision_probability"].shape == (2, 3)


def test_action_set_episode_loss_needs_one_correct_row_per_episode() -> None:
    structural = torch.tensor([-4.0, 5.0, -3.0, -5.0], requires_grad=True)
    conditional = torch.tensor([
        [0.0, 0.0], [6.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    ], requires_grad=True)
    loss = action_set_episode_loss(
        {"structural_logit": structural, "conditional_decision_logits": conditional},
        torch.tensor([1, 1, 1, 0]), torch.tensor([0, 0, 0, -1]),
    )
    assert float(loss["negative"].detach()) < .02
    assert float(loss["positive_episode_joint"].detach()) < .02
    loss["total"].backward()
    assert structural.grad is not None and conditional.grad is not None


def test_action_token_normalization_preserves_confidence_and_heading() -> None:
    values = stack_raw_action_tokens({
        seed: {"global_sequence_index": np.arange(4), **_outputs()} for seed in range(3)
    })
    np.testing.assert_allclose(raw_action_tokens_one_seed(_outputs()), values[:, 0])
    mean, scale = fit_action_token_normalization(values, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(mean[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(scale[:3], np.ones(3, dtype=np.float32))
    assert np.all(scale[3:] > 0.0)
