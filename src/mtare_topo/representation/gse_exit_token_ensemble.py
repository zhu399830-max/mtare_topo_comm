"""Frozen three-seed exit-token association used by online GSE-Graph.

The deployment backend consumes only two frozen GSE observations and their
measured spatial separation.  Spatial candidate enumeration may be
precomputed for efficiency, but every score is pair-local and independent of
future graph state, Teacher identities, traversal IDs, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mtare_topo.representation.gse_exit_token_association import (
    GSEExitTokenAssociationVerifier,
    exit_token_pair_features,
)
from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    observation_features,
    symmetric_pair_features,
)


FROZEN_ENSEMBLE_THRESHOLD = 0.9431912302970886
FROZEN_NODE_MATCHABILITY_THRESHOLD = 0.982292910416921
FROZEN_EVENT_NODE_THRESHOLD = 0.9597587988776491
FROZEN_ENSEMBLE_SEEDS = (0, 1, 2)


def _raw_token_outputs(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = (
        "exit_confidence",
        "exit_heading_unit",
        "exit_opening_width_m",
        "exit_vertical_profile",
        "exit_descriptor",
    )
    result = {name: np.asarray(outputs[name], dtype=np.float32) for name in names}
    rows = len(result["exit_confidence"])
    if any(len(value) != rows or not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("frozen exit-token outputs are invalid or misaligned")
    return result


def _pair_code(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    low = np.minimum(left, right).astype(np.uint64)
    high = np.maximum(left, right).astype(np.uint64)
    if np.any(low > np.iinfo(np.uint32).max) or np.any(high > np.iinfo(np.uint32).max):
        raise ValueError("sequence identity exceeds the frozen uint32 pair-code contract")
    return (low << np.uint64(32)) | high


@dataclass(frozen=True)
class FrozenAssociationDecision:
    seed_scores: tuple[float, float, float]
    ensemble_score: float
    distance_m: float
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_scores": list(self.seed_scores),
            "ensemble_score": self.ensemble_score,
            "distance_m": self.distance_m,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


class FrozenWorldAssociationScores:
    """Read-only score table for one world, shared by every causal replay."""

    def __init__(
        self,
        pair_codes: np.ndarray,
        distances_m: np.ndarray,
        seed_scores: np.ndarray,
        *,
        threshold: float,
        maximum_candidate_distance_m: float,
        node_keys: np.ndarray | None = None,
        node_seed_scores: np.ndarray | None = None,
        node_threshold: float = FROZEN_NODE_MATCHABILITY_THRESHOLD,
        node_event_probabilities: np.ndarray | None = None,
        event_node_threshold: float = FROZEN_EVENT_NODE_THRESHOLD,
    ) -> None:
        order = np.argsort(pair_codes, kind="stable")
        self._pair_codes = np.asarray(pair_codes, dtype=np.uint64)[order]
        self._distances_m = np.asarray(distances_m, dtype=np.float32)[order]
        self._seed_scores = np.asarray(seed_scores, dtype=np.float32)[:, order]
        if (
            self._pair_codes.ndim != 1
            or self._distances_m.shape != self._pair_codes.shape
            or self._seed_scores.shape != (3, len(self._pair_codes))
            or len(np.unique(self._pair_codes)) != len(self._pair_codes)
            or not np.all(np.isfinite(self._seed_scores))
        ):
            raise ValueError("world association score table violates its frozen contract")
        self.threshold = float(threshold)
        self.maximum_candidate_distance_m = float(maximum_candidate_distance_m)
        self.node_threshold = float(node_threshold)
        self.event_node_threshold = float(event_node_threshold)
        if node_keys is None or node_seed_scores is None:
            self._node_keys = None
            self._node_seed_scores = None
            self._node_event_probabilities = None
        else:
            node_order = np.argsort(node_keys, kind="stable")
            self._node_keys = np.asarray(node_keys, dtype=np.int64)[node_order]
            self._node_seed_scores = np.asarray(node_seed_scores, dtype=np.float32)[:, node_order]
            self._node_event_probabilities = (
                None
                if node_event_probabilities is None
                else np.asarray(node_event_probabilities, dtype=np.float32)[:, node_order]
            )
            if (
                self._node_seed_scores.shape != (3, len(self._node_keys))
                or len(np.unique(self._node_keys)) != len(self._node_keys)
                or not np.all(np.isfinite(self._node_seed_scores))
            ):
                raise ValueError("world node-matchability table violates its frozen contract")
            if self._node_event_probabilities is not None and (
                self._node_event_probabilities.shape != (3, len(self._node_keys), 5)
                or not np.all(np.isfinite(self._node_event_probabilities))
                or not np.allclose(self._node_event_probabilities.sum(axis=2), 1.0, atol=1e-5)
            ):
                raise ValueError("world event-probability table violates its frozen contract")

    @property
    def pair_count(self) -> int:
        return int(len(self._pair_codes))

    def evaluate_pair(self, left_key: int, right_key: int, distance_m: float) -> dict[str, Any]:
        distance = float(distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("candidate distance must be finite and nonnegative")
        if distance > self.maximum_candidate_distance_m + 1e-9:
            return FrozenAssociationDecision(
                seed_scores=(0.0, 0.0, 0.0),
                ensemble_score=0.0,
                distance_m=distance,
                accepted=False,
                rejection_reason="outside_16m_candidate_domain",
            ).to_dict()
        code = _pair_code(np.asarray([left_key]), np.asarray([right_key]))[0]
        index = int(np.searchsorted(self._pair_codes, code))
        if index >= len(self._pair_codes) or self._pair_codes[index] != code:
            raise RuntimeError("eligible online candidate pair was not precomputed")
        expected_distance = float(self._distances_m[index])
        if abs(expected_distance - distance) > 1e-3:
            raise RuntimeError("online candidate distance drifted from frozen score table")
        values = tuple(float(value) for value in self._seed_scores[:, index])
        ensemble = float(np.mean(values, dtype=np.float64))
        accepted = ensemble >= self.threshold
        return FrozenAssociationDecision(
            seed_scores=values,
            ensemble_score=ensemble,
            distance_m=distance,
            accepted=accepted,
            rejection_reason=None if accepted else "below_frozen_ensemble_threshold",
        ).to_dict()

    def evaluate_node(self, key: int) -> dict[str, Any]:
        if self._node_keys is None or self._node_seed_scores is None:
            raise RuntimeError("world score table has no frozen node-matchability evidence")
        index = int(np.searchsorted(self._node_keys, int(key)))
        if index >= len(self._node_keys) or int(self._node_keys[index]) != int(key):
            raise RuntimeError("deployment observation key is absent from node-matchability table")
        values = tuple(float(value) for value in self._node_seed_scores[:, index])
        ensemble = float(np.mean(values, dtype=np.float64))
        if self._node_event_probabilities is None:
            accepted = ensemble >= self.node_threshold
            return {
                "seed_scores": list(values),
                "ensemble_score": ensemble,
                "threshold": self.node_threshold,
                "accepted": accepted,
                "rejection_reason": None if accepted else "below_frozen_node_matchability_threshold",
            }
        event_probability = self._node_event_probabilities[:, index].mean(axis=0, dtype=np.float64)
        structural_probability = float(1.0 - event_probability[0])
        combined = float(ensemble * structural_probability)
        accepted = combined >= self.event_node_threshold
        return {
            "seed_scores": list(values),
            "ensemble_score": ensemble,
            "mean_event_probability": event_probability.tolist(),
            "structural_probability": structural_probability,
            "combined_score": combined,
            "threshold": self.event_node_threshold,
            "accepted": accepted,
            "rejection_reason": None if accepted else "below_frozen_event_node_threshold",
        }


class FrozenExitTokenEnsembleRuntime:
    """Load frozen V2 verifiers and materialize pair-local C09 scores."""

    def __init__(
        self,
        *,
        outputs_by_seed: Mapping[int, Mapping[str, Any]],
        model_directories: Mapping[int, Path],
        threshold: float = FROZEN_ENSEMBLE_THRESHOLD,
        maximum_candidate_distance_m: float = 16.0,
        device: str | torch.device = "cpu",
        event_node_gate: bool = False,
    ) -> None:
        if tuple(sorted(outputs_by_seed)) != FROZEN_ENSEMBLE_SEEDS:
            raise ValueError("deployment requires exactly GSE seeds 0/1/2")
        if tuple(sorted(model_directories)) != FROZEN_ENSEMBLE_SEEDS:
            raise ValueError("deployment requires exactly verifier seeds 0/1/2")
        contract = OpenSetAssociationContract()
        if abs(maximum_candidate_distance_m - contract.maximum_candidate_distance_m) > 1e-12:
            raise ValueError("deployment candidate distance must remain frozen at 16 m")
        if abs(threshold - FROZEN_ENSEMBLE_THRESHOLD) > 1e-15:
            raise ValueError("deployment ensemble threshold drift")
        self.threshold = float(threshold)
        self.maximum_candidate_distance_m = float(maximum_candidate_distance_m)
        self.device = torch.device(device)
        self.event_node_gate = bool(event_node_gate)
        self._keys: dict[int, np.ndarray] = {}
        self._row_by_key: dict[int, dict[int, int]] = {}
        self._normalized_observation: dict[int, np.ndarray] = {}
        self._tokens: dict[int, dict[str, np.ndarray]] = {}
        self._pair_mean: dict[int, np.ndarray] = {}
        self._pair_std: dict[int, np.ndarray] = {}
        self._models: dict[int, GSEExitTokenAssociationVerifier] = {}
        self._node_matchability: dict[int, np.ndarray] = {}
        self._event_probabilities: dict[int, np.ndarray] = {}
        for seed in FROZEN_ENSEMBLE_SEEDS:
            outputs = outputs_by_seed[seed]
            keys = np.asarray(outputs["global_sequence_index"], dtype=np.int64)
            if keys.ndim != 1 or len(np.unique(keys)) != len(keys):
                raise ValueError(f"seed{seed} output sequence identities are not unique")
            directory = Path(model_directories[seed]).resolve()
            with np.load(directory / "normalization.npz", allow_pickle=False) as archive:
                normalization = {name: archive[name].astype(np.float32) for name in archive.files}
            features = observation_features(outputs)
            self._event_probabilities[seed] = features[:, :5].copy()
            normalized = (
                (features - normalization["observation_mean"]) / normalization["observation_std"]
            ).astype(np.float32)
            checkpoint = torch.load(directory / "best.pt", map_location=self.device, weights_only=False)
            if checkpoint.get("schema_version") != "gse_exit_token_association_checkpoint_v2":
                raise RuntimeError(f"seed{seed} verifier checkpoint schema drift")
            model = GSEExitTokenAssociationVerifier().to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self._keys[seed] = keys
            self._row_by_key[seed] = {int(key): row for row, key in enumerate(keys)}
            self._normalized_observation[seed] = normalized
            self._tokens[seed] = _raw_token_outputs(outputs)
            self._pair_mean[seed] = normalization["pair_mean"]
            self._pair_std[seed] = normalization["pair_std"]
            self._models[seed] = model
            node_scores = []
            with torch.inference_mode():
                for start in range(0, len(normalized), 8192):
                    values = torch.from_numpy(normalized[start : start + 8192]).to(self.device)
                    node_scores.append(torch.sigmoid(model.matchability(values).squeeze(1)).cpu().numpy())
            self._node_matchability[seed] = np.concatenate(node_scores).astype(np.float32)
        reference = set(self._row_by_key[0])
        if any(set(self._row_by_key[seed]) != reference for seed in FROZEN_ENSEMBLE_SEEDS[1:]):
            raise RuntimeError("three GSE seed output identity sets differ")

    def prepare_world(
        self,
        sequence_keys: Sequence[int],
        xyz_by_key: Mapping[int, Sequence[float]],
        *,
        batch_size: int = 8192,
    ) -> FrozenWorldAssociationScores:
        keys = np.asarray(sorted({int(value) for value in sequence_keys}), dtype=np.int64)
        if len(keys) != len(sequence_keys) or any(int(key) not in xyz_by_key for key in keys):
            raise ValueError("world sequence keys must be unique and have replay positions")
        xyz = np.stack([np.asarray(xyz_by_key[int(key)], dtype=np.float64) for key in keys])
        if xyz.shape != (len(keys), 3) or not np.all(np.isfinite(xyz)):
            raise ValueError("world replay positions must be finite xyz vectors")
        # cKDTree is part of the frozen formal sidecar.  It enumerates candidate
        # identities only; learned scores never consume coordinates directly.
        from scipy.spatial import cKDTree

        local_pairs = cKDTree(xyz).query_pairs(
            self.maximum_candidate_distance_m, output_type="ndarray"
        )
        local_pairs = np.asarray(local_pairs, dtype=np.int64).reshape(-1, 2)
        left_keys = keys[local_pairs[:, 0]]
        right_keys = keys[local_pairs[:, 1]]
        distances = np.linalg.norm(
            xyz[local_pairs[:, 0]] - xyz[local_pairs[:, 1]], axis=1
        ).astype(np.float32)
        scores = np.empty((3, len(local_pairs)), dtype=np.float32)
        node_scores = np.empty((3, len(keys)), dtype=np.float32)
        event_probabilities = np.empty((3, len(keys), 5), dtype=np.float32)
        for seed in FROZEN_ENSEMBLE_SEEDS:
            row_lookup = self._row_by_key[seed]
            try:
                left_rows = np.asarray([row_lookup[int(key)] for key in left_keys], dtype=np.int64)
                right_rows = np.asarray([row_lookup[int(key)] for key in right_keys], dtype=np.int64)
            except KeyError as exc:
                raise RuntimeError(f"seed{seed} lacks a C09 world sequence") from exc
            node_rows = np.asarray([row_lookup[int(key)] for key in keys], dtype=np.int64)
            node_scores[seed] = self._node_matchability[seed][node_rows]
            event_probabilities[seed] = self._event_probabilities[seed][node_rows]
            observation = self._normalized_observation[seed]
            tokens = self._tokens[seed]
            model = self._models[seed]
            with torch.inference_mode():
                for start in range(0, len(local_pairs), batch_size):
                    stop = min(start + batch_size, len(local_pairs))
                    li, ri = left_rows[start:stop], right_rows[start:stop]
                    base_pair = symmetric_pair_features(observation[li], observation[ri], distances[start:stop])
                    token_pair = exit_token_pair_features(
                        {name: value[li] for name, value in tokens.items()},
                        {name: value[ri] for name, value in tokens.items()},
                    )
                    pair = np.concatenate((base_pair, token_pair), axis=1).astype(np.float32)
                    pair = ((pair - self._pair_mean[seed]) / self._pair_std[seed]).astype(np.float32)
                    output = model(
                        torch.from_numpy(observation[li]).to(self.device),
                        torch.from_numpy(observation[ri]).to(self.device),
                        torch.from_numpy(pair).to(self.device),
                    )
                    scores[seed, start:stop] = output["association_score"].cpu().numpy()
        return FrozenWorldAssociationScores(
            _pair_code(left_keys, right_keys),
            distances,
            scores,
            threshold=self.threshold,
            maximum_candidate_distance_m=self.maximum_candidate_distance_m,
            node_keys=keys,
            node_seed_scores=node_scores,
            node_threshold=FROZEN_NODE_MATCHABILITY_THRESHOLD,
            node_event_probabilities=event_probabilities if self.event_node_gate else None,
            event_node_threshold=FROZEN_EVENT_NODE_THRESHOLD,
        )


__all__ = [
    "FROZEN_ENSEMBLE_SEEDS",
    "FROZEN_ENSEMBLE_THRESHOLD",
    "FROZEN_NODE_MATCHABILITY_THRESHOLD",
    "FROZEN_EVENT_NODE_THRESHOLD",
    "FrozenAssociationDecision",
    "FrozenExitTokenEnsembleRuntime",
    "FrozenWorldAssociationScores",
]
