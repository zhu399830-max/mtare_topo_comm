"""Frozen all-pairs runtime for Factorized GSE decision-node association."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from mtare_topo.representation.gse_factorized_association import (
    FactorizedAssociationVerifier,
    factorized_pair_features,
)
from mtare_topo.topology.factorized_gse_graph import FactorizedConsensusMetricBackend


TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)


class FactorizedConsensusMetricRuntime:
    """Materialize pair-local scores without consuming graph state or Teacher identity."""

    def __init__(
        self,
        *,
        global_sequence_index: Sequence[int],
        observation_by_seed: Mapping[int, np.ndarray],
        token_outputs_by_seed: Mapping[int, Mapping[str, np.ndarray]],
        geometry_profiles_by_seed: Mapping[int, np.ndarray],
        model_directories: Mapping[int, Path],
        seed_thresholds: Sequence[float],
        votes_required: int = 2,
        maximum_candidate_distance_m: float = 4.0,
        device: str | torch.device = "cpu",
    ) -> None:
        seeds = (0, 1, 2)
        if any(tuple(sorted(values)) != seeds for values in (
            observation_by_seed, token_outputs_by_seed,
            geometry_profiles_by_seed, model_directories,
        )):
            raise ValueError("factorized deployment requires seeds 0/1/2")
        keys = np.asarray(global_sequence_index, dtype=np.int64)
        if keys.ndim != 1 or len(keys) == 0 or len(np.unique(keys)) != len(keys):
            raise ValueError("global sequence identities must be unique")
        thresholds = np.asarray(seed_thresholds, dtype=np.float64)
        if thresholds.shape != (3,) or not np.all(np.isfinite(thresholds)):
            raise ValueError("factorized runtime requires three thresholds")
        self.keys = keys
        self.row_by_key = {int(key): row for row, key in enumerate(keys)}
        self.seed_thresholds = tuple(float(value) for value in thresholds)
        self.votes_required = int(votes_required)
        self.maximum_candidate_distance_m = float(maximum_candidate_distance_m)
        self.device = torch.device(device)
        self.observation: dict[int, np.ndarray] = {}
        self.tokens: dict[int, dict[str, np.ndarray]] = {}
        self.profiles: dict[int, np.ndarray] = {}
        self.mean: dict[int, np.ndarray] = {}
        self.std: dict[int, np.ndarray] = {}
        self.models: dict[int, FactorizedAssociationVerifier] = {}
        for seed in seeds:
            observation = np.asarray(observation_by_seed[seed], dtype=np.float32)
            profile = np.asarray(geometry_profiles_by_seed[seed], dtype=np.float32)
            tokens = {
                name: np.asarray(token_outputs_by_seed[seed][name], dtype=np.float32)
                for name in TOKEN_KEYS
            }
            if (
                observation.shape != (len(keys), 146)
                or profile.shape != (len(keys), 9)
                or any(value.shape[0] != len(keys) for value in tokens.values())
                or not np.all(np.isfinite(observation))
                or not np.all(np.isfinite(profile))
                or any(not np.all(np.isfinite(value)) for value in tokens.values())
            ):
                raise ValueError(f"factorized runtime feature population drift: seed {seed}")
            directory = Path(model_directories[seed]).resolve()
            with np.load(directory / "normalization.npz", allow_pickle=False) as archive:
                mean = archive["mean"].astype(np.float32)
                std = archive["std"].astype(np.float32)
            checkpoint = torch.load(directory / "best.pt", map_location=self.device, weights_only=False)
            if (
                checkpoint.get("schema_version") != "gse_factorized_association_checkpoint_v1"
                or checkpoint.get("seed") != seed
                or checkpoint.get("include_route_geometry") is not True
            ):
                raise RuntimeError(f"factorized runtime checkpoint drift: seed {seed}")
            model = FactorizedAssociationVerifier(include_route_geometry=True).to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.observation[seed] = observation
            self.tokens[seed] = tokens
            self.profiles[seed] = profile
            self.mean[seed] = mean
            self.std[seed] = std
            self.models[seed] = model

    def prepare_world(
        self,
        sequence_keys: Sequence[int],
        xyz_by_key: Mapping[int, Sequence[float]],
        *,
        batch_size: int = 8192,
    ) -> FactorizedConsensusMetricBackend:
        keys = np.asarray(sorted({int(value) for value in sequence_keys}), dtype=np.int64)
        if len(keys) != len(sequence_keys) or any(int(key) not in xyz_by_key for key in keys):
            raise ValueError("world sequence keys must be unique and have positions")
        try:
            rows = np.asarray([self.row_by_key[int(key)] for key in keys], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError("world key is absent from factorized outputs") from exc
        xyz = np.stack([np.asarray(xyz_by_key[int(key)], dtype=np.float64) for key in keys])
        if xyz.shape != (len(keys), 3) or not np.all(np.isfinite(xyz)):
            raise ValueError("world positions must be finite xyz")
        from scipy.spatial import cKDTree
        local_pairs = cKDTree(xyz).query_pairs(
            self.maximum_candidate_distance_m, output_type="ndarray"
        )
        local_pairs = np.asarray(local_pairs, dtype=np.int64).reshape(-1, 2)
        left_keys = keys[local_pairs[:, 0]]
        right_keys = keys[local_pairs[:, 1]]
        left_rows = rows[local_pairs[:, 0]]
        right_rows = rows[local_pairs[:, 1]]
        scores = np.empty((3, len(local_pairs)), dtype=np.float32)
        for seed in (0, 1, 2):
            model = self.models[seed]
            with torch.inference_mode():
                for start in range(0, len(local_pairs), batch_size):
                    stop = min(start + batch_size, len(local_pairs))
                    features = factorized_pair_features(
                        self.observation[seed], self.tokens[seed], self.profiles[seed],
                        left_rows[start:stop], right_rows[start:stop],
                        include_route_geometry=True,
                    )
                    normalized = ((features - self.mean[seed]) / self.std[seed]).astype(np.float32)
                    scores[seed, start:stop] = torch.sigmoid(
                        model(torch.from_numpy(normalized).to(self.device))
                    ).cpu().numpy()
        score_by_pair = {
            (int(left), int(right)): tuple(float(value) for value in scores[:, index])
            for index, (left, right) in enumerate(zip(left_keys, right_keys, strict=True))
        }
        return FactorizedConsensusMetricBackend(
            score_by_pair, seed_thresholds=self.seed_thresholds,
            votes_required=self.votes_required,
            maximum_candidate_distance_m=self.maximum_candidate_distance_m,
        )


__all__ = ["FactorizedConsensusMetricRuntime", "TOKEN_KEYS"]
