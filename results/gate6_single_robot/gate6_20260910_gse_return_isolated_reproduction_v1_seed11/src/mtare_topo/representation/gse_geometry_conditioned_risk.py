"""Leakage-safe features for a causal geometry-conditioned event risk readout.

The module deliberately contains no dataset access and no fitting side effect.
Callers must supply deployment-available predictions aligned to a causal
traversal stream.  Objective identity/world values are accepted only by the
separate sample-weight routine and can never enter the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_CAUSAL_LAGS = (2, 4, 6, 8, 10, 12)
EVENT_NAMES = ("corridor", "junction", "terminal", "turn", "geometry_transition")


@dataclass(frozen=True)
class CausalRiskFeatures:
    values: np.ndarray
    history_available: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        history = np.asarray(self.history_available)
        if (
            values.ndim != 2
            or history.ndim != 2
            or values.shape[0] != history.shape[0]
            or values.shape[1] != len(self.names)
            or not np.all(np.isfinite(values))
            or not np.all((history == 0) | (history == 1))
        ):
            raise ValueError("causal risk feature contract is invalid")
        values.setflags(write=False)
        history.setflags(write=False)


@dataclass(frozen=True)
class WeightedStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("weighted standardizer arrays are invalid")
        mean.setflags(write=False)
        scale.setflags(write=False)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.mean)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("risk readout values do not match the standardizer")
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class GeometryConditionedRiskReadout:
    """Frozen linear five-class readout used by the bounded capacity proof."""

    standardizer: WeightedStandardizer
    coefficient: np.ndarray
    intercept: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient, dtype=np.float64)
        intercept = np.asarray(self.intercept, dtype=np.float64)
        if (
            coefficient.shape != (len(EVENT_NAMES), len(self.feature_names))
            or intercept.shape != (len(EVENT_NAMES),)
            or len(self.standardizer.mean) != len(self.feature_names)
            or not np.all(np.isfinite(coefficient))
            or not np.all(np.isfinite(intercept))
        ):
            raise ValueError("geometry-conditioned risk readout contract is invalid")
        coefficient.setflags(write=False)
        intercept.setflags(write=False)

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        normalized = self.standardizer.transform(values)
        logits = normalized @ self.coefficient.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        return probability


@dataclass(frozen=True)
class LinearRiskFitSummary:
    observations: int
    features: int
    optimizer_iterations: int
    objective: float
    weighted_negative_log_likelihood: float
    l2_penalty: float
    converged: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "observations": self.observations,
            "features": self.features,
            "optimizer_iterations": self.optimizer_iterations,
            "objective": self.objective,
            "weighted_negative_log_likelihood": self.weighted_negative_log_likelihood,
            "l2_penalty": self.l2_penalty,
            "converged": self.converged,
        }


def _aligned_predictions(
    geometry: np.ndarray,
    uncertainty: Iterable[float],
    event_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = np.asarray(geometry, dtype=np.float64)
    uncertainty = np.asarray(tuple(uncertainty), dtype=np.float64)
    probability = np.asarray(event_probability, dtype=np.float64)
    if (
        geometry.ndim != 2
        or geometry.shape[1] != 4
        or uncertainty.shape != (len(geometry),)
        or probability.shape != (len(geometry), 5)
        or not np.all(np.isfinite(geometry))
        or not np.all(np.isfinite(uncertainty))
        or not np.all(np.isfinite(probability))
        or np.any((uncertainty < 0.0) | (uncertainty > 1.0))
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("geometry, uncertainty and event probability must be aligned predictions")
    return geometry, uncertainty, probability


def causal_geometry_risk_features(
    geometry: np.ndarray,
    uncertainty: Iterable[float],
    event_probability: np.ndarray,
    traversal_id: Iterable[str],
    sequence_index: Iterable[int],
    partition: Iterable[int],
    geometry_valid: Iterable[bool],
    *,
    lags: Sequence[int] = DEFAULT_CAUSAL_LAGS,
) -> CausalRiskFeatures:
    """Build fixed multi-lag features using only current and past predictions.

    Every lag contributes signed and absolute deltas for the four predicted
    metric geometry fields plus an availability bit.  Missing history is
    represented by zero deltas and a zero bit; it is never borrowed from a
    different traversal or development partition.  The tail contains current
    structural probability, four conditional structural-class probabilities,
    and current uncertainty.
    """

    geometry, uncertainty, probability = _aligned_predictions(
        geometry, uncertainty, event_probability
    )
    traversal = np.asarray(tuple(traversal_id), dtype=str)
    sequence = np.asarray(tuple(sequence_index), dtype=np.int64)
    split = np.asarray(tuple(partition), dtype=np.int8)
    valid = np.asarray(tuple(geometry_valid), dtype=np.bool_)
    rows = len(geometry)
    if (
        traversal.shape != (rows,)
        or sequence.shape != (rows,)
        or split.shape != (rows,)
        or valid.shape != (rows,)
        or np.any(sequence < 0)
        or np.any(~np.isin(split, (0, 1)))
    ):
        raise ValueError("causal stream arrays must align with the prediction rows")
    lag_tuple = tuple(int(value) for value in lags)
    if (
        not lag_tuple
        or any(value <= 0 for value in lag_tuple)
        or tuple(sorted(set(lag_tuple))) != lag_tuple
    ):
        raise ValueError("causal lags must be unique, positive and strictly increasing")
    lookup = {(str(traversal[row]), int(sequence[row])): row for row in range(rows)}
    if len(lookup) != rows:
        raise ValueError("duplicate traversal/sequence key")

    block = np.zeros((rows, len(lag_tuple), 9), dtype=np.float64)
    history = np.zeros((rows, len(lag_tuple)), dtype=np.uint8)
    for row in range(rows):
        if not valid[row]:
            continue
        for lag_column, lag in enumerate(lag_tuple):
            past = lookup.get((str(traversal[row]), int(sequence[row]) - lag))
            if past is None or not valid[past]:
                continue
            if split[past] != split[row]:
                raise ValueError("causal risk history crosses a development partition")
            delta = geometry[row] - geometry[past]
            block[row, lag_column, :4] = delta
            block[row, lag_column, 4:8] = np.abs(delta)
            block[row, lag_column, 8] = 1.0
            history[row, lag_column] = 1

    structural = 1.0 - probability[:, :1]
    conditional = probability[:, 1:] / np.clip(structural, 1e-12, None)
    # A fully corridor prediction has no conditional evidence by definition.
    conditional[structural[:, 0] <= 1e-12] = 0.0
    values = np.concatenate(
        (
            block.reshape(rows, -1),
            structural,
            conditional,
            uncertainty.reshape(rows, 1),
        ),
        axis=1,
    )
    geometry_names = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    names: list[str] = []
    for lag in lag_tuple:
        names.extend(f"delta_{name}_lag{lag}" for name in geometry_names)
        names.extend(f"abs_delta_{name}_lag{lag}" for name in geometry_names)
        names.append(f"history_available_lag{lag}")
    names.append("structural_probability")
    names.extend(f"conditional_{name}_probability" for name in EVENT_NAMES[1:])
    names.append("uncertainty")
    return CausalRiskFeatures(values=values, history_available=history, names=tuple(names))


def identity_balanced_event_weights(
    event: Iterable[int],
    identity: Iterable[str | None],
    parent_id: Iterable[str],
    fit_mask: Iterable[bool],
) -> np.ndarray:
    """Assign equal class mass and equal independent-unit mass within class.

    Structural rows are balanced by persistent identity.  Corridor rows have
    no structural identity and are balanced by parent world.  Selection rows
    receive exactly zero weight, so they cannot influence fitting.
    """

    event = np.asarray(tuple(event), dtype=np.int64)
    identity = np.asarray(tuple("" if value is None else str(value) for value in identity), dtype=str)
    parent = np.asarray(tuple(parent_id), dtype=str)
    fit = np.asarray(tuple(fit_mask), dtype=np.bool_)
    if (
        event.ndim != 1
        or identity.shape != event.shape
        or parent.shape != event.shape
        or fit.shape != event.shape
        or len(event) == 0
        or np.any((event < 0) | (event >= len(EVENT_NAMES)))
        or np.any(parent == "")
    ):
        raise ValueError("event balancing arrays are invalid or misaligned")
    weights = np.zeros(len(event), dtype=np.float64)
    present_classes = sorted(set(event[fit].tolist()))
    if present_classes != list(range(len(EVENT_NAMES))):
        raise ValueError("fit population must contain all five event classes")
    class_mass = 1.0 / len(present_classes)
    for event_index in present_classes:
        class_rows = np.where(fit & (event == event_index))[0]
        groups = parent[class_rows] if event_index == 0 else identity[class_rows]
        if event_index != 0 and np.any(groups == ""):
            raise ValueError("fit structural event row lacks a persistent identity")
        unique_groups = sorted(set(groups.tolist()))
        group_mass = class_mass / len(unique_groups)
        for group in unique_groups:
            members = class_rows[groups == group]
            weights[members] = group_mass / len(members)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12) or np.any(weights[~fit] != 0.0):
        raise RuntimeError("identity-balanced event weights violate the fit-only contract")
    weights.setflags(write=False)
    return weights


def fit_weighted_standardizer(
    values: np.ndarray,
    sample_weight: Iterable[float],
    *,
    minimum_scale: float = 1e-6,
) -> WeightedStandardizer:
    """Fit deterministic normalization using nonzero fit weights only."""

    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(tuple(sample_weight), dtype=np.float64)
    if (
        values.ndim != 2
        or weight.shape != (len(values),)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weight))
        or np.any(weight < 0.0)
        or not np.isclose(weight.sum(), 1.0, atol=1e-12)
        or not np.isfinite(minimum_scale)
        or minimum_scale <= 0.0
    ):
        raise ValueError("weighted standardization inputs are invalid")
    mean = np.sum(values * weight[:, None], axis=0)
    variance = np.sum(np.square(values - mean) * weight[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), minimum_scale)
    return WeightedStandardizer(mean=mean, scale=scale)


def fit_geometry_conditioned_risk_readout(
    features: CausalRiskFeatures,
    event: Iterable[int],
    sample_weight: Iterable[float],
    *,
    l2_strength: float = 1e-3,
    maximum_iterations: int = 500,
) -> tuple[GeometryConditionedRiskReadout, LinearRiskFitSummary]:
    """Fit a deterministic low-capacity multinomial risk readout.

    Corridor is the fixed reference class, eliminating softmax parameter
    non-identifiability.  Only rows with positive fit weights affect either
    normalization or the L-BFGS objective.  The caller remains responsible for
    producing those weights from development identities only.
    """

    from scipy.optimize import minimize
    from scipy.special import logsumexp

    labels = np.asarray(tuple(event), dtype=np.int64)
    weight = np.asarray(tuple(sample_weight), dtype=np.float64)
    values = np.asarray(features.values, dtype=np.float64)
    if (
        labels.shape != (len(values),)
        or weight.shape != (len(values),)
        or np.any((labels < 0) | (labels >= len(EVENT_NAMES)))
        or np.any(weight < 0.0)
        or not np.all(np.isfinite(weight))
        or not np.isclose(weight.sum(), 1.0, atol=1e-12)
        or not np.isfinite(l2_strength)
        or l2_strength <= 0.0
        or not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or maximum_iterations <= 0
    ):
        raise ValueError("geometry-conditioned risk fit contract is invalid")
    active = weight > 0.0
    if sorted(set(labels[active].tolist())) != list(range(len(EVENT_NAMES))):
        raise ValueError("weighted fit rows must contain every event class")
    standardizer = fit_weighted_standardizer(values, weight)
    design = standardizer.transform(values[active])
    active_label = labels[active]
    active_weight = weight[active]
    feature_count = design.shape[1]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = theta.reshape(len(EVENT_NAMES) - 1, feature_count + 1)
        return matrix[:, :feature_count], matrix[:, feature_count]

    def objective_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient, intercept = unpack(theta)
        noncorridor = design @ coefficient.T + intercept
        logits = np.concatenate((np.zeros((len(design), 1)), noncorridor), axis=1)
        log_normalizer = logsumexp(logits, axis=1)
        probability = np.exp(logits - log_normalizer[:, None])
        negative_log_likelihood = -float(
            np.sum(active_weight * (logits[np.arange(len(design)), active_label] - log_normalizer))
        )
        penalty = 0.5 * l2_strength * float(np.sum(np.square(coefficient)))
        residual = probability
        residual[np.arange(len(design)), active_label] -= 1.0
        residual *= active_weight[:, None]
        gradient_coefficient = residual[:, 1:].T @ design + l2_strength * coefficient
        gradient_intercept = residual[:, 1:].sum(axis=0)
        gradient = np.concatenate(
            (gradient_coefficient, gradient_intercept[:, None]), axis=1
        ).reshape(-1)
        return negative_log_likelihood + penalty, gradient

    initial = np.zeros((len(EVENT_NAMES) - 1) * (feature_count + 1), dtype=np.float64)
    optimized = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maximum_iterations, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        raise RuntimeError(f"geometry-conditioned risk optimization failed: {optimized.message}")
    noncorridor_coefficient, noncorridor_intercept = unpack(optimized.x)
    coefficient = np.zeros((len(EVENT_NAMES), feature_count), dtype=np.float64)
    intercept = np.zeros(len(EVENT_NAMES), dtype=np.float64)
    coefficient[1:] = noncorridor_coefficient
    intercept[1:] = noncorridor_intercept
    readout = GeometryConditionedRiskReadout(
        standardizer=standardizer,
        coefficient=coefficient,
        intercept=intercept,
        feature_names=features.names,
    )
    probability = readout.predict_probability(values[active])
    negative_log_likelihood = -float(
        np.sum(active_weight * np.log(np.clip(probability[np.arange(len(probability)), active_label], 1e-15, None)))
    )
    penalty = 0.5 * l2_strength * float(np.sum(np.square(coefficient)))
    summary = LinearRiskFitSummary(
        observations=int(active.sum()),
        features=feature_count,
        optimizer_iterations=int(optimized.nit),
        objective=negative_log_likelihood + penalty,
        weighted_negative_log_likelihood=negative_log_likelihood,
        l2_penalty=penalty,
        converged=True,
    )
    return readout, summary


__all__ = [
    "CausalRiskFeatures",
    "DEFAULT_CAUSAL_LAGS",
    "EVENT_NAMES",
    "GeometryConditionedRiskReadout",
    "LinearRiskFitSummary",
    "WeightedStandardizer",
    "causal_geometry_risk_features",
    "fit_geometry_conditioned_risk_readout",
    "fit_weighted_standardizer",
    "identity_balanced_event_weights",
]
