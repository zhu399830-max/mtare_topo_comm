"""Pre-registered block-level analysis for stochastic single-robot closed loop."""

from __future__ import annotations

from itertools import product
import math
import random
import statistics
from typing import Any, Mapping, Sequence


FAMILIES = ("original_mtare", "m1d_topology", "layered_gt_map_oracle")
METRICS = (
    "coverage_time_auc_m3_s",
    "final_explored_volume_m3",
    "mean_explored_volume_m3",
    "traveling_distance_m",
    "cumulative_point_redundancy",
    "final_volume_per_travel_meter_m2",
    "planner_runtime_p95_sec",
)
PRIMARY_METRIC = "coverage_time_auc_m3_s"


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("sign-flip test requires differences")
    observed = abs(statistics.fmean(values))
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        candidate = abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        extreme += candidate >= observed - 1e-15
        total += 1
    return extreme / total


def paired_bootstrap_ci(
    differences: Sequence[float], *, seed: int = 20260820, samples: int = 10000
) -> tuple[float, float]:
    values = tuple(float(value) for value in differences)
    if not values or samples <= 0:
        raise ValueError("bootstrap requires values and positive sample count")
    rng = random.Random(seed)
    means = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _family_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _dispersion(vectors: Sequence[Sequence[float]]) -> float:
    dimensions = len(vectors[0])
    center = [statistics.fmean(vector[index] for vector in vectors) for index in range(dimensions)]
    return statistics.fmean(math.dist(vector, center) for vector in vectors)


def analyze_stochastic_cases(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(summaries) != 90:
        raise ValueError("stochastic analysis requires exactly 90 case summaries")
    case_ids = [str(item["case"]["case_id"]) for item in summaries]
    if len(set(case_ids)) != 90:
        raise ValueError("case summaries are not unique")
    blocks: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    family_values: dict[str, dict[str, list[float]]] = {
        family: {metric: [] for metric in METRICS} for family in FAMILIES
    }
    for item in summaries:
        case = item["case"]
        metrics = item["metrics"]
        if item.get("status") != "PASS_SINGLE_ROBOT_CASE_V1":
            raise ValueError("analysis cannot include a failed case")
        family = case["method_family"]
        if family not in FAMILIES:
            raise ValueError("unknown method family")
        blocks.setdefault(case["block_id"], {}).setdefault(family, []).append(item)
        for metric in METRICS:
            value = float(metrics[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric: {metric}")
            family_values[family][metric].append(value)
    if len(blocks) != 10:
        raise ValueError("analysis requires ten world/environment blocks")
    block_records: list[dict[str, Any]] = []
    for block_id in sorted(blocks):
        families = blocks[block_id]
        if set(families) != set(FAMILIES) or any(len(families[family]) != 3 for family in FAMILIES):
            raise ValueError(f"incomplete stochastic block: {block_id}")
        record: dict[str, Any] = {"block_id": block_id, "families": {}}
        for family in FAMILIES:
            group = families[family]
            record["families"][family] = {
                "metric_means": {
                    metric: statistics.fmean(float(item["metrics"][metric]) for item in group)
                    for metric in METRICS
                },
                "metric_sample_std": {
                    metric: statistics.stdev(float(item["metrics"][metric]) for item in group)
                    for metric in METRICS
                },
                "final_pose_dispersion_m": _dispersion([item["metrics"]["final_pose_xyz_m"] for item in group]),
                "final_waypoint_dispersion_m": _dispersion([item["metrics"]["final_waypoint_xyz_m"] for item in group]),
            }
        block_records.append(record)

    comparisons: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for metric in METRICS:
        m1d = [record["families"]["m1d_topology"]["metric_means"][metric] for record in block_records]
        baseline = [record["families"]["original_mtare"]["metric_means"][metric] for record in block_records]
        oracle = [record["families"]["layered_gt_map_oracle"]["metric_means"][metric] for record in block_records]
        differences = [left - right for left, right in zip(m1d, baseline)]
        ci = paired_bootstrap_ci(differences)
        pvalue = exact_sign_flip_pvalue(differences)
        raw_pvalues[metric] = pvalue
        baseline_mean = statistics.fmean(baseline)
        comparisons[metric] = {
            "m1d_minus_original_block_differences": differences,
            "mean_difference": statistics.fmean(differences),
            "relative_difference_fraction": statistics.fmean(differences) / baseline_mean if baseline_mean else None,
            "paired_bootstrap_95_ci": list(ci),
            "exact_two_sided_sign_flip_p": pvalue,
            "m1d_minus_oracle_mean_difference": statistics.fmean(left - right for left, right in zip(m1d, oracle)),
        }
    ordered = sorted(raw_pvalues, key=raw_pvalues.get)
    running = 0.0
    for rank, metric in enumerate(ordered):
        adjusted = min(1.0, raw_pvalues[metric] * (len(ordered) - rank))
        running = max(running, adjusted)
        comparisons[metric]["holm_adjusted_p"] = running

    stability = {
        family: {
            "mean_within_block_final_pose_dispersion_m": statistics.fmean(
                record["families"][family]["final_pose_dispersion_m"] for record in block_records
            ),
            "mean_within_block_final_waypoint_dispersion_m": statistics.fmean(
                record["families"][family]["final_waypoint_dispersion_m"] for record in block_records
            ),
            "mean_within_block_auc_sample_std": statistics.fmean(
                record["families"][family]["metric_sample_std"][PRIMARY_METRIC] for record in block_records
            ),
        }
        for family in FAMILIES
    }
    return {
        "schema_version": "mtare_single_robot_stochastic_analysis_v1",
        "case_count": 90,
        "block_count": 10,
        "independent_world_count": 2,
        "primary_metric": PRIMARY_METRIC,
        "family_summaries": {
            family: {metric: _family_summary(values) for metric, values in metrics.items()}
            for family, metrics in family_values.items()
        },
        "block_records": block_records,
        "m1d_comparisons": comparisons,
        "stability": stability,
        "inference_warning": (
            "Execution repeats and checkpoint seeds quantify stochastic/model variation but do not create additional "
            "independent worlds. Confirmatory generalization still requires sealed later-world evaluation."
        ),
    }


__all__ = [
    "FAMILIES",
    "METRICS",
    "PRIMARY_METRIC",
    "analyze_stochastic_cases",
    "exact_sign_flip_pvalue",
    "paired_bootstrap_ci",
]
