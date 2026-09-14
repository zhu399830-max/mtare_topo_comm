"""V5-labelled corrected comparison over the frozen V1 statistical core."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.corrected_stochastic_comparison import (
    analyze_corrected_stochastic_cases,
)
from mtare_topo.evaluation.stochastic_closed_loop import METRICS


def analyze_v5_corrected_stochastic_cases(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the unchanged composition/statistics while labelling V5 truthfully."""

    base = analyze_corrected_stochastic_cases(source_summaries, corrected_summaries)
    effects = base["v4_vs_defective_v9"]
    renamed_metrics = {}
    for metric, value in effects["metrics"].items():
        copied = dict(value)
        copied["v5_minus_defective_v9_block_differences"] = copied.pop(
            "v4_minus_v9_block_differences"
        )
        defective_values = [
            float(item["metrics"][metric])
            for item in source_summaries
            if item["case"]["method_family"] == "m1d_topology"
        ]
        if len(defective_values) != 30:
            raise ValueError("V5 comparison requires 30 defective V9 baseline values")
        defective_mean = statistics.fmean(defective_values)
        if defective_mean == 0.0:
            raise ValueError(f"V5 comparison has zero defective baseline mean: {metric}")
        copied["defective_v9_mean"] = defective_mean
        copied["relative_mean_difference_fraction"] = (
            float(copied["mean_difference"]) / defective_mean
        )
        copied["relative_bootstrap_95_ci"] = [
            float(endpoint) / defective_mean
            for endpoint in copied["paired_bootstrap_95_ci"]
        ]
        renamed_metrics[metric] = copied
    ordered = sorted(
        METRICS,
        key=lambda metric: float(
            renamed_metrics[metric]["exact_two_sided_sign_flip_p"]
        ),
    )
    running = 0.0
    for rank, metric in enumerate(ordered):
        raw = float(renamed_metrics[metric]["exact_two_sided_sign_flip_p"])
        adjusted = min(1.0, raw * (len(ordered) - rank))
        running = max(running, adjusted)
        renamed_metrics[metric]["holm_adjusted_p"] = running
    composition = dict(base["composition"])
    composition["schema_version"] = "corrected_v5_stochastic_composition_v1"
    composition["corrected_method"] = "V5 verified re-anchor plus frontier execution feedback"
    return {
        "schema_version": "corrected_v5_stochastic_comparison_v1",
        "composition": composition,
        "corrected_main_analysis": base["corrected_main_analysis"],
        "v5_vs_defective_v9": {
            **{key: value for key, value in effects.items() if key != "metrics"},
            "schema_version": "corrected_v5_vs_defective_v9_effects_v1",
            "metrics": renamed_metrics,
        },
    }


__all__ = ["analyze_v5_corrected_stochastic_cases"]
