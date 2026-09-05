"""Compose corrected V4 cases with immutable baseline evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import statistics
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.stochastic_closed_loop import (
    METRICS,
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
)
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import (
    SOURCE_STATUS,
    analyze_finalized_v2_cases,
)


IDENTITY_FIELDS = (
    "case_id",
    "block_id",
    "method_family",
    "method_id",
    "world",
    "environment_seed",
    "execution_repeat",
    "checkpoint_seed",
    "runtime_sec",
)


def _by_id(values: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    result = {str(item.get("case", {}).get("case_id")): item for item in values}
    if len(result) != len(values) or "None" in result:
        raise ValueError(f"{name} case identities are missing or duplicated")
    return result


def _validate_finalized(values: Sequence[Mapping[str, Any]], *, name: str) -> None:
    for item in values:
        if item.get("status") != SOURCE_STATUS:
            raise ValueError(f"{name} contains a non-finalized V2 case")


def compose_corrected_cases(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace only the source M1D cases with matched V4 summaries."""

    if len(source_summaries) != 90 or len(corrected_summaries) != 30:
        raise ValueError("corrected composition requires 90 source and 30 corrected cases")
    _validate_finalized(source_summaries, name="source")
    _validate_finalized(corrected_summaries, name="corrected")
    source = _by_id(source_summaries, name="source")
    corrected = _by_id(corrected_summaries, name="corrected")
    source_m1d_ids = {
        case_id for case_id, item in source.items()
        if item["case"]["method_family"] == "m1d_topology"
    }
    if set(corrected) != source_m1d_ids:
        raise ValueError("corrected cases do not exactly match source M1D identities")
    for case_id, item in corrected.items():
        source_case = source[case_id]["case"]
        corrected_case = item["case"]
        for field in IDENTITY_FIELDS:
            if corrected_case.get(field) != source_case.get(field):
                raise ValueError(f"corrected case identity drift: {case_id}: {field}")

    combined = [
        deepcopy(dict(corrected[item["case"]["case_id"]]))
        if item["case"]["method_family"] == "m1d_topology"
        else deepcopy(dict(item))
        for item in source_summaries
    ]
    quotas = Counter(item["case"]["method_family"] for item in combined)
    if quotas != Counter({"original_mtare": 30, "m1d_topology": 30, "layered_gt_map_oracle": 30}):
        raise ValueError("combined family quotas are not 30/30/30")
    provenance = {
        "schema_version": "corrected_stochastic_composition_v1",
        "source_case_count": 90,
        "corrected_case_count": 30,
        "combined_case_count": 90,
        "replaced_family": "m1d_topology",
        "replaced_case_count": 30,
        "reused_original_mtare_case_count": 30,
        "reused_oracle_diagnostic_case_count": 30,
        "source_mutation_permitted": False,
        "identity_fields": list(IDENTITY_FIELDS),
    }
    return combined, provenance


def _corrective_effects(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old = _by_id(source_summaries, name="source")
    corrected = _by_id(corrected_summaries, name="corrected")
    effects: dict[str, Any] = {}
    for metric in METRICS:
        by_block: dict[str, list[float]] = defaultdict(list)
        for case_id, item in corrected.items():
            new_value = float(item["metrics"][metric])
            old_value = float(old[case_id]["metrics"][metric])
            if not math.isfinite(new_value) or not math.isfinite(old_value):
                raise ValueError(f"non-finite corrective metric: {metric}")
            by_block[item["case"]["block_id"]].append(new_value - old_value)
        if len(by_block) != 10 or any(len(values) != 3 for values in by_block.values()):
            raise ValueError("corrective effects require ten blocks of three checkpoint pairs")
        block_differences = [statistics.fmean(by_block[key]) for key in sorted(by_block)]
        effects[metric] = {
            "v4_minus_v9_block_differences": block_differences,
            "mean_difference": statistics.fmean(block_differences),
            "paired_bootstrap_95_ci": list(paired_bootstrap_ci(block_differences)),
            "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(block_differences),
        }
    return {
        "schema_version": "corrected_v4_vs_v9_effects_v1",
        "paired_case_count": 30,
        "block_count": 10,
        "checkpoint_pairs_per_block": 3,
        "metrics": effects,
        "inference_unit": "world-by-environment block after averaging three checkpoint pairs",
    }


def analyze_corrected_stochastic_cases(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined, provenance = compose_corrected_cases(source_summaries, corrected_summaries)
    return {
        "schema_version": "corrected_stochastic_comparison_v1",
        "composition": provenance,
        "corrected_main_analysis": analyze_finalized_v2_cases(combined),
        "v4_vs_defective_v9": _corrective_effects(source_summaries, corrected_summaries),
    }


__all__ = [
    "IDENTITY_FIELDS",
    "analyze_corrected_stochastic_cases",
    "compose_corrected_cases",
]
