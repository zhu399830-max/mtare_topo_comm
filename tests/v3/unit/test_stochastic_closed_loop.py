from __future__ import annotations

import math

import pytest

from mtare_topo.evaluation.stochastic_closed_loop import (
    METRICS,
    analyze_stochastic_cases,
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
)


def synthetic_summaries():
    values = []
    families = ("original_mtare", "m1d_topology", "layered_gt_map_oracle")
    offsets = {"original_mtare": 0.0, "m1d_topology": 10.0, "layered_gt_map_oracle": 20.0}
    for block in range(10):
        for family in families:
            for replicate in range(3):
                base = 100.0 + block + offsets[family] + replicate
                metrics = {metric: base + index for index, metric in enumerate(METRICS)}
                metrics.update(
                    final_pose_xyz_m=[float(block), float(replicate), offsets[family]],
                    final_waypoint_xyz_m=[float(block), float(replicate), offsets[family] + 1.0],
                )
                values.append(
                    {
                        "status": "PASS_SINGLE_ROBOT_CASE_V1",
                        "case": {
                            "case_id": f"{block}_{family}_{replicate}",
                            "block_id": f"block_{block}",
                            "method_family": family,
                        },
                        "metrics": metrics,
                    }
                )
    return values


def test_exact_sign_flip_and_bootstrap_are_deterministic() -> None:
    differences = [1.0] * 10
    assert exact_sign_flip_pvalue(differences) == pytest.approx(2.0 / 1024.0)
    assert paired_bootstrap_ci(differences) == (1.0, 1.0)
    assert paired_bootstrap_ci(differences) == paired_bootstrap_ci(differences)


def test_stochastic_analysis_preserves_blocks_and_family_sample_counts() -> None:
    result = analyze_stochastic_cases(synthetic_summaries())
    assert result["case_count"] == 90
    assert result["block_count"] == 10
    assert result["independent_world_count"] == 2
    for family in ("original_mtare", "m1d_topology", "layered_gt_map_oracle"):
        for metric in METRICS:
            assert result["family_summaries"][family][metric]["n"] == 30
    for metric in METRICS:
        comparison = result["m1d_comparisons"][metric]
        assert comparison["mean_difference"] == pytest.approx(10.0)
        assert comparison["paired_bootstrap_95_ci"] == pytest.approx([10.0, 10.0])
        assert comparison["exact_two_sided_sign_flip_p"] == pytest.approx(2.0 / 1024.0)
        assert math.isfinite(comparison["holm_adjusted_p"])


def test_stochastic_analysis_rejects_missing_or_duplicate_cases() -> None:
    summaries = synthetic_summaries()
    with pytest.raises(ValueError, match="exactly 90"):
        analyze_stochastic_cases(summaries[:-1])
    summaries[-1]["case"]["case_id"] = summaries[0]["case"]["case_id"]
    with pytest.raises(ValueError, match="not unique"):
        analyze_stochastic_cases(summaries)
