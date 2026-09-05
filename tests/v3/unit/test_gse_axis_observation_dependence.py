"""Cached-output correspondence tests on exact synthetic populations only."""
import copy
import importlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from mtare_topo.evaluation.gse_axis_observation_dependence import (
    same_parent_derangement, query_redundancy, score_observation, summarize, compare,
)


def _manifest():
    return [{"task": f"S{parent:02d}_fixture_C01__c1_mixed", "row_index": row}
            for parent in range(1, 11) for row in range(18)]


def _axes(y=0.):
    line = np.array([[-1., y, 0.], [0., y, 0.], [1., y, 0.]], dtype=np.float64)
    return np.repeat(line[None], 32, axis=0)


def _score(prediction, teacher_axis):
    return score_observation(prediction, torch.from_numpy(teacher_axis[None, None].copy()),
                             torch.ones(1, 1, dtype=torch.bool))


def test_fixed_nine_step_mapping_has_no_self_pair_and_never_crosses_parent():
    manifest = _manifest()
    mapping = same_parent_derangement(manifest)
    assert mapping == same_parent_derangement(copy.deepcopy(manifest))
    assert sorted(mapping) == list(range(180))
    for index, source in enumerate(mapping):
        assert source != index
        assert manifest[source]["task"] == manifest[index]["task"]
        assert source // 18 == index // 18
        assert source % 18 == (index % 18 + 9) % 18
        assert mapping[source] == index


def test_derangement_uses_frozen_manifest_order_not_numerical_row_identity():
    manifest = _manifest()
    for index, row in enumerate(manifest):
        row["row_index"] = 1000 - index
    interleaved = [manifest[parent * 18 + row] for row in range(18) for parent in range(10)]
    mapping = same_parent_derangement(interleaved)
    for index, source in enumerate(mapping):
        assert interleaved[index]["task"] == interleaved[source]["task"]
        assert source == (index + 90) % 180


@pytest.mark.parametrize("issue", ["179", "181", "duplicate", "eleven_parents", "unbalanced"])
def test_invalid_180_population_is_rejected(issue):
    manifest = _manifest()
    if issue == "179":
        manifest.pop()
    elif issue == "181":
        manifest.append({"task": manifest[0]["task"], "row_index": 999})
    elif issue == "duplicate":
        manifest[1] = copy.deepcopy(manifest[0])
    elif issue == "eleven_parents":
        manifest[0]["task"] = "S11_fixture_C01__c1_mixed"
    else:
        manifest[0]["task"] = manifest[18]["task"]
        manifest[0]["row_index"] = 999
    with pytest.raises(ValueError):
        same_parent_derangement(manifest)


def test_redundancy_detects_exact_reverse_duplicates_without_direction_threshold():
    axes = np.stack([_axes(float(i))[0] for i in range(32)])
    axes[31] = axes[0, ::-1]
    result = query_redundancy(axes)
    assert result["exact_duplicate_unordered_pairs"] == 1
    assert result["nearest_other_query_coordinate_mae_m"][0] == 0
    assert result["nearest_other_query_coordinate_mae_m"][31] == 0
    assert result["nearest_other_query_coordinate_mae_m"][10] == pytest.approx(1 / 3)
    order = np.random.default_rng(18).permutation(32)
    changed = query_redundancy(axes[order, ::-1])
    assert changed["exact_duplicate_unordered_pairs"] == 1
    np.testing.assert_array_equal(changed["nearest_other_query_coordinate_mae_m"],
                                  np.array(result["nearest_other_query_coordinate_mae_m"])[order])


def test_identical32_query_bank_has496_unordered_duplicate_pairs():
    result = query_redundancy(_axes())
    assert result["exact_duplicate_unordered_pairs"] == 32 * 31 // 2
    assert result["nearest_other_query_coordinate_mae_m"] == [0.] * 32


@pytest.mark.parametrize("issue", ["nan", "shape"])
def test_invalid_candidate_bank_rejected(issue):
    axes = _axes()
    if issue == "nan":
        axes[0, 0, 0] = np.nan
    else:
        axes = axes[:31]
    with pytest.raises(ValueError):
        query_redundancy(axes)


def test_per_observation_perfect_axes_lose_fit_under_fixed_wrong_correspondence():
    manifest = _manifest()
    mapping = same_parent_derangement(manifest)
    banks = [_axes(float(row["row_index"])) for row in manifest]
    correct = [_score(bank, bank[0]) for bank in banks]
    wrong = [_score(banks[source], banks[index][0]) for index, source in enumerate(mapping)]
    correct_summary, wrong_summary = summarize(correct, manifest), summarize(wrong, manifest)
    assert correct_summary["macro"]["coordinate_mae_m"] == 0
    assert wrong_summary["macro"]["coordinate_mae_m"] == 3
    assert wrong_summary["macro"]["point_mean_euclidean_m"] == 9
    result = compare(correct_summary, wrong_summary)
    assert result["coordinate_shuffled_minus_correct_m"] == 3
    assert result["parents_correct_better"] == 10
    assert result["parents_tied"] == result["parents_correct_worse"] == 0
    assert result["scientific_gate_pass"] is False


def test_constant32_query_bank_cannot_gain_from_correct_observation_correspondence():
    manifest = _manifest()
    mapping = same_parent_derangement(manifest)
    predictions = [_axes() for _ in manifest]
    teachers = [_axes(float(row["row_index"]))[0] for row in manifest]
    correct = [_score(predictions[index], teacher) for index, teacher in enumerate(teachers)]
    wrong = [_score(predictions[mapping[index]], teacher) for index, teacher in enumerate(teachers)]
    first, second = summarize(correct, manifest), summarize(wrong, manifest)
    assert first == second
    result = compare(first, second)
    assert result["coordinate_shuffled_minus_correct_m"] == 0
    assert result["parents_tied"] == 10 and result["parents_correct_better"] == 0
    assert result["scientific_gate_pass"] is False


def test_degenerate_teacher_retains_fit_and_counts_unknown_direction_not_zero():
    axes = np.zeros((32, 3, 3), dtype=np.float64)
    result = _score(axes, axes[0])
    assert result["fit"]["n_targets"] == 1
    assert result["fit"]["coordinate_mae_m"] == 0
    assert len(result["layout"]) == 1
    assert result["layout_mean"]["undirected_direction_error_deg"] is None
    assert result["layout_mean"]["transverse_rms_m"] is None
    assert result["layout_counts"]["undirected_direction_error_deg"] == {"resolved": 0, "unresolved": 1}
    assert result["layout_counts"]["polyline_symmetric_m"] == {"resolved": 1, "unresolved": 0}
    summary = summarize([result] * 180, _manifest())["macro"]
    assert summary["matched_targets"] == 180
    assert summary["layout"]["undirected_direction_error_deg"]["mean"] is None
    assert summary["layout"]["undirected_direction_error_deg"]["fragments_unresolved"] == 180


def test_query_histogram_and_surplus_are_reported_without_false_positive_precision():
    result = _score(_axes(), _axes()[0])
    summary = summarize([result] * 180, _manifest())
    macro = summary["macro"]
    assert len(summary["parents"]) == 10
    assert macro["matched_targets"] == 180
    assert macro["surplus_queries"] == 180 * 31
    assert macro["query_match_counts"] == [180] + [0] * 31
    assert macro["query_use_effective_count"] == 1
    assert macro["exact_duplicate_unordered_pairs"] == 180 * 496
    assert "precision" not in macro and "detected" not in macro


def test_comparison_rejects_parent_key_mismatch():
    row = _score(_axes(), _axes()[0])
    first = summarize([row] * 180, _manifest())
    second = copy.deepcopy(first)
    second["parents"].pop(next(iter(second["parents"])))
    with pytest.raises(ValueError):
        compare(first, second)


def test_plot_summary_writes_real_svg_with_undefined_direction_means(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_axis_observation_dependence_v1")
    degenerate = np.zeros((32, 3, 3), dtype=np.float64)
    manifest = _manifest()
    row = _score(degenerate, degenerate[0])
    summary = summarize([row] * len(manifest), manifest)
    assert summary["macro"]["layout"]["undirected_direction_error_deg"]["mean"] is None
    assert summary["macro"]["layout"]["transverse_rms_m"]["mean"] is None
    results = {method: {"correct": copy.deepcopy(summary), "shuffled": copy.deepcopy(summary)}
               for method in runner.METHODS}
    before = copy.deepcopy(results)
    (tmp_path / "previews").mkdir()
    runner.plot_summary(tmp_path, results)
    output = tmp_path / "previews/correspondence_comparison.svg"
    assert output.is_file() and output.stat().st_size > 1000
    root = ET.parse(output).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.findall(".//{http://www.w3.org/2000/svg}path")
    assert results == before, "Plotting must not replace UNKNOWN scientific values or parent records"
