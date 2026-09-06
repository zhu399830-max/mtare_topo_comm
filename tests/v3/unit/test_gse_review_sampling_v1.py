"""Synthetic metadata only; never materialize a real dataset or review run."""
from collections import Counter
from dataclasses import replace
import random

import pytest

from mtare_topo.data.gse_review_sampling_v1 import (
    CandidateSegment, ParentMetadata, ParentPopulation, VariantSegmentMetadata,
    STRATA, partition_parents, sample_review_segments,
)

VARIANTS = ("synthetic_round", "synthetic_mixed", "synthetic_rect")


def population():
    return ParentPopulation(tuple(
        ParentMetadata(f"S{i:02}_synthetic_{cohort}", cohort)
        for cohort, count in (("C01", 4), ("C02", 4), ("C07", 10)) for i in range(1, count + 1)
    ), "a" * 64, True)


def candidate(parent, stratum="junction", index=0, *, structure=None, traversal="route0", frame_stride=1):
    start = index * 30
    records = []
    for offset, name in enumerate(VARIANTS):
        rows = tuple(start + j + offset * 20000 for j in range(21))
        frames = tuple(tuple((start + j + h + offset * 20000) * frame_stride for h in range(5)) for j in range(21))
        records.append(VariantSegmentMetadata(
            variant=name, task=parent + "__" + name,
            source_sequence_ids=tuple(start + j for j in range(21)), sequence_rows=rows,
            frame_rows=frames, frame_traversal_ids=tuple((traversal,) * 5 for _ in range(21)),
            decision_frame_rows=tuple(row[-1] for row in frames),
            route_arc_m=tuple(float(start + j) * (1 + offset * .25) for j in range(21)),
            source_sequence_count=1000000, source_frame_count=1000000,
        ))
    return CandidateSegment(parent, structure or f"structure{index}", stratum, traversal, tuple(records))


def inventory(pop=None):
    pop = pop or population()
    return tuple(candidate(parent.parent_id, stratum, 2 * group + i)
                 for parent in pop.parents for group, stratum in enumerate(STRATA) for i in range(2))


def run(candidates=None, pop=None, variants=VARIANTS):
    return sample_review_segments(pop or population(), inventory() if candidates is None else candidates, variant_names=variants)


def test_exact_quota_unique_frames_sequence_counts_and_actual_spacing():
    result = run()
    assert result.complete and not result.deficits
    assert len(result.selected) == 48
    counts = Counter((s.split, s.candidate.stratum) for s in result.selected)
    assert counts == Counter({(split, stratum): n for split, n in (("fit", 8), ("calibration", 2), ("development", 2)) for stratum in STRATA})
    meta = result.metadata["selected_counts"]
    assert meta["decision_observation_references"] == meta["unique_variant_sequences"] == 3024
    assert meta["unique_logical_sequences"] == 1008
    assert meta["raw_frame_references"] == 15120
    assert meta["unique_raw_frames"] == 3600  # 21 overlapping histories -> 25 frames, not105
    assert meta["independent_structures"] == 48 and meta["duration_s"] is None
    assert sorted(set(x["span_m"] for x in meta["spacing"])) == [20., 25., 30.]
    assert result.metadata["continuity_requires_source_audit"]
    assert not result.metadata["real_data_verified"] and not result.metadata["human_review_complete"]


def test_order_independent_parent_candidate_and_variant_order():
    pop = population()
    candidates = list(inventory(pop)); random.Random(555).shuffle(candidates)
    candidates = tuple(replace(c, variants=tuple(reversed(c.variants))) for c in candidates)
    changed = run(candidates, replace(pop, parents=tuple(reversed(pop.parents))), tuple(reversed(VARIANTS)))
    assert changed == run()


def test_parent_split_is_based_on_complete_population_not_candidate_subset():
    pop = population(); split = partition_parents(pop)
    chosen_parent = next(p for p, s in split.items() if s == "development")
    partial = run((candidate(chosen_parent),), pop)
    assert partial.selected[0].split == "development"
    assert partial.metadata["parent_split"] == split
    assert Counter(split.values()) == {"fit": 8, "calibration": 5, "development": 5}
    for parent in pop.parents:
        if parent.cohort != "C07": assert split[parent.parent_id] == "fit"


def test_round_robin_uses_distinct_parents_before_second_structure():
    result = run()
    for stratum in STRATA:
        chosen = [s for s in result.selected if s.split == "fit" and s.candidate.stratum == stratum]
        assert len({s.candidate.parent_id for s in chosen}) == 8


def test_same_structure_across_traversals_does_not_fill_quota():
    parent = population().parents[0].parent_id
    first = candidate(parent, structure="same")
    revisit = candidate(parent, index=50, structure="same", traversal="route1")
    result = run((first, revisit))
    assert len(result.selected) == 1
    assert result.metadata["removed_repeat_structure_candidates"] == 1
    deficit = next(d for d in result.deficits if d["split"] == "fit" and d["stratum"] == "junction")
    assert deficit["available_independent_structures"] == 1 and deficit["missing"] == 7
    assert result == run((revisit, first))


def test_no_candidates_reports_all_twelve_deficits_without_fake_data():
    result = run(())
    assert not result.complete and result.selected == ()
    assert len(result.deficits) == 12 and sum(d["missing"] for d in result.deficits) == 48
    assert result.metadata["selected_counts"]["unique_raw_frames"] == 0


def test_nonunit_frame_step_is_reported_not_invented_as_clock_or_bad_continuity():
    result = run((candidate(population().parents[0].parent_id, frame_stride=3),))
    assert result.metadata["selected_counts"]["unique_raw_frames"] == 75
    assert result.metadata["selected_counts"]["duration_s"] is None
    assert all(x == 3 for x in result.metadata["selected_counts"]["spacing"][0]["decision_frame_index_gaps"])


def test_selection_digest_binds_frames_not_only_logical_sequence_ids():
    parent = population().parents[0].parent_id
    a, b = run((candidate(parent),)), run((candidate(parent, frame_stride=2),))
    assert a.metadata["selection_sha256"] != b.metadata["selection_sha256"]


def test_raw_frame_cannot_belong_to_two_traversals():
    parent = population().parents[0].parent_id
    a = candidate(parent)
    b = candidate(parent, index=1, traversal="other")
    b = replace(b, variants=tuple(replace(v, frame_rows=w.frame_rows, decision_frame_rows=w.decision_frame_rows)
                                 for v, w in zip(b.variants, a.variants)))
    with pytest.raises(ValueError, match="same raw frame"): run((a, b))


@pytest.mark.parametrize("fault", ["complete", "digest", "duplicate", "odd_c07", "testworld", "cohort", "parent_type"])
def test_parent_population_validation(fault):
    p = population()
    if fault == "complete": p = replace(p, complete=1)
    elif fault == "digest": p = replace(p, inventory_sha256="not a seal")
    elif fault == "duplicate": p = replace(p, parents=p.parents + (p.parents[0],))
    elif fault == "odd_c07": p = replace(p, parents=p.parents[:-1])
    elif fault == "testworld": p = replace(p, parents=p.parents + (ParentMetadata("S01_synthetic_C08", "C08"),))
    elif fault == "cohort": p = replace(p, parents=(replace(p.parents[0], cohort="C03"),) + p.parents[1:])
    else: p = replace(p, parents=tuple({"parent_id": x.parent_id, "cohort": x.cohort} for x in p.parents))
    with pytest.raises(ValueError): partition_parents(p)


@pytest.mark.parametrize("fault", ["dict", "duplicate", "missing_variant", "variant_identity", "source_id_float",
    "source_id_bool", "noncontiguous", "frame_order", "future_frame", "cross_traversal", "bounds", "bound_bool",
    "arc_nan", "arc_int", "arc_reversed", "parent", "stratum", "conflicting_strata", "task", "five_frames"])
def test_candidate_contract_failures(fault):
    c = candidate(population().parents[0].parent_id); v = c.variants[0]; cs = None
    if fault == "dict": cs = (c.__dict__,)
    elif fault == "duplicate": cs = (c, c)
    elif fault == "missing_variant": c = replace(c, variants=c.variants[:2])
    elif fault == "variant_identity": v = replace(v, source_sequence_ids=tuple(100 + j for j in range(21)))
    elif fault == "source_id_float": v = replace(v, source_sequence_ids=(0.,) + v.source_sequence_ids[1:])
    elif fault == "source_id_bool": v = replace(v, source_sequence_ids=(False,) + v.source_sequence_ids[1:])
    elif fault == "noncontiguous": v = replace(v, sequence_rows=tuple(i * 2 for i in range(21)))
    elif fault == "frame_order": v = replace(v, frame_rows=(tuple(reversed(v.frame_rows[0])),) + v.frame_rows[1:])
    elif fault == "future_frame": v = replace(v, decision_frame_rows=tuple(f - 1 for f in v.decision_frame_rows))
    elif fault == "cross_traversal": v = replace(v, frame_traversal_ids=(("other",) * 5,) + v.frame_traversal_ids[1:])
    elif fault == "bounds": v = replace(v, source_frame_count=20)
    elif fault == "bound_bool": v = replace(v, source_sequence_count=True)
    elif fault == "arc_nan": v = replace(v, route_arc_m=(float("nan"),) + v.route_arc_m[1:])
    elif fault == "arc_int": v = replace(v, route_arc_m=tuple(range(21)))
    elif fault == "arc_reversed": v = replace(v, route_arc_m=tuple(reversed(v.route_arc_m)))
    elif fault == "parent": c = replace(c, parent_id="S01_synthetic_C08")
    elif fault == "stratum": c = replace(c, stratum="model_high_score")
    elif fault == "conflicting_strata": cs = (c, candidate(c.parent_id, "terminal", 50, structure=c.independent_structure_id))
    elif fault == "task": v = replace(v, task="S01_synthetic_C08__" + v.variant)
    else: v = replace(v, frame_rows=(v.frame_rows[0][:4],) + v.frame_rows[1:])
    if v is not c.variants[0] and len(c.variants) == 3:
        c = replace(c, variants=(v,) + c.variants[1:])
    with pytest.raises(ValueError): run(cs if cs is not None else (c,))


def test_cross_candidate_metadata_identity_drift_is_rejected():
    parent = population().parents[0].parent_id
    a = candidate(parent)
    b = replace(a, independent_structure_id="another")
    v = b.variants[0]
    b = replace(b, variants=(replace(v, route_arc_m=tuple(x + .1 for x in v.route_arc_m)),) + b.variants[1:])
    with pytest.raises(ValueError, match="inconsistent metadata"): run((a, b))


@pytest.mark.parametrize("variants", [("a", "a", "b"), ("a", "b"), ("a", "b", {}), ["a", "b", "c"]])
def test_variant_inventory_is_explicit_and_strict(variants):
    with pytest.raises(ValueError): run((), variants=variants)
