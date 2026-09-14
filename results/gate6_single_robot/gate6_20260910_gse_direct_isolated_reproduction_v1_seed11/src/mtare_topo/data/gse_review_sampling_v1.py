"""Pure metadata selection for the 48-segment human review pilot.

No I/O, model scores, inferred identities or labels. The caller must establish
complete parent inventory and provenance; validation cannot prove metadata true.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re

SEED = 20260906
STRATA = ("junction", "terminal", "corridor", "alias")
QUOTAS = {"fit": 8, "calibration": 2, "development": 2}
DECISIONS = 21
HISTORY = 5


@dataclass(frozen=True)
class ParentMetadata:
    parent_id: str
    cohort: str


@dataclass(frozen=True)
class ParentPopulation:
    parents: tuple[ParentMetadata, ...]
    inventory_sha256: str
    complete: bool


@dataclass(frozen=True)
class VariantSegmentMetadata:
    variant: str
    task: str
    source_sequence_ids: tuple[int, ...]
    sequence_rows: tuple[int, ...]
    frame_rows: tuple[tuple[int, ...], ...]
    frame_traversal_ids: tuple[tuple[str, ...], ...]
    decision_frame_rows: tuple[int, ...]
    route_arc_m: tuple[float, ...]
    source_sequence_count: int
    source_frame_count: int


@dataclass(frozen=True)
class CandidateSegment:
    parent_id: str
    independent_structure_id: str
    stratum: str
    traversal_id: str
    variants: tuple[VariantSegmentMetadata, ...]


@dataclass(frozen=True)
class SelectedSegment:
    split: str
    candidate: CandidateSegment


@dataclass(frozen=True)
class SamplingResult:
    complete: bool
    selected: tuple[SelectedSegment, ...]
    deficits: tuple[dict, ...]
    metadata: dict


def _name(value):
    return type(value) is str and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value))


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _hash(*parts):
    return hashlib.sha256(json.dumps([SEED, *parts], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def partition_parents(population):
    """Split ALL supplied C07 parents, never just parents with candidates."""
    if type(population) is not ParentPopulation or population.complete is not True:
        raise ValueError("explicit complete ParentPopulation required")
    if not isinstance(population.inventory_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", population.inventory_sha256):
        raise ValueError("parent inventory provenance SHA-256 required")
    if type(population.parents) is not tuple or not population.parents:
        raise ValueError("nonempty strict parent tuple required")
    mapping = {}
    for parent in population.parents:
        if type(parent) is not ParentMetadata or not _name(parent.parent_id):
            raise ValueError("strict parent identity required")
        if parent.cohort not in tuple(f"C{i:02}" for i in range(1, 8)) or not parent.parent_id.endswith("_" + parent.cohort):
            raise ValueError("C01-C07 parent cohort identity required; C08+ forbidden")
        if parent.parent_id in mapping:
            raise ValueError("duplicate parent population identity")
        mapping[parent.parent_id] = "fit" if parent.cohort != "C07" else None
    c07 = sorted((p for p, split in mapping.items() if split is None), key=lambda p: (_hash("parent", p), p))
    if not c07 or len(c07) % 2:
        raise ValueError("complete C07 population must have positive even size for exact halves")
    half = len(c07) // 2
    mapping.update({p: "calibration" if i < half else "development" for i, p in enumerate(c07)})
    return dict(sorted(mapping.items()))


def _validate_candidates(candidates, mapping, variants):
    if type(variants) is not tuple or len(variants) != 3 or not all(_name(v) for v in variants) or len(set(variants)) != 3:
        raise ValueError("exact three distinct explicit variant names required")
    if type(candidates) is not tuple:
        raise ValueError("strict candidate tuple required")
    seen, structure_strata, sequence_identity, row_identity, task_bounds, frame_identity = set(), {}, {}, {}, {}, {}
    for candidate in candidates:
        if type(candidate) is not CandidateSegment:
            raise ValueError("strict CandidateSegment required; no score dictionaries")
        if not _name(candidate.parent_id) or candidate.parent_id not in mapping:
            raise ValueError("candidate parent absent from complete permitted inventory")
        if not _name(candidate.independent_structure_id) or not _name(candidate.traversal_id) or candidate.stratum not in STRATA:
            raise ValueError("explicit structure/traversal/stratum required")
        structure = (candidate.parent_id, candidate.independent_structure_id)
        if structure in structure_strata and structure_strata[structure] != candidate.stratum:
            raise ValueError("same independent structure has conflicting strata")
        structure_strata[structure] = candidate.stratum
        if type(candidate.variants) is not tuple or len(candidate.variants) != 3:
            raise ValueError("three variant records required, never fill a missing variant")
        found = {}
        shared = None
        for variant in candidate.variants:
            if type(variant) is not VariantSegmentMetadata or variant.variant not in variants or variant.variant in found:
                raise ValueError("strict unique variant records required")
            found[variant.variant] = variant
            if variant.task != candidate.parent_id + "__" + variant.variant:
                raise ValueError("task does not bind parent and variant")
            if not _integer(variant.source_sequence_count, 1) or not _integer(variant.source_frame_count, 1):
                raise ValueError("strict positive source bounds required")
            bounds = (variant.source_sequence_count, variant.source_frame_count)
            if variant.task in task_bounds and task_bounds[variant.task] != bounds:
                raise ValueError("same task source bounds drift")
            task_bounds[variant.task] = bounds
            for field in (variant.source_sequence_ids, variant.sequence_rows, variant.frame_rows,
                          variant.frame_traversal_ids, variant.decision_frame_rows, variant.route_arc_m):
                if type(field) is not tuple or len(field) != DECISIONS:
                    raise ValueError("each variant requires exactly21 decisions")
            if not all(_integer(i) for i in variant.source_sequence_ids) or len(set(variant.source_sequence_ids)) != DECISIONS:
                raise ValueError("source sequence identities must be21 distinct strict integers")
            if shared is None:
                shared = variant.source_sequence_ids
            elif shared != variant.source_sequence_ids:
                raise ValueError("variants must correspond one-to-one by source sequence identity")
            if not all(_integer(i) and i < variant.source_sequence_count for i in variant.sequence_rows):
                raise ValueError("sequence row bounds/type failure")
            if any(b != a + 1 for a, b in zip(variant.sequence_rows, variant.sequence_rows[1:])):
                raise ValueError("21 decisions must be consecutive source sequence rows")
            if not all(type(a) is float and math.isfinite(a) and a >= 0 for a in variant.route_arc_m):
                raise ValueError("explicit finite nonnegative float route arcs required")
            if any(b <= a for a, b in zip(variant.route_arc_m, variant.route_arc_m[1:])):
                raise ValueError("traversal distance must increase strictly")
            last_frame = -1
            for identity, row, frames, traversals, decision_frame, arc in zip(
                variant.source_sequence_ids, variant.sequence_rows, variant.frame_rows,
                variant.frame_traversal_ids, variant.decision_frame_rows, variant.route_arc_m,
            ):
                if type(frames) is not tuple or len(frames) != HISTORY or not all(_integer(f) and f < variant.source_frame_count for f in frames):
                    raise ValueError("exact five historical frame rows within source bounds required")
                if any(b <= a for a, b in zip(frames, frames[1:])) or frames[-1] <= last_frame:
                    raise ValueError("historical frame rows and decision frames must be strictly chronological")
                if not _integer(decision_frame) or decision_frame != frames[-1]:
                    raise ValueError("five causal frames must end at the explicit current decision frame")
                if type(traversals) is not tuple or len(traversals) != HISTORY or any(t != candidate.traversal_id for t in traversals):
                    raise ValueError("all five frames must explicitly belong to the same traversal")
                for frame in frames:
                    framekey = (variant.task, frame)
                    if framekey in frame_identity and frame_identity[framekey] != candidate.traversal_id:
                        raise ValueError("same raw frame claimed by different traversals")
                    frame_identity[framekey] = candidate.traversal_id
                last_frame = frames[-1]
                record = (row, frames, arc, candidate.traversal_id)
                key = (variant.task, identity)
                if key in sequence_identity and sequence_identity[key] != record:
                    raise ValueError("same source sequence identity has inconsistent metadata")
                sequence_identity[key] = record
                rowkey = (variant.task, row)
                if rowkey in row_identity and row_identity[rowkey] != identity:
                    raise ValueError("same stored sequence row aliases different source identities")
                row_identity[rowkey] = identity
        identity = (candidate.parent_id, candidate.traversal_id, shared)
        if identity in seen:
            raise ValueError("duplicate candidate source segment")
        seen.add(identity)


def _candidate_key(candidate):
    v = min(candidate.variants, key=lambda x: x.variant)
    return _hash("segment", candidate.parent_id, candidate.independent_structure_id,
                 candidate.traversal_id, v.source_sequence_ids)


def _counts(selected):
    sequences, frames, logical_sequences, structures, traversals, parents = set(), set(), set(), set(), set(), set()
    spacing = []
    for item in selected:
        c = item.candidate
        parents.add(c.parent_id)
        structures.add((c.parent_id, c.independent_structure_id))
        traversals.add((c.parent_id, c.traversal_id))
        for v in sorted(c.variants, key=lambda x: x.variant):
            sequences.update((v.task, x) for x in v.source_sequence_ids)
            logical_sequences.update((c.parent_id, x) for x in v.source_sequence_ids)
            frames.update((v.task, f) for row in v.frame_rows for f in row)
            deltas = tuple(b - a for a, b in zip(v.route_arc_m, v.route_arc_m[1:]))
            spacing.append({"parent_id": c.parent_id, "structure_id": c.independent_structure_id,
                            "traversal_id": c.traversal_id, "variant": v.variant,
                            "decision_spacing_m": deltas, "span_m": v.route_arc_m[-1] - v.route_arc_m[0],
                            "history_frame_index_gaps": tuple(tuple(b - a for a, b in zip(row, row[1:])) for row in v.frame_rows),
                            "decision_frame_index_gaps": tuple(b - a for a, b in zip(v.decision_frame_rows, v.decision_frame_rows[1:]))})
    return {"segments": len(selected), "independent_parents": len(parents), "independent_structures": len(structures),
            "traversals": len(traversals), "decision_observation_references": len(selected) * DECISIONS * 3,
            "unique_variant_sequences": len(sequences), "unique_logical_sequences": len(logical_sequences),
            "raw_frame_references": len(selected) * DECISIONS * HISTORY * 3, "unique_raw_frames": len(frames),
            "spacing": tuple(spacing), "duration_s": None, "acquisition_clock_known": False}


def sample_review_segments(population, candidates, *, variant_names):
    """Fixed hash/parent-round-robin choice; shortages return a partial ledger.

    Strata are caller metadata for review sampling, NOT accepted event labels.
    Identity-only decisions, no replacement, and no implicit data reads.
    """
    mapping = partition_parents(population)
    _validate_candidates(candidates, mapping, variant_names)
    # Collapse repeat visits before quota selection, not after counting them.
    independent = {}
    for candidate in candidates:
        candidate = replace(candidate, variants=tuple(sorted(candidate.variants, key=lambda v: v.variant)))
        key = (candidate.parent_id, candidate.independent_structure_id)
        if key not in independent or _candidate_key(candidate) < _candidate_key(independent[key]):
            independent[key] = candidate
    selected, deficits, available = [], [], {}
    for split, quota in QUOTAS.items():
        available[split] = {}
        for stratum_index, stratum in enumerate(STRATA):
            buckets = defaultdict(list)
            for c in independent.values():
                if mapping[c.parent_id] == split and c.stratum == stratum:
                    buckets[c.parent_id].append(c)
            available[split][stratum] = sum(map(len, buckets.values()))
            for bucket in buckets.values():
                bucket.sort(key=_candidate_key)
            order = sorted(buckets, key=lambda p: (_hash("parent", p), p))
            if order:
                offset = stratum_index % len(order)
                order = order[offset:] + order[:offset]
            chosen = []
            while len(chosen) < quota:
                progressed = False
                for parent in order:
                    if buckets[parent]:
                        chosen.append(buckets[parent].pop(0))
                        progressed = True
                        if len(chosen) == quota:
                            break
                if not progressed:
                    break
            selected.extend(SelectedSegment(split, c) for c in chosen)
            if len(chosen) < quota:
                deficits.append({"split": split, "stratum": stratum, "required": quota,
                                 "available_independent_structures": available[split][stratum],
                                 "selected": len(chosen), "missing": quota - len(chosen)})
    selected = tuple(selected)
    identity_ledger = [asdict(x) for x in selected]
    return SamplingResult(not deficits, selected, tuple(deficits), {
        "seed": SEED, "parent_inventory_sha256": population.inventory_sha256, "parent_split": mapping,
        "variant_names": tuple(sorted(variant_names)), "candidate_segments": len(candidates),
        "available_independent_structures": len(independent), "available_by_split_stratum": available,
        "removed_repeat_structure_candidates": len(candidates) - len(independent),
        "selection_sha256": _hash("selection", population.inventory_sha256, mapping, identity_ledger), "target_segments": 48,
        "selected_counts": _counts(selected),
        "split_counts": {s: _counts(tuple(x for x in selected if x.split == s)) for s in QUOTAS},
        "real_data_verified": False, "human_review_complete": False,
        "metadata_truth_requires_source_reader": True,
        "continuity_requires_source_audit": True,
    })
