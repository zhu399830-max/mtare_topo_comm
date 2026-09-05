#!/usr/bin/env python3
"""Generate the immutable C01-C08 corrected causal Teacher manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_corrected_teacher_manifest import (
    apply_causal_change_point_labels,
    corrected_identity_summaries,
    corrected_manifest_summary,
    corrected_world_association_pairs,
)
from mtare_topo.governance import write_json


EXPECTED_EVENT_COUNTS = {
    "corridor": 150964,
    "geometry_transition": 1031,
    "junction": 26608,
    "terminal": 7525,
    "turn": 1998,
}
EXPECTED_IDENTITY_COUNTS = {
    "geometry_transition": 76,
    "junction": 563,
    "terminal": 503,
    "turn": 392,
}
EXPECTED_PAIR_COUNTS = {
    "cross_traversal_positive": 37029,
    "same_event_geometry_hard_negative": 36972,
    "same_traversal_revisit_positive": 63,
}
EXPECTED_OBSERVATIONS = 188126
EXPECTED_TRAVERSALS = 16078
EXPECTED_PROOF_LABELS = 1090
EXPECTED_APPLIED_LABELS = 1031
EXPECTED_SUPPRESSED_LABELS = 59


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _load_causal_labels(path: Path) -> tuple[dict[tuple[str, int], dict], dict[str, str]]:
    labels: dict[tuple[str, int], dict] = {}
    identity_kind: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row["traversal_id"]), int(row["sequence_index"]))
            if key in labels:
                raise RuntimeError("causal proof label keys are not unique")
            labels[key] = row
            identity = str(row["identity"])
            kind = str(row["identity_kind"])
            previous = identity_kind.setdefault(identity, kind)
            if previous != kind:
                raise RuntimeError("causal identity kind drift")
    if len(labels) != EXPECTED_PROOF_LABELS or len(identity_kind) != 76:
        raise RuntimeError("causal proof label/identity count drift")
    return labels, identity_kind


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--old-teacher-run", required=True, type=Path)
    parser.add_argument("--proof-run", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    old_run = args.old_teacher_run.resolve()
    proof_run = args.proof_run.resolve()
    started = time.monotonic()
    for directory in (run_dir / "artifacts", run_dir / "metrics"):
        directory.mkdir(parents=True, exist_ok=True)

    labels, causal_identity_kind = _load_causal_labels(
        proof_run / "artifacts/causal_change_point_labels.jsonl"
    )
    labels_by_parent: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    for key, row in labels.items():
        labels_by_parent[str(row["parent_id"])][key] = row

    streams = {
        name: (run_dir / f"artifacts/{name}.jsonl").open("w", encoding="utf-8")
        for name in (
            "teacher_observations",
            "traversal_manifest",
            "association_pairs",
            "structural_identities",
            "label_application_audit",
            "world_manifest_summary",
        )
    }
    traversal_count = 0
    with (old_run / "artifacts/traversal_manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            _write_row(streams["traversal_manifest"], row)
            traversal_count += 1
    if traversal_count != EXPECTED_TRAVERSALS:
        raise RuntimeError("corrected traversal count drift")

    world_rows: dict[str, list[dict]] = defaultdict(list)
    with (old_run / "artifacts/teacher_observations.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            parent_id = str(row["parent_id"])
            if not parent_id.endswith(tuple(f"_C{index:02d}" for index in range(1, 9))):
                raise RuntimeError("non-C01-C08 observation entered corrected Teacher")
            world_rows[parent_id].append(row)
    if len(world_rows) != 80 or sum(len(rows) for rows in world_rows.values()) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("corrected Teacher source scope drift")

    event_totals: Counter[str] = Counter()
    pair_totals: Counter[str] = Counter()
    identity_sets: dict[str, set[str]] = defaultdict(set)
    action_totals: Counter[str] = Counter()
    observation_ids: set[str] = set()
    global_sequence_indices: set[int] = set()
    consumed_label_keys: set[tuple[str, int]] = set()
    identity_count = 0
    pair_count = 0
    for world_index, (parent_id, raw_rows) in enumerate(sorted(world_rows.items()), start=1):
        world_labels = labels_by_parent[parent_id]
        corrected, audit = apply_causal_change_point_labels(raw_rows, world_labels)
        consumed_label_keys.update(world_labels)
        pairs = corrected_world_association_pairs(corrected)
        identities = corrected_identity_summaries(corrected, causal_identity_kind)
        world_summary = corrected_manifest_summary(corrected)
        world_pair_counts = Counter(str(pair["pair_kind"]) for pair in pairs)
        world_action_counts = Counter(str(row["action"]) for row in audit)

        for old, new in zip(raw_rows, corrected, strict=True):
            if old.keys() != new.keys():
                raise RuntimeError("corrected observation schema drift")
            for key in old:
                if key not in {"event", "identity"} and old[key] != new[key]:
                    raise RuntimeError(f"non-Teacher observation field changed: {key}")
            observation_id = str(new["observation_id"])
            global_index = int(new["global_sequence_index"])
            if observation_id in observation_ids or global_index in global_sequence_indices:
                raise RuntimeError("corrected observation/global sequence identity duplicated")
            observation_ids.add(observation_id)
            global_sequence_indices.add(global_index)
            _write_row(streams["teacher_observations"], new)
            event_totals[str(new["event"])] += 1
            if new["event"] != "corridor":
                identity_sets[str(new["event"])].add(str(new["identity"]))
        for row in audit:
            _write_row(streams["label_application_audit"], row)
            action_totals[str(row["action"])] += 1
        for pair in pairs:
            _write_row(streams["association_pairs"], {"parent_id": parent_id, "split": "train", **pair})
            pair_totals[str(pair["pair_kind"])] += 1
            pair_count += 1
        for identity in identities:
            _write_row(streams["structural_identities"], {"split": "train", **identity})
            identity_count += 1
        _write_row(
            streams["world_manifest_summary"],
            {
                "parent_id": parent_id,
                "split": "train",
                **world_summary,
                "association_pair_counts": dict(sorted(world_pair_counts.items())),
                "label_application_counts": dict(sorted(world_action_counts.items())),
            },
        )
        print(
            json.dumps(
                {
                    "world": parent_id,
                    "index": world_index,
                    "of": len(world_rows),
                    "events": world_summary["event_counts"],
                    "identities": world_summary["structural_identity_count"],
                    "pairs": len(pairs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for stream in streams.values():
        stream.close()

    applied = action_totals["applied_geometry_transition"]
    suppressed = (
        action_totals["suppressed_by_junction_priority"]
        + action_totals["suppressed_by_terminal_priority"]
    )
    checks = {
        "exact_80_worlds": len(world_rows) == 80,
        "exact_16078_traversals": traversal_count == EXPECTED_TRAVERSALS,
        "exact_188126_observations": len(observation_ids) == EXPECTED_OBSERVATIONS,
        "global_sequence_indices_exact_0_to_188125": global_sequence_indices == set(range(EXPECTED_OBSERVATIONS)),
        "all_1090_proof_labels_consumed_once": consumed_label_keys == set(labels),
        "exact_1031_labels_applied": applied == EXPECTED_APPLIED_LABELS,
        "exact_59_protected_labels_suppressed": suppressed == EXPECTED_SUPPRESSED_LABELS,
        "event_counts_exact": dict(event_totals) == EXPECTED_EVENT_COUNTS,
        "identity_counts_exact": {event: len(values) for event, values in identity_sets.items()} == EXPECTED_IDENTITY_COUNTS,
        "pair_counts_exact": dict(pair_totals) == EXPECTED_PAIR_COUNTS,
        "pair_total_exact_74064": pair_count == 74064,
        "structural_identity_total_exact_1534": identity_count == 1534,
        "old_transition_identity_absent": all(
            ":geometry_transition:" not in identity
            for values in identity_sets.values()
            for identity in values
        ),
        "new_change_point_identity_exact_76": len(identity_sets["geometry_transition"]) == 76,
        "zero_c09_c10_mtare_model_training_reads": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_corrected_causal_teacher_manifest_v1",
        "overall_status": "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1" if passed else "FAIL_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1",
        "world_count": len(world_rows),
        "traversal_count": traversal_count,
        "observation_count": len(observation_ids),
        "event_counts": {name: int(event_totals[name]) for name in EXPECTED_EVENT_COUNTS},
        "identity_counts": {name: len(identity_sets[name]) for name in EXPECTED_IDENTITY_COUNTS},
        "structural_identity_count": identity_count,
        "association_pair_counts": {name: int(pair_totals[name]) for name in EXPECTED_PAIR_COUNTS},
        "association_pair_count": pair_count,
        "label_application_counts": dict(sorted(action_totals.items())),
        "proof_label_count": len(labels),
        "applied_change_point_label_count": applied,
        "protected_suppressed_label_count": suppressed,
        "checks": checks,
        "c09_worlds_consumed": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "model_inference_frames": 0,
        "training_samples_consumed": 0,
        "optimizer_steps": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("corrected causal Teacher manifest checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
