#!/usr/bin/env python3
"""Build one immutable C01--C08 cache from already frozen exit-token outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.representation.gse_action_set_node import (
    fit_action_token_normalization,
    raw_action_tokens_one_seed,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references_unordered,
)


EXPECTED_OBSERVATIONS = 188126
EXPECTED_FIT = 142184
EXPECTED_SELECTION = 45942


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--seed0", required=True, type=Path)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("action-set cache output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)

    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("action-set Teacher population drift")
    global_ids = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    identity = np.asarray([str(row["identity"]) for row in rows])
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    if len(np.unique(global_ids)) != len(global_ids):
        raise RuntimeError("action-set Teacher identities are not unique")

    with np.load(args.pair_cache.resolve(), allow_pickle=False) as pairs:
        compact = pairs["compact_to_global_sequence_index"].astype(np.int64)
        partition = pairs["partition_code"].astype(np.uint8)
        pair_parent = pairs["parent_id"].astype(str)
    if (
        not np.array_equal(compact, global_ids)
        or not np.array_equal(pair_parent, parent)
        or int(np.sum(partition == 0)) != EXPECTED_FIT
        or int(np.sum(partition == 1)) != EXPECTED_SELECTION
        or np.any(~np.isin(partition, (0, 1)))
    ):
        raise RuntimeError("action-set partition identity/count drift")

    token_path = args.output_dir / "raw_tokens.npy"
    token_file = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.float16, shape=(EXPECTED_OBSERVATIONS, 3, 6, 40)
    )
    token_inputs = (args.seed0, args.seed1, args.seed2)
    for seed, path in enumerate(token_inputs):
        with np.load(path.resolve(), allow_pickle=False) as archive:
            archive_ids = archive["global_sequence_index"].astype(np.int64)
            if not np.array_equal(archive_ids, global_ids):
                raise RuntimeError(f"seed{seed} action-token identity drift")
            token_file[:, seed] = raw_action_tokens_one_seed(archive).astype(np.float16)
        token_file.flush()
    del token_file

    history_references, history_mask = causal_history_row_references_unordered(
        traversal, sequence, maximum=5
    )
    episode_bank = materialize_causal_episode_references(rows)
    decision_target = np.where(
        np.isin(episode_bank.event_index, (1, 2)), episode_bank.event_index, 0
    ).astype(np.int8)
    decision_episode = np.where(decision_target > 0, episode_bank.episode_id, -1).astype(np.int64)
    if np.any((decision_target == 0) != (decision_episode < 0)):
        raise RuntimeError("action-set decision episode contract drift")

    np.save(args.output_dir / "global_sequence_index.npy", global_ids)
    np.save(args.output_dir / "partition_code.npy", partition)
    np.save(args.output_dir / "parent_id.npy", parent)
    np.save(args.output_dir / "traversal_id.npy", traversal)
    np.save(args.output_dir / "sequence_index.npy", sequence)
    np.save(args.output_dir / "identity.npy", identity)
    np.save(args.output_dir / "history_references.npy", history_references.astype(np.int32))
    np.save(args.output_dir / "history_mask.npy", history_mask)
    np.save(args.output_dir / "decision_target.npy", decision_target)
    np.save(args.output_dir / "decision_episode_id.npy", decision_episode)
    raw = np.load(token_path, mmap_mode="r")
    fit_rows = np.flatnonzero(partition == 0)
    mean, scale = fit_action_token_normalization(raw, fit_rows)
    np.save(args.output_dir / "normalization_mean.npy", mean)
    np.save(args.output_dir / "normalization_scale.npy", scale)

    fit_events = decision_target[partition == 0]
    selection_events = decision_target[partition == 1]
    files = sorted(path for path in args.output_dir.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema_version": "gse_action_set_node_cache_v1",
        "causal_observations": EXPECTED_OBSERVATIONS,
        "fit_observations": EXPECTED_FIT,
        "selection_observations": EXPECTED_SELECTION,
        "fit_decision_rows": int(np.sum(fit_events > 0)),
        "selection_decision_rows": int(np.sum(selection_events > 0)),
        "fit_decision_episodes": int(len(np.unique(decision_episode[(partition == 0) & (decision_episode >= 0)]))),
        "selection_decision_episodes": int(len(np.unique(decision_episode[(partition == 1) & (decision_episode >= 0)]))),
        "fit_junction_episodes": int(len(np.unique(decision_episode[(partition == 0) & (decision_target == 1)]))),
        "fit_terminal_episodes": int(len(np.unique(decision_episode[(partition == 0) & (decision_target == 2)]))),
        "selection_junction_episodes": int(len(np.unique(decision_episode[(partition == 1) & (decision_target == 1)]))),
        "selection_terminal_episodes": int(len(np.unique(decision_episode[(partition == 1) & (decision_target == 2)]))),
        "history_observations": 5,
        "student_inputs": ["frozen_exit_tokens_seed0", "frozen_exit_tokens_seed1", "frozen_exit_tokens_seed2", "past_only_history_mask"],
        "forbidden_student_inputs": ["event", "identity", "parent_id", "world", "pose", "future_frame", "objective_geometry"],
        "normalization_fit_partition_only": True,
        "token_storage_dtype": "float16",
        "input_sha256": {
            "teacher": _sha256(args.teacher.resolve()),
            "pair_cache": _sha256(args.pair_cache.resolve()),
            **{f"seed{seed}": _sha256(path.resolve()) for seed, path in enumerate(token_inputs)},
        },
        "array_sha256": {path.name: _sha256(path) for path in files},
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
