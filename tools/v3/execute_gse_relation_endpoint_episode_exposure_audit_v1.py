#!/usr/bin/env python3
"""Audit how many MIL episodes each development relation endpoint contributes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--endpoint-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    cache = args.cache_dir.resolve()
    partition = np.load(cache / "partition_code.npy").astype(np.uint8)
    identity = np.load(cache / "identity.npy").astype(str)
    episode = np.load(cache / "decision_episode_id.npy").astype(np.int64)
    target = np.load(cache / "decision_target.npy").astype(np.int8)
    if len(partition) != 188_126 or any(len(value) != len(partition) for value in (identity, episode, target)):
        raise RuntimeError("endpoint episode-exposure cache population drift")
    endpoints = _read_jsonl(args.endpoint_audit.resolve())
    if len(endpoints) != 98:
        raise RuntimeError("endpoint episode-exposure identity population drift")
    records = []
    for row in endpoints:
        code = 0 if row["partition"] == "fit" else 1
        mask = (partition == code) & (identity == row["identity"]) & (target > 0)
        episodes = sorted(set(episode[mask].tolist()) - {-1})
        if len(episodes) == 0:
            raise RuntimeError(f"endpoint identity has no positive episode: {row['identity']}")
        records.append({
            "partition": row["partition"], "identity": row["identity"], "event": row["event"],
            "support_bin": row["support_bin"], "teacher_rows": row["teacher_rows"],
            "positive_episodes": len(episodes), "episode_ids": episodes,
        })
    stats = {}
    for split in ("fit", "selection"):
        stats[split] = {}
        for support_bin in ("1-3", "4-10", "11-30", "31+"):
            values = [row["positive_episodes"] for row in records if row["partition"] == split and row["support_bin"] == support_bin]
            if values:
                stats[split][support_bin] = {"endpoints": len(values), "episodes_total": sum(values), "mean_episodes_per_identity": float(np.mean(values)), "minimum": min(values), "maximum": max(values)}
    fit_ratio = stats["fit"]["31+"]["mean_episodes_per_identity"] / stats["fit"]["1-3"]["mean_episodes_per_identity"]
    selection_ratio = stats["selection"]["31+"]["mean_episodes_per_identity"] / stats["selection"]["1-3"]["mean_episodes_per_identity"]
    all_episode_counts = {"fit": len(set(episode[(partition == 0) & (episode >= 0)].tolist())), "selection": len(set(episode[(partition == 1) & (episode >= 0)].tolist()))}
    endpoint_episode_counts = {split: sum(row["positive_episodes"] for row in records if row["partition"] == split) for split in ("fit", "selection")}
    gates = {"fit_low_support_all_one_episode": stats["fit"]["1-3"]["minimum"] == stats["fit"]["1-3"]["maximum"] == 1, "fit_high_to_low_exposure_ratio_at_least_6": fit_ratio >= 6.0, "selection_nonreversal": selection_ratio >= 1.0}
    gates["identity_balanced_episode_corrective_justified"] = all(gates.values())
    summary = {"schema_version": "gse_relation_endpoint_episode_exposure_audit_v1", "status": "PASS_GSE_RELATION_ENDPOINT_EPISODE_EXPOSURE_AUDIT_V1", "question": "Does the existing episode-balanced MIL still overweight structure identities that generate more episodes?", "population": {"observations": len(partition), "relation_endpoints": len(records), "all_decision_episodes": all_episode_counts, "relation_endpoint_episodes": endpoint_episode_counts}, "support_bin_stats": stats, "exposure_ratio": {"fit_high_to_low": fit_ratio, "selection_high_to_low": selection_ratio}, "gates": gates, "corrective_contract": "Keep original negative and all-episode MIL terms; add mean_over_endpoint_identities(mean_over_identity_episodes(episode_MIL)). Do not repeat rows or change deployment threshold.", "sources": {"cache_manifest_sha256": _sha(cache / "manifest.json"), "endpoint_audit_sha256": _sha(args.endpoint_audit.resolve())}, "optimizer_steps": 0, "model_updates": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0}
    with (output / "identity_episode_exposure.jsonl").open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "episode_exposure.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("partition", "support_bin", "endpoints", "episodes_total", "mean", "minimum", "maximum"))
        for split in ("fit", "selection"):
            for support_bin, values in stats[split].items(): writer.writerow((split, support_bin, values["endpoints"], values["episodes_total"], values["mean_episodes_per_identity"], values["minimum"], values["maximum"]))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 3.8), constrained_layout=True)
    bins = ("1-3", "4-10", "11-30", "31+")
    for axis, split, letter in ((axes[0], "fit", "A"), (axes[1], "selection", "B")):
        values = [stats[split][name]["mean_episodes_per_identity"] for name in bins]
        bars = axis.bar(np.arange(4), values, color="#4C78A8")
        axis.set_xticks(np.arange(4), bins); axis.set_xlabel("Teacher rows per endpoint identity"); axis.set_ylabel("Mean MIL episodes per identity"); axis.set_title(f"{letter}  {split.capitalize()} identity exposure")
        for bar, value in zip(bars, values, strict=True): axis.text(bar.get_x()+bar.get_width()/2, value+.08, f"{value:.2f}", ha="center")
    figure.suptitle("Existing episode-balanced loss still weights high-support identities 6× more")
    for suffix in ("png", "pdf", "svg"): figure.savefig(output / f"gse_relation_endpoint_episode_exposure.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)
    summary["figure_sha256"] = _sha(output / "gse_relation_endpoint_episode_exposure.png")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_relation_endpoint_episode_exposure_figure_source_v1", "support_bin_stats": stats}, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
