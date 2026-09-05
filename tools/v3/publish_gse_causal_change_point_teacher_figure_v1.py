#!/usr/bin/env python3
"""Publish the paper-grade causal change-point Teacher evidence bundle."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROOF_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
OLD_TEACHER_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
OUTPUT_DIR = PROJECT_ROOT / "docs/figures/gse_graph"
STEM = OUTPUT_DIR / "gse_causal_change_point_teacher"
EXPECTED_PROOF_STATUS = "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_seal(run_dir: Path) -> str:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    expected_files: set[Path] = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, raw_path = line.split("  ", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"sealed source mismatch: {path}")
        expected_files.add(path.resolve())
    actual_files = {
        path.resolve()
        for path in run_dir.rglob("*")
        if path.is_file() and path != seal
    }
    if expected_files != actual_files:
        raise RuntimeError("source seal does not cover the run directory exactly")
    return _sha256(seal)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    state = _load_json(PROOF_RUN / "RUN_STATE.json")
    summary = _load_json(PROOF_RUN / "metrics/summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != EXPECTED_PROOF_STATUS:
        raise RuntimeError("causal change-point proof is not sealed PASS")
    if summary.get("overall_status") != EXPECTED_PROOF_STATUS:
        raise RuntimeError("proof summary status drift")
    proof_seal_sha = _verify_seal(PROOF_RUN)
    old_seal_sha = _verify_seal(OLD_TEACHER_RUN)

    labels_path = PROOF_RUN / "artifacts/causal_change_point_labels.jsonl"
    old_observations_path = OLD_TEACHER_RUN / "artifacts/teacher_observations.jsonl"
    labels: dict[tuple[str, int], dict] = {}
    with labels_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row["traversal_id"]), int(row["sequence_index"]))
            if key in labels:
                raise RuntimeError("proof labels are not unique")
            labels[key] = row
    retained_by_family: Counter[str] = Counter()
    suppressed_by_family: Counter[str] = Counter()
    retained_identities: set[str] = set()
    matched_keys: set[tuple[str, int]] = set()
    with old_observations_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            key = (str(row["traversal_id"]), int(row["sequence_index"]))
            label = labels.get(key)
            if label is None:
                continue
            matched_keys.add(key)
            family = str(row["parent_id"]).split("_", 1)[0]
            if row["event"] in {"junction", "terminal"}:
                suppressed_by_family[family] += 1
            else:
                retained_by_family[family] += 1
                retained_identities.add(str(label["identity"]))
    if matched_keys != set(labels):
        raise RuntimeError("proof labels do not map exactly to old Teacher observations")
    if sum(retained_by_family.values()) != 1031 or sum(suppressed_by_family.values()) != 59:
        raise RuntimeError("terminal/junction priority audit drift")
    if len(retained_identities) != 76:
        raise RuntimeError("priority audit lost a causal identity")

    source = _load_json(PROOF_RUN / "previews/gse_causal_change_point_proof_source.json")
    family_rows = source["family_rows"]
    families = [str(row["family"]) for row in family_rows]
    old_identity = np.asarray([int(row["old_transition_identity_count"]) for row in family_rows])
    new_identity = np.asarray([int(row["new_emitted_identity_count"]) for row in family_rows])
    retained = np.asarray([retained_by_family[family] for family in families])
    suppressed = np.asarray([suppressed_by_family[family] for family in families])
    pipeline = source["pipeline"]
    delays = np.asarray(source["causal_delays_m"], dtype=np.float64)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.6), constrained_layout=True)
    x = np.arange(len(families))
    axes[0, 0].bar(x - 0.19, old_identity, width=0.38, color="#9CA3AF", label="old short-frame identities")
    axes[0, 0].bar(x + 0.19, new_identity, width=0.38, color="#2563EB", label="persistent causal identities")
    axes[0, 0].set_yscale("symlog", linthresh=1.0)
    axes[0, 0].set_xticks(x, families)
    axes[0, 0].set_ylabel("identity count (symlog)")
    axes[0, 0].set_title("a  Stable identities replace short mesh fluctuations", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(families, retained, color="#0EA5E9", label="retained causal labels")
    axes[0, 1].bar(
        families,
        suppressed,
        bottom=retained,
        color="#CBD5E1",
        hatch="//",
        label="junction/terminal priority suppression",
    )
    axes[0, 1].set_ylabel("five-frame observations")
    axes[0, 1].set_title("b  Final Teacher support: 1,031 retained > global floor 500", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8)

    pipeline_names = ["directional\npersistent", "bidirectional", "causal both", "emitted"]
    pipeline_values = [
        int(pipeline["persistent_directional_episode_count"]),
        int(pipeline["bidirectional_candidate_count"]),
        int(pipeline["causal_bidirectional_candidate_count"]),
        int(pipeline["emitted_point_count"]),
    ]
    axes[1, 0].bar(pipeline_names, pipeline_values, color=["#94A3B8", "#60A5FA", "#22C55E", "#15803D"])
    for index, value in enumerate(pipeline_values):
        axes[1, 0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    axes[1, 0].set_ylabel("candidate/event count")
    axes[1, 0].set_title("c  Fail-closed evidence funnel", loc="left")

    axes[1, 1].hist(delays, bins=np.arange(6.75, 10.76, 0.5), color="#F59E0B", edgecolor="white")
    axes[1, 1].axvline(float(np.median(delays)), color="#92400E", linestyle="--", linewidth=1.3, label=f"median {np.median(delays):.1f} m")
    axes[1, 1].set_xlabel("causal detection delay after boundary (m)")
    axes[1, 1].set_ylabel("directional episodes")
    axes[1, 1].set_title("d  Delayed evidence is back-projected to the boundary", loc="left")
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("Persistent bidirectional causal geometry change-point Teacher — C01–C08")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {})):
        figure.savefig(STEM.with_suffix(f".{suffix}"), **kwargs)
    plt.close(figure)

    csv_path = STEM.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "old_transition_identities", "persistent_causal_identities", "retained_causal_labels", "junction_terminal_suppressed_labels"))
        for index, family in enumerate(families):
            writer.writerow((family, int(old_identity[index]), int(new_identity[index]), int(retained[index]), int(suppressed[index])))

    source_path = OUTPUT_DIR / f"{STEM.name}_source.json"
    source_document = {
        "schema_version": "gse_causal_change_point_teacher_paper_source_v1",
        "formal_proof_summary": summary,
        "formal_figure_source": source,
        "priority_integration": {
            "proof_labels": len(labels),
            "retained_labels": int(sum(retained_by_family.values())),
            "junction_terminal_suppressed_labels": int(sum(suppressed_by_family.values())),
            "retained_identities": len(retained_identities),
            "retained_by_family": dict(sorted(retained_by_family.items())),
            "suppressed_by_family": dict(sorted(suppressed_by_family.items())),
        },
    }
    source_path.write_text(json.dumps(source_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_path = OUTPUT_DIR / f"{STEM.name}_provenance.json"
    provenance = {
        "schema_version": "gse_causal_change_point_teacher_paper_provenance_v1",
        "generator": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "formal_proof_run": str(PROOF_RUN.relative_to(PROJECT_ROOT)),
        "formal_proof_seal_sha256": proof_seal_sha,
        "old_teacher_run": str(OLD_TEACHER_RUN.relative_to(PROJECT_ROOT)),
        "old_teacher_seal_sha256": old_seal_sha,
        "scope": "C01-C08 only; C09/C10/M-TARE/model/training reads zero",
        "correction": "The formal proof figure remains immutable. This paper figure correctly applies junction/terminal label priority and does not draw the global 500-observation floor as a per-family requirement."
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    bundle = [
        STEM.with_suffix(".png"),
        STEM.with_suffix(".pdf"),
        STEM.with_suffix(".svg"),
        csv_path,
        source_path,
        provenance_path,
    ]
    manifest_path = OUTPUT_DIR / f"{STEM.name}_sha256.txt"
    manifest_path.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in bundle),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS_PAPER_FIGURE_PUBLISHED",
                "bundle_files": len(bundle),
                "manifest_sha256": _sha256(manifest_path),
                "retained_labels": int(sum(retained_by_family.values())),
                "retained_identities": len(retained_identities),
                "c09_c10_mtare_reads": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
