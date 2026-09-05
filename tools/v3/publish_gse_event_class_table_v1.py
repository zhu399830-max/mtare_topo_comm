#!/usr/bin/env python3
"""Publish the five-class GSE versus M1D event table from sealed C09 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


EXPECTED_RUN_ID = "gate3_20260824_gse_perception_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
TABLE_ID = "gse_event_class_table"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise RuntimeError("event table requires three finite seed values")
    return {"mean": float(array.mean()), "sample_std": float(array.std(ddof=1))}


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    summary = load_json(run_dir / "metrics/summary.json")
    gate = load_json(run_dir / "metrics/perception_gate.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or summary.get("overall_status") != EXPECTED_STATUS
        or gate.get("passed") is not True
        or gate.get("strict_test_worlds_read") != 0
        or gate.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("event-class table requires the sealed perception PASS")

    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    baseline_path = run_dir / "artifacts/exit_only_baseline/summary.json"
    source_paths = [run_dir / "RUN_STATE.json", run_dir / "metrics/summary.json", run_dir / "metrics/perception_gate.json", baseline_path]
    baseline = load_json(baseline_path)
    if baseline.get("status") != "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1":
        raise RuntimeError("event-class baseline source is not PASS")
    baseline_by_seed = {int(item["seed"]): item for item in baseline.get("per_seed", ())}
    gse_by_seed: dict[int, dict] = {}
    for seed in (0, 1, 2):
        path = run_dir / f"artifacts/calibration/seed{seed}_summary.json"
        source_paths.append(path)
        record = load_json(path)
        selected = record.get("event", {}).get("selected_class_metrics", {})
        if (
            record.get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1"
            or record.get("strict_test_worlds_read") != 0
            or record.get("mtare_worlds_read") != 0
            or selected.get("selection_effect") != "NONE_REPLAY_OF_FROZEN_EVENT_POINT"
            or abs(float(selected.get("macro_f1", -1.0)) - float(record["event"]["rejection_selection"]["macro_f1"])) > 1e-12
        ):
            raise RuntimeError(f"seed {seed} selected event-class evidence is invalid")
        gse_by_seed[seed] = selected
    if set(baseline_by_seed) != {0, 1, 2}:
        raise RuntimeError("event-class baseline seeds are incomplete")
    for path in source_paths:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"event-class source is absent from the run seal: {relative}")

    raw_rows = []
    aggregate: dict[str, dict] = {}
    for event_name in EVENT_NAMES:
        supports = []
        aggregate[event_name] = {}
        for method in ("M1D exit-only", "GSE-Graph"):
            method_rows = []
            for seed in (0, 1, 2):
                metrics = (
                    baseline_by_seed[seed]["event"]["per_class"][event_name]
                    if method == "M1D exit-only"
                    else gse_by_seed[seed]["per_class"][event_name]
                )
                row = {
                    "event": event_name,
                    "method": method,
                    "seed": seed,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "support": int(metrics["support"]),
                }
                raw_rows.append(row)
                method_rows.append(row)
                supports.append(row["support"])
            aggregate[event_name][method] = {
                metric: _mean_std([row[metric] for row in method_rows])
                for metric in ("precision", "recall", "f1")
            }
        if len(set(supports)) != 1:
            raise RuntimeError(f"event support drift across methods/seeds: {event_name}")
        aggregate[event_name]["support"] = supports[0]
        aggregate[event_name]["f1_absolute_improvement"] = (
            aggregate[event_name]["GSE-Graph"]["f1"]["mean"]
            - aggregate[event_name]["M1D exit-only"]["f1"]["mean"]
        )

    names = (
        f"{TABLE_ID}.csv",
        f"{TABLE_ID}.md",
        f"{TABLE_ID}.tex",
        f"{TABLE_ID}_source.json",
        f"{TABLE_ID}_provenance.json",
        f"{TABLE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("event-class table destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{TABLE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("event", "method", "seed", "precision", "recall", "f1", "support"))
        writer.writeheader()
        writer.writerows(raw_rows)

    def pm(record: dict[str, float]) -> str:
        return f"{record['mean']:.3f} ± {record['sample_std']:.3f}"

    markdown = [
        "| Event | Support | M1D P | M1D R | M1D F1 | GSE P | GSE R | GSE F1 | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Event & Support & M1D P & M1D R & M1D F1 & GSE P & GSE R & GSE F1 & $\\Delta$F1 \\\\",
        "\\midrule",
    ]
    for event_name in EVENT_NAMES:
        row = aggregate[event_name]
        markdown.append(
            f"| {event_name.replace('_', ' ')} | {row['support']} | {pm(row['M1D exit-only']['precision'])} | {pm(row['M1D exit-only']['recall'])} | {pm(row['M1D exit-only']['f1'])} | {pm(row['GSE-Graph']['precision'])} | {pm(row['GSE-Graph']['recall'])} | {pm(row['GSE-Graph']['f1'])} | {row['f1_absolute_improvement']:+.3f} |"
        )
        label = event_name.replace("_", "\\_")
        latex.append(
            f"{label} & {row['support']} & {pm(row['M1D exit-only']['precision']).replace('±', '$\\pm$')} & {pm(row['M1D exit-only']['recall']).replace('±', '$\\pm$')} & {pm(row['M1D exit-only']['f1']).replace('±', '$\\pm$')} & {pm(row['GSE-Graph']['precision']).replace('±', '$\\pm$')} & {pm(row['GSE-Graph']['recall']).replace('±', '$\\pm$')} & {pm(row['GSE-Graph']['f1']).replace('±', '$\\pm$')} & {row['f1_absolute_improvement']:+.3f} \\\\"
        )
    latex.extend(("\\bottomrule", "\\end{tabular}"))
    (destination / f"{TABLE_ID}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (destination / f"{TABLE_ID}.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    write_json(
        destination / f"{TABLE_ID}_source.json",
        {
            "schema_version": "gse_event_class_table_source_v1",
            "selection_effect": "NONE_REPORTS_FROZEN_SELECTED_POINTS",
            "raw_seed_rows": raw_rows,
            "aggregate": aggregate,
        },
    )
    write_json(
        destination / f"{TABLE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_table_provenance_v1",
            "table_id": TABLE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "all five event classes and all three predeclared seeds for both methods; GSE uses its frozen rejection point and M1D uses its frozen role-to-event mapping",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in source_paths},
        },
    )
    manifest = destination / f"{TABLE_ID}_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {"table_id": TABLE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
