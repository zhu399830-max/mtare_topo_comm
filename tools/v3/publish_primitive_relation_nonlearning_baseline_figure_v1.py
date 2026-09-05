#!/usr/bin/env python3
"""Publish the sealed C07 same-input non-learning primitive baseline figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


FIGURE_ID = "primitive_relation_nonlearning_baseline"
SOURCE_RUN_ID = "gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"
SOURCE_STATUS = "PASS_PRIMITIVE_RELATION_NONLEARNING_C07_V1"
SOURCE_EVIDENCE_SHA256 = "605df99e2c542642bdc45d9530a10c89df1512cde95651c97f1a822688a637f7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def publish(destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    destination = destination.resolve()
    destination.relative_to(root)
    source_run = root / "results/gate3_semantics" / SOURCE_RUN_ID
    state = _load_json(source_run / "RUN_STATE.json")
    if state.get("state") != "COMPLETED" or state.get("error") is not None:
        raise RuntimeError("source baseline run is not a clean completion")
    if state.get("overall_status") != SOURCE_STATUS:
        raise RuntimeError("source baseline status drift")
    evidence = source_run / "artifacts/evidence_sha256.txt"
    if _sha256(evidence) != SOURCE_EVIDENCE_SHA256:
        raise RuntimeError("source baseline evidence manifest drift")

    names = (
        f"{FIGURE_ID}.png",
        f"{FIGURE_ID}.pdf",
        f"{FIGURE_ID}.svg",
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("non-learning baseline publication bundle exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    preview_root = source_run / "previews"
    source_figures = {
        suffix: preview_root / f"nonlearning_c07_baseline.{suffix}"
        for suffix in ("png", "pdf", "svg")
    }
    if not all(path.is_file() for path in source_figures.values()):
        raise RuntimeError("source baseline figure bundle is incomplete")
    for suffix, source in source_figures.items():
        shutil.copyfile(source, destination / f"{FIGURE_ID}.{suffix}")

    summary_path = source_run / "metrics/summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("scientific_pass") is not True
        or summary.get("rows") != 64644
        or summary.get("independent_parent_worlds") != 10
        or summary.get("paired_geometry_tasks") != 30
        or summary.get("c08_rows_read") != 0
        or summary.get("c09_c10_worlds_read") != 0
    ):
        raise RuntimeError("source baseline population or isolation drift")

    compact_source = {
        "schema_version": "primitive_relation_nonlearning_baseline_figure_source_v1",
        "figure_id": FIGURE_ID,
        "partition": summary["partition"],
        "independent_parent_worlds": summary["independent_parent_worlds"],
        "paired_geometry_tasks": summary["paired_geometry_tasks"],
        "rows": summary["rows"],
        "primitive_detection": summary["primitive_detection"],
        "geometry": summary["geometry"],
        "attachment": summary["attachment"],
        "disconnected_overlap": summary["disconnected_overlap"],
        "temporal_correspondence_accuracy": summary["temporal_correspondence_accuracy"],
        "c08_rows_read": summary["c08_rows_read"],
        "c09_c10_worlds_read": summary["c09_c10_worlds_read"],
        "source_summary_sha256": _sha256(summary_path),
    }
    source_output = destination / f"{FIGURE_ID}_source.json"
    write_json(source_output, compact_source)

    generator = Path(__file__).resolve()
    provenance_output = destination / f"{FIGURE_ID}_provenance.json"
    write_json(
        provenance_output,
        {
            "schema_version": "primitive_relation_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "kind": "same-input non-learning C07 development baseline",
            "source_run": str(source_run.relative_to(root)),
            "source_evidence_manifest_sha256": SOURCE_EVIDENCE_SHA256,
            "source_summary": str(summary_path.relative_to(root)),
            "source_summary_sha256": _sha256(summary_path),
            "source_figure_sha256": {
                suffix: _sha256(path) for suffix, path in source_figures.items()
            },
            "generator": str(generator.relative_to(root)),
            "generator_sha256": _sha256(generator),
            "manual_value_entry": False,
            "model_outcome_values": False,
            "c08_reads": 0,
            "c09_c10_reads": 0,
        },
    )

    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    retained = [path for path in targets if path != manifest]
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(retained)),
        encoding="utf-8",
    )
    return {
        "figure_id": FIGURE_ID,
        "published_files": len(targets),
        "manifest_sha256": _sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
