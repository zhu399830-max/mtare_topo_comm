#!/usr/bin/env python3
"""Publish the active primitive-relation dataset and Teacher overview figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


FIGURE_ID = "primitive_relation_dataset_overview"
P1A_RUN = "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B_RUN = "gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
P1A_EVIDENCE_SHA256 = "79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668"
P1B_EVIDENCE_SHA256 = "f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47"
SHARD_RELATIVE = Path("artifacts/dataset/fit/S01_flat_tree_small_C01__c1_mixed.zarr")
SELECTION_RULE = (
    "Lexicographically first fit topology parent (S01), first fit construction index (C01), "
    "C1-mixed realization, and global frame 0; fixed without inspecting model outcomes."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _verify_run(run: Path, expected_status: str, expected_evidence_sha: str) -> dict:
    state = _load_json(run / "RUN_STATE.json")
    if state.get("state") != "COMPLETED" or state.get("error") is not None:
        raise RuntimeError(f"source run is not a clean completion: {run}")
    if state.get("overall_status") != expected_status:
        raise RuntimeError(f"source run status drift: {run}")
    evidence = run / "artifacts/evidence_sha256.txt"
    if _sha256(evidence) != expected_evidence_sha:
        raise RuntimeError(f"source evidence manifest drift: {run}")
    return state


def _plot_topology(axis, axis_xyz: np.ndarray, traversal_index: np.ndarray, traversal_ids: list[str]) -> dict:
    unique = np.unique(traversal_index)
    endpoints: list[np.ndarray] = []
    plotted_edges = 0
    for index in unique:
        mask = traversal_index == index
        route = axis_xyz[mask]
        endpoints.append(route[0])
        traversal_id = traversal_ids[int(index)]
        if traversal_id.endswith(":d0"):
            axis.plot(route[:, 0], route[:, 1], color="#2A6FBB", linewidth=1.45, alpha=0.92)
            plotted_edges += 1

    rounded = np.round(np.asarray(endpoints), decimals=3)
    node_xyz, degree = np.unique(rounded, axis=0, return_counts=True)
    terminal = degree == 1
    junction = degree >= 3
    ordinary = ~(terminal | junction)
    axis.scatter(node_xyz[ordinary, 0], node_xyz[ordinary, 1], s=12, color="#6B7280", zorder=3, label="degree 2")
    axis.scatter(node_xyz[terminal, 0], node_xyz[terminal, 1], s=25, color="#111827", zorder=4, label="terminal")
    axis.scatter(node_xyz[junction, 0], node_xyz[junction, 1], s=35, color="#D1495B", zorder=5, label="junction")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.22)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("(a) Generated tunnel topology and swept axes", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=8, loc="best")
    return {"physical_edges_plotted": plotted_edges, "nodes": int(node_xyz.shape[0]), "junctions": int(junction.sum())}


def _plot_cross_sections(axis) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    for exponent, label, color in (
        (2.0, "ellipse  n=2", "#2A6FBB"),
        (4.0, "mixed  n=4", "#E67E22"),
        (8.0, "rounded rectangle  n=8", "#2A9D3F"),
    ):
        cosine = np.cos(theta)
        sine = np.sin(theta)
        x = np.sign(cosine) * np.abs(cosine) ** (2.0 / exponent)
        y = np.sign(sine) * np.abs(sine) ** (2.0 / exponent)
        axis.plot(x, y, linewidth=2.0, label=label, color=color)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.22)
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel("lateral / half-width")
    axis.set_ylabel("vertical / half-height")
    axis.set_title("(b) One continuous cross-section family", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=8, loc="lower center")


def publish(destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    destination = destination.resolve()
    destination.relative_to(root)
    p1a = root / "results/gate3_semantics" / P1A_RUN
    p1b = root / "results/gate3_semantics" / P1B_RUN
    _verify_run(p1a, "PASS_PRIMITIVE_RELATION_P1A_LOSSLESS_STORAGE_CORRECTIVE_V1", P1A_EVIDENCE_SHA256)
    _verify_run(p1b, "PASS_PRIMITIVE_RELATION_P1B_TEACHER_MATERIALIZATION_V1R", P1B_EVIDENCE_SHA256)

    names = (
        f"{FIGURE_ID}.png",
        f"{FIGURE_ID}.pdf",
        f"{FIGURE_ID}.svg",
        f"{FIGURE_ID}_source.npz",
        f"{FIGURE_ID}_summary.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("primitive-relation dataset figure exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    shard = p1a / SHARD_RELATIVE
    group = zarr.open_group(str(shard), mode="r")
    axis_xyz = np.asarray(group["axis_xyz_m"][:], dtype=np.float64)
    traversal_index = np.asarray(group["traversal_index"][:], dtype=np.int32)
    range_m = np.asarray(group["range_m"][0], dtype=np.float32)
    valid_mask = np.asarray(group["valid_mask"][0], dtype=np.uint8).astype(bool)
    membership_code = np.asarray(group["primitive_membership_code"][0], dtype=np.uint16)
    traversal_ids = list(group.attrs["traversal_ids"])
    p1a_summary = _load_json(p1a / "metrics/summary.json")
    p1b_summary = _load_json(p1b / "metrics/summary.json")

    source_npz = destination / f"{FIGURE_ID}_source.npz"
    np.savez_compressed(
        source_npz,
        axis_xyz_m=axis_xyz,
        traversal_index=traversal_index,
        range_m=range_m,
        valid_mask=valid_mask.astype(np.uint8),
        primitive_membership_code=membership_code,
        traversal_ids=np.asarray(traversal_ids, dtype="U96"),
    )

    figure = plt.figure(figsize=(13.8, 10.0))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.2, 0.62, 0.62), hspace=0.52, wspace=0.26)
    topology_axis = figure.add_subplot(grid[0, 0])
    section_axis = figure.add_subplot(grid[0, 1])
    range_axis = figure.add_subplot(grid[1, :])
    membership_axis = figure.add_subplot(grid[2, :])

    topology_stats = _plot_topology(topology_axis, axis_xyz, traversal_index, traversal_ids)
    _plot_cross_sections(section_axis)

    masked_range = np.ma.masked_where(~valid_mask, range_m)
    range_image = range_axis.imshow(masked_range, aspect="auto", cmap="viridis", vmin=0.0, vmax=50.0, interpolation="nearest")
    range_axis.set_title("(c) Causal LiDAR frame used by the student", loc="left", weight="bold")
    range_axis.set_ylabel("elevation row")
    range_axis.set_xticks([])
    figure.colorbar(range_image, ax=range_axis, fraction=0.018, pad=0.012, label="range (m)")

    masked_code = np.ma.masked_where(~valid_mask, membership_code)
    code_image = membership_axis.imshow(masked_code, aspect="auto", cmap="tab20", interpolation="nearest")
    membership_axis.set_title("(d) Training-only identity-preserving primitive source-set code", loc="left", weight="bold")
    membership_axis.set_ylabel("elevation row")
    membership_axis.set_xlabel("azimuth column (0.5°)")
    figure.colorbar(code_image, ax=membership_axis, fraction=0.018, pad=0.012, label="uint16 source-set code")

    figure.suptitle("Procedural tunnel construction becomes causal primitive-relation supervision", fontsize=16, weight="bold", y=0.985)
    counts = (
        f"{p1a_summary['worlds']} topology parents · {p1a_summary['tasks']} paired geometry tasks · "
        f"{p1a_summary['frames']:,} scans · {p1b_summary['sequences']:,} five-frame windows · "
        f"{p1b_summary['realized_primitives']:,} realized primitives\n"
        f"{p1b_summary['directed_attachment_labels']:,} directed attachments · "
        f"{p1b_summary['undirected_disconnected_overlap_labels']:,} disconnected overlaps · "
        f"{p1b_summary['temporal_dustbin_labels']:,} temporal dustbins · construction identity is never a deployed-model input"
    )
    figure.text(0.5, 0.016, counts, ha="center", va="bottom", fontsize=9.0, color="#374151", linespacing=1.35)
    figure.subplots_adjust(bottom=0.095, top=0.93, left=0.07, right=0.94)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    summary = {
        "schema_version": "primitive_relation_dataset_figure_source_v1",
        "figure_id": FIGURE_ID,
        "selection_rule": SELECTION_RULE,
        "selected_parent": "S01_flat_tree_small_C01",
        "selected_partition": "fit",
        "selected_geometry": "c1_mixed",
        "selected_global_frame_index": 0,
        "source_array_sha256": {
            "axis_xyz_m": _array_sha256(axis_xyz),
            "traversal_index": _array_sha256(traversal_index),
            "range_m": _array_sha256(range_m),
            "valid_mask": _array_sha256(valid_mask.astype(np.uint8)),
            "primitive_membership_code": _array_sha256(membership_code),
        },
        "topology_stats": topology_stats,
        "dataset_counts": {
            "worlds": p1a_summary["worlds"],
            "tasks": p1a_summary["tasks"],
            "frames": p1a_summary["frames"],
            "rays": p1a_summary["rays"],
            "sequences": p1b_summary["sequences"],
            "realized_primitives": p1b_summary["realized_primitives"],
            "directed_attachment_labels": p1b_summary["directed_attachment_labels"],
            "disconnected_overlap_labels": p1b_summary["undirected_disconnected_overlap_labels"],
            "temporal_dustbin_labels": p1b_summary["temporal_dustbin_labels"],
        },
    }
    summary_path = destination / f"{FIGURE_ID}_summary.json"
    write_json(summary_path, summary)
    contract = root / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md"
    generator = Path(__file__).resolve()
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "primitive_relation_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "kind": "dataset and Teacher overview; no model outcome values",
            "selection_rule": SELECTION_RULE,
            "source_runs": [str(p1a.relative_to(root)), str(p1b.relative_to(root))],
            "source_evidence_manifest_sha256": [P1A_EVIDENCE_SHA256, P1B_EVIDENCE_SHA256],
            "source_shard": str(shard.relative_to(root)),
            "source_npz": str(source_npz.relative_to(root)),
            "source_npz_sha256": _sha256(source_npz),
            "source_summary": str(summary_path.relative_to(root)),
            "source_summary_sha256": _sha256(summary_path),
            "method_contract": str(contract.relative_to(root)),
            "method_contract_sha256": _sha256(contract),
            "generator": str(generator.relative_to(root)),
            "generator_sha256": _sha256(generator),
            "manual_value_entry": False,
            "model_outcome_values": False,
            "c09_c10_reads": 0,
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    retained = [path for path in targets if path != manifest]
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(retained)),
        encoding="utf-8",
    )
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
