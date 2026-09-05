#!/usr/bin/env python3
"""Publish fixed-world qualitative GSE topology examples from sealed PASS evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate4_20260824_gse_offline_topology_validation_v1_seed0"
EXPECTED_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V1"
FIGURE_ID = "gse_topology_examples"
WORLDS = (
    "S01_flat_tree_small_C09",
    "S06_3d_branch_medium_C09",
    "S10_3d_complex_C09",
)
METHODS = (
    "exit_only_rule_graph",
    "nonlearning_geometry_rule_graph",
    "gse_rule_association_ablation",
    "gse_learned_association",
)
LABELS = ("Exit-only + rule", "Geometry + rule", "GSE + rule", "GSE-Graph")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def publish(run_dir: Path, destination: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    run_dir.relative_to(root)
    destination.relative_to(root)
    state = load_json(run_dir / "RUN_STATE.json")
    runner = load_json(run_dir / "metrics/summary.json")
    overall = load_json(run_dir / "artifacts/offline_topology/summary.json")
    if (
        run_dir.name != EXPECTED_RUN_ID
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or runner.get("overall_status") != EXPECTED_STATUS
        or overall.get("overall_status") != EXPECTED_STATUS
        or overall.get("scientific_gate", {}).get("passed") is not True
        or overall.get("strict_test_worlds_read") != 0
        or overall.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("topology examples require the sealed offline validation PASS")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected

    bundles: dict[tuple[str, str], dict] = {}
    source_files = [run_dir / "RUN_STATE.json", run_dir / "metrics/summary.json", run_dir / "artifacts/offline_topology/summary.json"]
    for method in METHODS:
        for world in WORLDS:
            world_dir = run_dir / f"artifacts/offline_topology/{method}/selected/seed0/{world}"
            nodes_path = world_dir / "nodes.jsonl"
            edges_path = world_dir / "edges.jsonl"
            summary_path = world_dir / "summary.json"
            source_files.extend((nodes_path, edges_path, summary_path))
            summary = load_json(summary_path)
            nodes = _jsonl(nodes_path)
            raw_edges = _jsonl(edges_path)
            if summary.get("association_mode") not in {"learned", "rule"}:
                raise RuntimeError("selected topology example lacks association provenance")
            retained = {int(value) for value in summary["collapsed_graph"]["node_ids"]}
            nodes = [node for node in nodes if int(node["id"]) in retained]
            edges = [
                {"id": index, "from": int(edge[0]), "to": int(edge[1])}
                for index, edge in enumerate(summary["collapsed_graph"]["edges"])
            ]
            if any(int(edge["from"]) not in retained or int(edge["to"]) not in retained for edge in edges):
                raise RuntimeError("collapsed qualitative edge references a suppressed node")
            bundles[(method, world)] = {"nodes": nodes, "edges": edges, "raw_edge_count": len(raw_edges), "summary": summary}
    for path in source_files:
        relative = str(path.relative_to(root))
        if sealed.get(relative) != _sha256(path):
            raise RuntimeError(f"qualitative source is absent from the run seal: {relative}")

    names = tuple(f"{FIGURE_ID}.{suffix}" for suffix in ("png", "pdf", "svg", "csv")) + (
        f"{FIGURE_ID}_source.json",
        f"{FIGURE_ID}_provenance.json",
        f"{FIGURE_ID}_sha256.txt",
    )
    targets = [destination / name for name in names]
    if any(path.exists() for path in targets):
        raise RuntimeError("topology example destination exists; refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / f"{FIGURE_ID}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("world", "method", "entity", "id", "from", "to", "x_m", "y_m", "z_m", "mapped_teacher_identity", "is_correct_edge"))
        for world in WORLDS:
            teacher = bundles[(METHODS[-1], world)]["summary"]
            teacher_xyz = teacher["teacher_node_xyz_m"]
            for identity, xyz in sorted(teacher_xyz.items()):
                writer.writerow((world, "teacher", "node", identity, "", "", *xyz, identity, ""))
            for first, second in teacher["teacher_graph"]["edges"]:
                writer.writerow((world, "teacher", "edge", "", first, second, "", "", "", "", True))
            for method in METHODS:
                bundle = bundles[(method, world)]
                mapping = {int(key): value for key, value in bundle["summary"]["predicted_to_teacher"].items()}
                teacher_edges = {frozenset(edge) for edge in bundle["summary"]["teacher_graph"]["edges"]}
                for node in bundle["nodes"]:
                    writer.writerow((world, method, "node", node["id"], "", "", *node["xyz_m"], mapping.get(int(node["id"]), ""), ""))
                for edge in bundle["edges"]:
                    endpoints = (int(edge["from"]), int(edge["to"]))
                    mapped = frozenset(mapping[value] for value in endpoints if value in mapping)
                    correct = len(mapped) == 2 and mapped in teacher_edges
                    writer.writerow((world, method, "edge", edge["id"], *endpoints, "", "", "", "", correct))

    figure, axes = plt.subplots(len(WORLDS), len(METHODS), figsize=(13.2, 8.4), constrained_layout=True)
    for row, world in enumerate(WORLDS):
        all_xyz = []
        teacher_summary = bundles[(METHODS[-1], world)]["summary"]
        teacher_xyz = {key: np.asarray(value, dtype=np.float64) for key, value in teacher_summary["teacher_node_xyz_m"].items()}
        all_xyz.extend(teacher_xyz.values())
        for method in METHODS:
            all_xyz.extend(np.asarray(node["xyz_m"], dtype=np.float64) for node in bundles[(method, world)]["nodes"])
        points = np.stack(all_xyz)
        low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
        pad = np.maximum((high - low) * 0.05, 1.0)
        z_low, z_high = float(points[:, 2].min()), float(points[:, 2].max())
        if abs(z_high - z_low) < 1e-9:
            z_high = z_low + 1.0
        for column, (method, label) in enumerate(zip(METHODS, LABELS, strict=True)):
            axis = axes[row, column]
            bundle = bundles[(method, world)]
            summary = bundle["summary"]
            mapping = {int(key): value for key, value in summary["predicted_to_teacher"].items()}
            teacher_edges = {frozenset(edge) for edge in summary["teacher_graph"]["edges"]}
            for first, second in summary["teacher_graph"]["edges"]:
                a, b = teacher_xyz[first], teacher_xyz[second]
                axis.plot((a[0], b[0]), (a[1], b[1]), color="#B8C0C8", linewidth=1.2, linestyle="--", zorder=1)
            node_by_id = {int(node["id"]): np.asarray(node["xyz_m"], dtype=np.float64) for node in bundle["nodes"]}
            for edge in bundle["edges"]:
                first, second = int(edge["from"]), int(edge["to"])
                mapped = frozenset(mapping[value] for value in (first, second) if value in mapping)
                correct = len(mapped) == 2 and mapped in teacher_edges
                a, b = node_by_id[first], node_by_id[second]
                axis.plot((a[0], b[0]), (a[1], b[1]), color="#2A9D8F" if correct else "#E76F51", linewidth=1.55, zorder=2)
            mapped_nodes = [node for node in bundle["nodes"] if int(node["id"]) in mapping]
            unmapped_nodes = [node for node in bundle["nodes"] if int(node["id"]) not in mapping]
            if mapped_nodes:
                xyz = np.asarray([node["xyz_m"] for node in mapped_nodes], dtype=np.float64)
                axis.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], cmap="viridis", vmin=z_low, vmax=z_high, s=18, edgecolor="white", linewidth=0.35, zorder=3)
            if unmapped_nodes:
                xyz = np.asarray([node["xyz_m"] for node in unmapped_nodes], dtype=np.float64)
                axis.scatter(xyz[:, 0], xyz[:, 1], marker="x", color="#E76F51", s=22, linewidth=1.0, zorder=4)
            axis.set_xlim(low[0] - pad[0], high[0] + pad[0])
            axis.set_ylim(low[1] - pad[1], high[1] + pad[1])
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.16)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(world.split("_C09")[0] + "\nY (m)")
            if row == len(WORLDS) - 1:
                axis.set_xlabel("X (m)")
        color_scale = plt.cm.ScalarMappable(norm=Normalize(vmin=z_low, vmax=z_high), cmap="viridis")
        figure.colorbar(color_scale, ax=axes[row, :].tolist(), shrink=0.58, pad=0.01, label="Node z (m)")
    figure.suptitle("Fixed validation worlds: dashed teacher, green correct edge, red error, × unmapped node", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{FIGURE_ID}.{suffix}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    write_json(
        destination / f"{FIGURE_ID}_source.json",
        {
            "worlds": list(WORLDS),
            "methods": list(METHODS),
            "seed": 0,
            "selection": "fixed S01/S06/S10 scale progression and seed0; declared before results",
            "source_files": {str(path.relative_to(run_dir)): _sha256(path) for path in source_files},
        },
    )
    write_json(
        destination / f"{FIGURE_ID}_provenance.json",
        {
            "schema_version": "gse_paper_figure_provenance_v1",
            "figure_id": FIGURE_ID,
            "source_run": str(run_dir.relative_to(root)),
            "source_seal": str(seal.relative_to(root)),
            "source_seal_sha256": _sha256(seal),
            "selection_rule": "S01/S06/S10 and seed0 fixed by topology scale before reading outcomes; every method shown",
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "manual_value_entry": False,
            "generator": str(Path(__file__).resolve().relative_to(root)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
        },
    )
    manifest = destination / f"{FIGURE_ID}_sha256.txt"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in sorted(targets) if path != manifest),
        encoding="utf-8",
    )
    return {"figure_id": FIGURE_ID, "published_files": len(targets), "manifest_sha256": _sha256(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "docs/figures/gse_graph")
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
