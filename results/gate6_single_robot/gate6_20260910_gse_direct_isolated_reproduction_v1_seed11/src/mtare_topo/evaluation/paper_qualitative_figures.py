"""Outcome-blind trajectory and online-graph panels for Gate-6 paper evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIXED_BLOCKS = ("garage_env11", "tunnel_env11")
FIXED_CHECKPOINT_SEED = 0
FIXED_ORIGINAL_REPEAT = 0
SELECTION_RULE = (
    "For each world, use environment seed 11; select original-M-TARE repeat 0 "
    "and the M1D/V5 case with checkpoint seed 0. The rule is identity-only and "
    "does not read coverage, travel, latency, graph size or outcome."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_xy(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"invalid qualitative coordinate: {name}")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"non-finite qualitative coordinate: {name}")
    return x, y


def select_fixed_qualitative_cases(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_by_id = {str(item["case"]["case_id"]): item for item in source_summaries}
    corrected_by_id = {str(item["case"]["case_id"]): item for item in corrected_summaries}
    if len(source_by_id) != len(source_summaries) or len(corrected_by_id) != len(corrected_summaries):
        raise ValueError("qualitative source case identities are duplicated")
    selections = []
    for block_id in FIXED_BLOCKS:
        originals = [
            item for item in source_summaries
            if item["case"]["block_id"] == block_id
            and item["case"]["method_family"] == "original_mtare"
            and item["case"]["execution_repeat"] == FIXED_ORIGINAL_REPEAT
        ]
        defective = [
            item for item in source_summaries
            if item["case"]["block_id"] == block_id
            and item["case"]["method_family"] == "m1d_topology"
            and item["case"]["checkpoint_seed"] == FIXED_CHECKPOINT_SEED
        ]
        if len(originals) != 1 or len(defective) != 1:
            raise ValueError(f"fixed qualitative selection is not unique: {block_id}")
        corrected = corrected_by_id.get(str(defective[0]["case"]["case_id"]))
        if corrected is None:
            raise ValueError(f"fixed qualitative corrected pair is missing: {block_id}")
        for field in (
            "case_id", "block_id", "world", "environment_seed", "checkpoint_seed",
            "execution_repeat", "runtime_sec",
        ):
            if corrected["case"].get(field) != defective[0]["case"].get(field):
                raise ValueError(f"fixed qualitative corrected identity drift: {block_id}: {field}")
        selections.append({
            "block_id": block_id,
            "world": originals[0]["case"]["world"],
            "original": originals[0],
            "defective_v9": defective[0],
            "corrected_v5": corrected,
        })
    return selections


def _parse_seal(root: Path, identity: Mapping[str, Any]) -> dict[str, str]:
    run = (root / str(identity["run"])).resolve()
    run.relative_to(root)
    seal = run / "artifacts/evidence_sha256.txt"
    if _sha256(seal) != identity["seal_sha256"]:
        raise ValueError(f"qualitative source seal identity drift: {run.name}")
    entries: dict[str, str] = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries:
            raise ValueError(f"duplicate qualitative seal entry: {relative}")
        entries[relative] = digest
    return entries


def _bound_path(root: Path, path: Path, entries: Mapping[str, str]) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root)
    relative = resolved.relative_to(root).as_posix()
    if entries.get(relative) != _sha256(resolved):
        raise ValueError(f"qualitative evidence is not seal-bound: {relative}")
    return resolved


def _case_files(
    root: Path,
    item: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    entries: Mapping[str, str],
    *,
    require_graph: bool,
) -> dict[str, Path]:
    summary_relative = manifest_row.get("summary_path", manifest_row.get("path"))
    if not isinstance(summary_relative, str):
        raise ValueError("qualitative summary path is missing from source manifest")
    summary_path = (root / summary_relative).resolve()
    case_dir = summary_path.parent
    trajectory = _bound_path(root, case_dir / "evidence/trajectory.jsonl", entries)
    result = {"trajectory": trajectory}
    if require_graph:
        relative = item.get("planner_evidence", {}).get("topology_snapshot")
        if not isinstance(relative, str):
            raise ValueError("qualitative topology snapshot identity is missing")
        result["snapshot"] = _bound_path(root, case_dir / relative, entries)
    return result


def _trajectory(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            points.append(_finite_xy(row.get("xyz_m"), name=path.name))
    if len(points) < 2:
        raise ValueError(f"qualitative trajectory is incomplete: {path}")
    return points


def _graph(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    graph = value.get("runtime", {}).get("graph", {})
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes:
        raise ValueError(f"qualitative graph is incomplete: {path}")
    ids = [int(node["id"]) for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"qualitative graph node identities are duplicated: {path}")
    return nodes, edges


def _plot_case(
    axis: Any, item: Mapping[str, Any], files: Mapping[str, Path], *, graph: bool, label: str
) -> None:
    points = _trajectory(files["trajectory"])
    axis.plot([p[0] for p in points], [p[1] for p in points], color="#475569", linewidth=1.2)
    axis.scatter(*points[0], s=45, color="#16a34a", marker="o", zorder=5, label="start")
    axis.scatter(*points[-1], s=48, color="#dc2626", marker="X", zorder=5, label="end")
    if graph:
        nodes, edges = _graph(files["snapshot"])
        by_id = {int(node["id"]): node for node in nodes}
        for edge in edges:
            left, right = by_id.get(int(edge["from"])), by_id.get(int(edge["to"]))
            if left is None or right is None:
                raise ValueError("qualitative graph edge endpoint is missing")
            x0, y0 = _finite_xy(left["xyz_m"], name="graph edge from")
            x1, y1 = _finite_xy(right["xyz_m"], name="graph edge to")
            axis.plot([x0, x1], [y0, y1], color="#16a34a", linewidth=1.8, alpha=0.8)
        anchor = [node for node in nodes if node.get("node_kind") == "anchor"]
        structural = [node for node in nodes if node.get("node_kind") != "anchor"]
        for group, marker, color, label in (
            (anchor, "s", "#7c3aed", "anchor node"),
            (structural, "o", "#0284c7", "structural node"),
        ):
            if group:
                xy = [_finite_xy(node["xyz_m"], name="graph node") for node in group]
                axis.scatter(
                    [p[0] for p in xy], [p[1] for p in xy], s=22,
                    marker=marker, color=color, edgecolor="white", linewidth=0.35,
                    zorder=4, label=label,
                )
    metrics = item["metrics"]
    axis.set_title(
        f"{label}\n"
        f"volume={float(metrics['final_explored_volume_m3']):.1f} m³, "
        f"travel={float(metrics['traveling_distance_m']):.1f} m",
        fontsize=10,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    axis.set_xlabel("x (m)")


def _save(figure: Any, stem: Path) -> list[Path]:
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    figure.savefig(
        png, dpi=220, bbox_inches="tight",
        metadata={"Software": "mtare_topo Gate-6 deterministic qualitative renderer"},
    )
    figure.savefig(
        pdf, bbox_inches="tight",
        metadata={"Creator": "mtare_topo", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return [png, pdf]


def render_fixed_qualitative_figures(
    source_summaries: Sequence[Mapping[str, Any]],
    corrected_summaries: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    corrected_manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selections = select_fixed_qualitative_cases(source_summaries, corrected_summaries)
    source_rows = {row["case_id"]: row for row in source_manifest["cases"]}
    corrected_rows = {row["case_id"]: row for row in corrected_manifest["cases"]}
    source_seals = {
        name: _parse_seal(root, identity)
        for name, identity in source_manifest["source_runs"].items()
    }
    corrected_entries = _parse_seal(root, {
        "run": corrected_manifest["run"], "seal_sha256": corrected_manifest["seal_sha256"],
    })
    files: list[Path] = []
    selected_rows = []
    for selected in selections:
        original, defective, corrected = (
            selected["original"], selected["defective_v9"], selected["corrected_v5"]
        )
        original_row = source_rows[original["case"]["case_id"]]
        defective_row = source_rows[defective["case"]["case_id"]]
        corrected_row = corrected_rows[corrected["case"]["case_id"]]
        original_files = _case_files(
            root, original, original_row, source_seals[original_row["source"]], require_graph=False
        )
        defective_files = _case_files(
            root, defective, defective_row, source_seals[defective_row["source"]], require_graph=True
        )
        corrected_files = _case_files(
            root, corrected, corrected_row, corrected_entries, require_graph=True
        )
        figure, axes = plt.subplots(
            1, 3, figsize=(14.5, 4.9), constrained_layout=True, sharex=True, sharey=True
        )
        _plot_case(axes[0], original, original_files, graph=False, label="Original M-TARE")
        _plot_case(axes[1], defective, defective_files, graph=True, label="Defective V9")
        _plot_case(axes[2], corrected, corrected_files, graph=True, label="Corrected V5")
        axes[0].set_ylabel("y (m)")
        figure.suptitle(
            f"Fixed qualitative block: {selected['block_id']} "
            f"(repeat 0 / checkpoint 0; identity-only selection)"
        )
        handles, labels = axes[2].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        files.extend(_save(figure, output_dir / f"{selected['world']}_fixed_trajectory_topology"))
        selected_rows.append({
            "block_id": selected["block_id"],
            "original_case_id": original["case"]["case_id"],
            "defective_v9_case_id": defective["case"]["case_id"],
            "corrected_v5_case_id": corrected["case"]["case_id"],
        })
    if Counter(path.suffix for path in files) != Counter({".png": 2, ".pdf": 2}):
        raise ValueError("qualitative paper figure output quota drift")
    return {
        "schema_version": "gate6_fixed_qualitative_figure_manifest_v1",
        "selection_rule": SELECTION_RULE,
        "selection_uses_performance_outcomes": False,
        "figure_count": 2,
        "file_count": 4,
        "selected_cases": selected_rows,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files)
        ],
    }


__all__ = ["render_fixed_qualitative_figures", "select_fixed_qualitative_cases"]
