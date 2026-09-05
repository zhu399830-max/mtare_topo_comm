"""Deterministic manuscript tables for the sealed Gate-6 comparison."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.paper_single_robot_figures import (
    FAMILY_LABELS,
    METRIC_LABELS,
)
from mtare_topo.evaluation.stochastic_closed_loop import FAMILIES, METRICS


MECHANISM_FIELDS = (
    ("Verified re-anchor", "verified_reanchor_count_total", "verified_reanchor_case_count"),
    (
        "Frontier execution rejection",
        "frontier_execution_rejection_count_total",
        "frontier_execution_rejection_case_count",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite paper table value: {name}")
    return result


def _number(value: Any) -> str:
    return format(_finite(value, name="formatted value"), ".6g")


def _latex(value: Any) -> str:
    return str(value).replace("%", r"\%").replace("_", r"\_")


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_markdown(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    lines = [
        "| " + " | ".join(str(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]], *, alignment: str
) -> None:
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(_latex(value) for value in header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(_latex(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_triplet(
    output_dir: Path,
    stem: str,
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    alignment: str,
) -> list[Path]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    _write_csv(csv_path, header, rows)
    _write_markdown(md_path, header, rows)
    _write_latex(tex_path, header, rows, alignment=alignment)
    return [csv_path, md_path, tex_path]


def _family_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    summaries = analysis["corrected_main_analysis"]["family_summaries"]
    rows = []
    for metric in METRICS:
        for family in FAMILIES:
            item = summaries[family][metric]
            if int(item["n"]) != 30:
                raise ValueError("paper family table requires 30 cases per method")
            mean = _number(item["mean"])
            std = _number(item["sample_std"])
            rows.append([
                METRIC_LABELS[metric], FAMILY_LABELS[family], int(item["n"]),
                mean, std, f"{mean} ± {std}",
            ])
    return rows


def _effect_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    main = analysis["corrected_main_analysis"]["m1d_comparisons"]
    corrective = analysis["v5_vs_defective_v9"]["metrics"]
    rows = []
    for comparison, values in (
        ("V5 - Original M-TARE", main),
        ("V5 - Defective V9", corrective),
    ):
        for metric in METRICS:
            item = values[metric]
            relative = item.get(
                "relative_difference_fraction",
                item.get("relative_mean_difference_fraction"),
            )
            if relative is None or "holm_adjusted_p" not in item:
                raise ValueError(f"paper effect table lacks frozen relative/Holm result: {metric}")
            ci = item["paired_bootstrap_95_ci"]
            rows.append([
                comparison,
                METRIC_LABELS[metric],
                _number(item["mean_difference"]),
                _number(100.0 * _finite(relative, name=f"{comparison}:{metric}:relative")),
                _number(ci[0]),
                _number(ci[1]),
                _number(item["exact_two_sided_sign_flip_p"]),
                _number(item["holm_adjusted_p"]),
            ])
    return rows


def _mechanism_rows(mechanisms: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for label, total_field, case_field in MECHANISM_FIELDS:
        total, cases = int(mechanisms[total_field]), int(mechanisms[case_field])
        if total < 0 or not 0 <= cases <= 30:
            raise ValueError(f"invalid V5 mechanism count: {label}")
        rows.append([label, total, cases])
    both = int(mechanisms["case_count_with_both_correction_types"])
    if not 0 <= both <= 30:
        raise ValueError("invalid both-correction case count")
    rows.append(["Cases with both correction types", "—", both])
    outcomes = mechanisms.get("frontier_execution_outcomes")
    if not isinstance(outcomes, Mapping):
        raise ValueError("V5 mechanism outcomes are absent")
    for outcome in sorted(outcomes):
        count = int(outcomes[outcome])
        if count < 0:
            raise ValueError(f"negative V5 mechanism outcome: {outcome}")
        rows.append([f"Frontier outcome: {outcome}", count, "—"])
    return rows


def render_single_robot_paper_tables(
    analysis: Mapping[str, Any],
    mechanisms: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write CSV/Markdown/LaTeX tables without manually entered result values."""

    if analysis.get("schema_version") != "corrected_v5_stochastic_comparison_v1":
        raise ValueError("paper tables require corrected V5 comparison evidence")
    main = analysis["corrected_main_analysis"]
    if main.get("case_count") != 90 or main.get("block_count") != 10:
        raise ValueError("paper tables require exactly 90 cases and ten blocks")
    if int(mechanisms.get("completed_case_count", -1)) != 30:
        raise ValueError("paper mechanism table requires exactly 30 corrected V5 cases")
    required_provenance = (
        "source_audit_run", "source_audit_seal_sha256",
        "corrected_v5_run", "corrected_v5_seal_sha256",
    )
    if any(not provenance.get(field) for field in required_provenance):
        raise ValueError("paper table source provenance is incomplete")

    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    files.extend(_write_triplet(
        output_dir, "single_robot_family_summary",
        ("Metric", "Method", "n", "Mean", "Std.", "Mean ± Std."),
        _family_rows(analysis), alignment="llrrrr",
    ))
    files.extend(_write_triplet(
        output_dir, "single_robot_paired_effects",
        ("Comparison", "Metric", "Mean diff.", "Relative diff. (%)", "CI low", "CI high", "p", "Holm p"),
        _effect_rows(analysis), alignment="llrrrrrr",
    ))
    files.extend(_write_triplet(
        output_dir, "v5_mechanism_counts",
        ("Mechanism / outcome", "Total events", "Cases"),
        _mechanism_rows(mechanisms), alignment="lrr",
    ))

    return {
        "schema_version": "gate6_single_robot_paper_table_manifest_v1",
        "table_count": 3,
        "file_count": len(files),
        "source_schema_version": analysis["schema_version"],
        "case_count": 90,
        "block_count": 10,
        "family_case_counts": {family: 30 for family in FAMILIES},
        "manual_value_entry": False,
        "source_provenance": dict(provenance),
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files)
        ],
    }


__all__ = ["render_single_robot_paper_tables"]
