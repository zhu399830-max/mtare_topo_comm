#!/usr/bin/env python3
"""Freeze C07-C08 relational event threshold and compare pooled baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_gse_action_set_node_selection_v1 import main as pooled_evaluator_main


PASS = "PASS_GSE_RELATIONAL_EXIT_TRANSPORT_SELECTION_V1"
FAIL = "FAIL_GSE_RELATIONAL_EXIT_TRANSPORT_SELECTION_V1"


def augment_summary(summary: dict, baseline_macro_f1: float) -> dict:
    result = dict(summary)
    ensemble = result["ensemble"]
    gain = float(ensemble["decision_episode_macro_f1"] - baseline_macro_f1)
    base_pass = result.get("status") == "PASS_GSE_ACTION_SET_NODE_SELECTION_V1"
    gates = dict(result["gates"])
    gates["macro_f1_gain_over_pooled_at_least_0p05"] = gain >= 0.05
    seed_metrics = result["seed_at_ensemble_threshold"]
    gates["all_seed_precision_at_least_0p98"] = all(
        row["decision_trigger_precision"] >= 0.98 for row in seed_metrics.values()
    )
    gates["all_seed_episode_recall_at_least_0p25"] = all(
        row["decision_episode_recall"] >= 0.25 for row in seed_metrics.values()
    )
    passed = base_pass and all(gates.values())
    result.update({
        "schema_version": "gse_relational_exit_transport_selection_v1",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "pooled_baseline_macro_f1": float(baseline_macro_f1),
        "macro_f1_gain_over_pooled": gain,
        "gates": gates,
    })
    return result


def _plot(output: Path, summary: dict) -> None:
    ensemble = summary["ensemble"]
    event = ensemble["per_event"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].bar([0, 1], [summary["pooled_baseline_macro_f1"], ensemble["decision_episode_macro_f1"]], color=["#9c755f", "#4e79a7"])
    axes[0].set_xticks([0, 1], ["Pooled tokens", "Relational transport"])
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("Decision episode macro-F1")
    axes[0].set_title("A  Representation gain")
    labels = ["Aggregate", "Junction", "Terminal"]
    precision = [ensemble["decision_trigger_precision"], event["junction"]["precision"], event["terminal"]["precision"]]
    recall = [ensemble["decision_episode_recall"], event["junction"]["recall"], event["terminal"]["recall"]]
    x = range(3)
    axes[1].bar([value - 0.18 for value in x], precision, 0.36, label="precision", color="#59a14f")
    axes[1].bar([value + 0.18 for value in x], recall, 0.36, label="episode recall", color="#f28e2b")
    axes[1].axhline(0.99, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_title("B  Safe structural commits")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Relational exit-token transport capacity on C07–C08")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_relational_exit_transport_selection_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    arguments = sys.argv
    try:
        index = arguments.index("--baseline-summary")
        baseline_path = Path(arguments[index + 1]).resolve()
        del arguments[index:index + 2]
        output_index = arguments.index("--output-dir")
        output = Path(arguments[output_index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("relational selection wrapper arguments are incomplete") from exc
    pooled_evaluator_main()
    path = output / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    baseline_outer = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_macro = float(baseline_outer["selection"]["ensemble"]["decision_episode_macro_f1"])
    summary = augment_summary(summary, baseline_macro)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_relational_exit_transport_selection_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "macro_f1_gain": summary["macro_f1_gain_over_pooled"], "gates": summary["gates"]}, indent=2, sort_keys=True))
    return 0 if summary["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
