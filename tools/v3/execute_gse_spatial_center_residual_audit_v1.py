#!/usr/bin/env python3
"""Decompose the sealed C07-C08 spatial event-center consistency tail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mtare_topo.evaluation.gse_spatial_center_residual import (
    cross_traversal_pairs,
    pair_distance_metrics,
)
from mtare_topo.teacher.gse_event_center_teacher import local_event_center_vectors


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key not in ("identity_rows", "pair_distance_m")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--baseline-projection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)

    with np.load(args.ensemble.resolve(), allow_pickle=False) as archive:
        ensemble = {key: archive[key] for key in archive.files}
    rows = ensemble["observation_row"].astype(np.int64)
    predicted_vector = ensemble["predicted_local_vector_m"].astype(np.float64)
    predicted_center = ensemble["predicted_center_xyz_m"].astype(np.float64)
    target_vector = ensemble["target_local_vector_m"].astype(np.float64)
    target_center = ensemble["target_center_xyz_m"].astype(np.float64)
    if len(rows) != 8839:
        raise RuntimeError("residual audit selection row count drift")
    with np.load(args.teacher.resolve(), allow_pickle=False) as archive:
        identity_all = archive["identity"].astype(str)
        event_all = archive["event"].astype(str)
        tangent = archive["route_tangent_xyz"].astype(np.float64)
        objective = archive["objective_center_xyz_m"].astype(np.float64)
        longitudinal_target = archive["signed_center_offset_m"].astype(np.float64)
        oracle = archive["oracle_longitudinal_center_xyz_m"].astype(np.float64)
        valid = archive["valid_mask"].astype(bool)
    traversal_all = np.load(args.action_cache.resolve() / "traversal_id.npy").astype(str)
    sensor_all = oracle - longitudinal_target[:, None] * tangent
    basis_all = local_event_center_vectors(sensor_all, objective, tangent, valid)["route_local_basis"].astype(np.float64)
    identity = identity_all[rows]; traversal = traversal_all[rows]; event = event_all[rows]
    sensor = sensor_all[rows]; basis = basis_all[rows]
    if not np.array_equal(target_center, objective[rows]) or set(event.tolist()) != {"junction", "terminal"}:
        raise RuntimeError("residual audit Teacher alignment drift")
    pairs = cross_traversal_pairs(identity, traversal)
    if len(pairs.pair_index) != 144533 or len(pairs.identity_names) != 272:
        raise RuntimeError("residual audit pair population drift")

    def centers(vector: np.ndarray) -> np.ndarray:
        return sensor + np.einsum("nij,ni->nj", basis, vector)

    with np.load(args.baseline_projection.resolve(), allow_pickle=False) as archive:
        baseline_longitudinal = archive["predicted_offset_m"][rows].astype(np.float64)
    baseline_vector = np.column_stack((baseline_longitudinal, np.zeros((len(rows), 2))))
    variants = {
        "scalar_zero_transverse": centers(baseline_vector),
        "spatial_full": predicted_center,
        "predicted_longitudinal_oracle_transverse": centers(np.column_stack((predicted_vector[:, 0], target_vector[:, 1:]))),
        "oracle_longitudinal_predicted_transverse": centers(np.column_stack((target_vector[:, 0], predicted_vector[:, 1:]))),
        "oracle_all": target_center,
    }
    seed_vectors = []
    seed_centers = []
    for seed in (0, 1, 2):
        with np.load(args.training_root.resolve() / f"seed{seed}/selection_outputs.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["observation_row"], rows):
                raise RuntimeError("residual audit seed identity drift")
            seed_vectors.append(archive["predicted_local_vector_m"].astype(np.float64))
            seed_centers.append(archive["predicted_center_xyz_m"].astype(np.float64))
    seed_vector_stack = np.stack(seed_vectors)
    seed_center_stack = np.stack(seed_centers)
    median_vector = np.median(seed_vector_stack, axis=0)
    variants["diagnostic_coordinate_median"] = centers(median_vector)
    seed_pairwise = np.linalg.norm(seed_center_stack[:, None] - seed_center_stack[None, :], axis=-1).sum(axis=1)
    medoid_seed = np.argmin(seed_pairwise, axis=0)
    variants["diagnostic_per_row_medoid"] = seed_center_stack[medoid_seed, np.arange(len(rows))]

    variant_metrics = {name: pair_distance_metrics(value, pairs) for name, value in variants.items()}
    event_metrics = {}
    pair_name_to_code = {name: code for code, name in enumerate(pairs.identity_names)}
    for event_name in ("junction", "terminal"):
        identity_names = {
            name for name in set(identity[event == event_name].tolist())
            if name in pair_name_to_code
        }
        identity_codes = np.asarray([pair_name_to_code[name] for name in identity_names], dtype=np.int32)
        pair_mask = np.isin(pairs.identity_code, identity_codes)
        compact_names = tuple(sorted(identity_names))
        remap = {pair_name_to_code[name]: index for index, name in enumerate(compact_names)}
        from mtare_topo.evaluation.gse_spatial_center_residual import CrossTraversalPairs
        local_pairs = CrossTraversalPairs(
            pairs.pair_index[pair_mask],
            np.asarray([remap[int(value)] for value in pairs.identity_code[pair_mask]], dtype=np.int32),
            compact_names,
        )
        event_metrics[event_name] = {
            name: _public(pair_distance_metrics(value, local_pairs))
            for name, value in variants.items()
            if not name.startswith("diagnostic_")
        }

    disagreement = np.mean(np.linalg.norm(seed_center_stack - predicted_center[None], axis=2), axis=0)
    full_distance = np.asarray(variant_metrics["spatial_full"]["pair_distance_m"])
    pair_disagreement = disagreement[pairs.pair_index].mean(axis=1)
    disagreement_summary = {
        "row_mean_m": float(np.mean(disagreement)),
        "row_p95_m": float(np.quantile(disagreement, .95)),
        "within4_pair_mean_m": float(np.mean(pair_disagreement[full_distance <= 4.0])),
        "outside4_pair_mean_m": float(np.mean(pair_disagreement[full_distance > 4.0])),
    }
    identity_records = []
    full_rows = {row["identity"]: row for row in variant_metrics["spatial_full"]["identity_rows"]}
    long_rows = {row["identity"]: row for row in variant_metrics["predicted_longitudinal_oracle_transverse"]["identity_rows"]}
    transverse_rows = {row["identity"]: row for row in variant_metrics["oracle_longitudinal_predicted_transverse"]["identity_rows"]}
    baseline_rows = {row["identity"]: row for row in variant_metrics["scalar_zero_transverse"]["identity_rows"]}
    for name in pairs.identity_names:
        row_mask = identity == name
        identity_records.append({
            "identity": name,
            "event": str(np.unique(event[row_mask]).item()),
            "rows": int(row_mask.sum()),
            "traversals": int(len(np.unique(traversal[row_mask]))),
            "pair_count": full_rows[name]["pair_count"],
            "baseline_within4": baseline_rows[name]["within_fraction"],
            "full_within4": full_rows[name]["within_fraction"],
            "predicted_long_oracle_transverse_within4": long_rows[name]["within_fraction"],
            "oracle_long_predicted_transverse_within4": transverse_rows[name]["within_fraction"],
            "full_mean_distance_m": full_rows[name]["mean_distance_m"],
            "mean_seed_disagreement_m": float(np.mean(disagreement[row_mask])),
        })
    identity_records.sort(key=lambda row: (row["full_within4"], -row["pair_count"], row["identity"]))
    with (output / "identity_residuals.jsonl").open("w", encoding="utf-8") as stream:
        for row in identity_records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    component_mae = {
        "baseline": np.mean(np.abs(baseline_vector - target_vector), axis=0).tolist(),
        "spatial": np.mean(np.abs(predicted_vector - target_vector), axis=0).tolist(),
    }
    summary = {
        "schema_version": "gse_spatial_center_residual_audit_v1",
        "selection_rows": len(rows), "identities": len(pairs.identity_names),
        "cross_traversal_pairs": len(pairs.pair_index),
        "variant_metrics": {name: _public(value) for name, value in variant_metrics.items()},
        "event_metrics": event_metrics,
        "component_mae_m": component_mae,
        "seed_disagreement": disagreement_summary,
        "worst_20_identities": identity_records[:20],
        "diagnostic_only_aggregation": ["diagnostic_coordinate_median", "diagnostic_per_row_medoid"],
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    primary = ["scalar_zero_transverse", "spatial_full", "predicted_longitudinal_oracle_transverse", "oracle_longitudinal_predicted_transverse", "oracle_all"]
    labels = ["scalar", "spatial", "pred-long\noracle-trans", "oracle-long\npred-trans", "oracle"]
    axes[0, 0].bar(labels, [variant_metrics[name]["identity_macro_within_fraction"] for name in primary])
    axes[0, 0].axhline(0.6239515464, color="red", linestyle="--", linewidth=1, label="frozen gate")
    axes[0, 0].set_ylim(0, 1.02); axes[0, 0].set_ylabel("identity-macro within 4 m"); axes[0, 0].legend()
    x = np.arange(3); width = .35
    axes[0, 1].bar(x - width / 2, component_mae["baseline"], width, label="scalar")
    axes[0, 1].bar(x + width / 2, component_mae["spatial"], width, label="spatial")
    axes[0, 1].set_xticks(x, ["forward", "lateral", "up"]); axes[0, 1].set_ylabel("MAE (m)"); axes[0, 1].legend()
    axes[1, 0].hist(variant_metrics["scalar_zero_transverse"]["pair_distance_m"], bins=60, range=(0, 12), alpha=.55, label="scalar")
    axes[1, 0].hist(full_distance, bins=60, range=(0, 12), alpha=.55, label="spatial")
    axes[1, 0].axvline(4, color="black", linestyle="--", linewidth=1); axes[1, 0].set_xlabel("same-identity cross-traversal center distance (m)"); axes[1, 0].set_ylabel("pairs"); axes[1, 0].legend()
    baseline_identity = np.asarray([baseline_rows[name]["within_fraction"] for name in pairs.identity_names])
    full_identity = np.asarray([full_rows[name]["within_fraction"] for name in pairs.identity_names])
    color = np.asarray([0 if np.unique(event[identity == name]).item() == "junction" else 1 for name in pairs.identity_names])
    axes[1, 1].scatter(baseline_identity[color == 0], full_identity[color == 0], s=12, alpha=.65, label="junction")
    axes[1, 1].scatter(baseline_identity[color == 1], full_identity[color == 1], s=12, alpha=.65, label="terminal")
    axes[1, 1].plot([0, 1], [0, 1], color="black", linewidth=1); axes[1, 1].set_xlabel("scalar within-4m by identity"); axes[1, 1].set_ylabel("spatial within-4m by identity"); axes[1, 1].legend()
    figure.suptitle("GSE-Graph spatial event-center residual audit — sealed C07–C08")
    figure.tight_layout(); figure.savefig(output / "spatial_center_residual_audit.png", dpi=200); plt.close(figure)
    manifest = {"schema_version": "gse_spatial_center_residual_figure_v1", "figure": "spatial_center_residual_audit.png", "sha256": _sha(output / "spatial_center_residual_audit.png")}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selection_rows": len(rows), "pairs": len(pairs.pair_index), "summary": summary["variant_metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
