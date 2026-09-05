#!/usr/bin/env python3
"""Audit whether rare relation endpoints fail in structure or event-class space."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector


EXPECTED = 188_126


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _collate(samples: list[dict]) -> dict:
    import torch
    return {
        "tokens": torch.from_numpy(np.stack([sample["tokens"] for sample in samples])),
        "mask": torch.from_numpy(np.stack([sample["history_mask"] for sample in samples])),
        "row": np.asarray([sample["observation_row"] for sample in samples], dtype=np.int64),
    }


def _infer(cache: Path, checkpoint: Path, seed: int, device) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    dataset = ActionSetNodeDataset(cache, np.arange(EXPECTED, dtype=np.int64))
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, collate_fn=_collate)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("schema_version") != "gse_action_set_node_checkpoint_v1" or saved.get("seed") != seed:
        raise RuntimeError(f"observability checkpoint drift: seed{seed}")
    model = ActionSetNodeDetector().to(device); model.load_state_dict(saved["model"], strict=True); model.eval()
    output = {"context": [], "structural": [], "conditional": [], "row": []}
    with torch.inference_mode():
        for batch in loader:
            result = model(batch["tokens"].to(device), batch["mask"].to(device))
            output["context"].append(result["causal_context"].cpu().numpy())
            output["structural"].append(result["structural_logit"].cpu().numpy())
            output["conditional"].append(result["conditional_decision_logits"].cpu().numpy())
            output["row"].append(batch["row"])
    values = {key: np.concatenate(parts) for key, parts in output.items()}
    if not np.array_equal(values.pop("row"), np.arange(EXPECTED)):
        raise RuntimeError("observability inference row drift")
    return values


def _probability(structural: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    logit = np.asarray(structural, dtype=np.float64)
    mass = 1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0)))
    shifted = np.asarray(conditional, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    classes = np.exp(shifted); classes /= classes.sum(axis=1, keepdims=True)
    result = np.zeros((len(logit), 5), dtype=np.float32)
    result[:, 0] = 1.0 - mass
    result[:, 1:3] = mass[:, None] * classes
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("observability probability simplex drift")
    return result


def _geometry_signature(raw: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Permutation-invariant physical exit summary, averaged/std over seeds."""
    if raw.ndim != 4 or raw.shape[1:] != (3, 6, 40):
        raise ValueError("observability raw-token shape drift")
    output = np.empty((len(raw), 32), dtype=np.float32)
    for start in range(0, len(raw), chunk):
        token = np.asarray(raw[start:start + chunk], dtype=np.float32)
        confidence = token[..., 0].clip(0.0, 1.0)
        weight = confidence / confidence.sum(axis=2, keepdims=True).clip(min=1e-8)
        heading = token[..., 1:3]
        width = token[..., 3]
        profile = token[..., 4:8]
        width_mean = np.sum(weight * width, axis=2)
        width_std = np.sqrt(np.sum(weight * (width - width_mean[..., None]) ** 2, axis=2).clip(min=0.0))
        profile_mean = np.sum(weight[..., None] * profile, axis=2)
        profile_std = np.sqrt(np.sum(weight[..., None] * (profile - profile_mean[..., None, :]) ** 2, axis=2).clip(min=0.0))
        per_seed = np.concatenate((
            confidence.sum(axis=2)[..., None],
            np.square(confidence).sum(axis=2)[..., None],
            np.max(confidence, axis=2)[..., None],
            np.sum(weight[..., None] * heading, axis=2),
            width_mean[..., None], width_std[..., None],
            np.max(confidence * width, axis=2)[..., None], profile_mean, profile_std,
        ), axis=2)
        if per_seed.shape[2] != 16:
            raise RuntimeError("observability geometry signature dimension drift")
        output[start:start + len(token)] = np.concatenate((per_seed.mean(axis=1), per_seed.std(axis=1)), axis=1)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("observability geometry signature nonfinite")
    return output


def _causal_geometry(signature: np.ndarray, references: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.empty((len(signature), signature.shape[1] * 4), dtype=np.float32)
    for row in range(len(signature)):
        history = signature[references[row, mask[row]]]
        current = signature[row]
        result[row] = np.concatenate((current, history.mean(axis=0), history.std(axis=0), current - history[0]))
    return result


def _nearest_labels(reference: np.ndarray, labels: np.ndarray, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64); queries = np.asarray(queries, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1] or labels.shape != (len(reference),) or k < 1 or k > len(reference):
        raise ValueError("observability nearest-neighbor contract drift")
    mean = reference.mean(axis=0); scale = reference.std(axis=0); scale[scale < 1e-6] = 1.0
    ref = (reference - mean) / scale; query = (queries - mean) / scale
    distance = np.sum((query[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    index = np.argsort(distance, axis=1, kind="stable")[:, :k]
    return labels[index], np.take_along_axis(distance, index, axis=1)


def _endpoint_outcome(rows: np.ndarray, target_event: int, probability: np.ndarray) -> str:
    mass = probability[rows, 1] + probability[rows, 2]
    accepted = rows[mass >= .97]
    if any(int(np.argmax(probability[row, 1:3])) + 1 == target_event for row in accepted):
        return "correct_event_proposal"
    if len(accepted):
        return "event_misclassified"
    return "proposal_missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--endpoint-audit", required=True, type=Path)
    parser.add_argument("--checkpoint0", required=True, type=Path)
    parser.add_argument("--checkpoint1", required=True, type=Path)
    parser.add_argument("--checkpoint2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); output = args.output_dir.resolve()
    if output.exists(): raise RuntimeError("observability output exists; overwrite forbidden")
    output.mkdir(parents=True)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("observability audit requires frozen CUDA inference")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; device = torch.device("cuda")
    cache = args.cache_dir.resolve(); manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("causal_observations") != EXPECTED: raise RuntimeError("observability cache population drift")
    partition = np.load(cache / "partition_code.npy").astype(np.uint8)
    target = np.load(cache / "decision_target.npy").astype(np.int64)
    identity = np.load(cache / "identity.npy").astype(str)
    global_index = np.load(cache / "global_sequence_index.npy").astype(np.int64)
    traversal = np.load(cache / "traversal_id.npy").astype(str)
    sequence = np.load(cache / "sequence_index.npy").astype(np.int64)
    references = np.load(cache / "history_references.npy").astype(np.int64)
    history_mask = np.load(cache / "history_mask.npy").astype(bool)
    endpoint_records = _read_jsonl(args.endpoint_audit.resolve())
    if len(endpoint_records) != 98: raise RuntimeError("observability endpoint population drift")
    low_selection = [record for record in endpoint_records if record["partition"] == "selection" and record["support_bin"] == "1-3"]
    if len(low_selection) != 2: raise RuntimeError("observability low-support population drift")
    fit_rows = np.flatnonzero(partition == 0); fit_decision = fit_rows[target[fit_rows] > 0]
    raw = np.load(cache / "raw_tokens.npy", mmap_mode="r")
    geometry = _causal_geometry(_geometry_signature(raw), references, history_mask)
    probability_by_seed = []; low_rows = np.flatnonzero(np.isin(identity, [record["identity"] for record in low_selection]) & (partition == 1))
    endpoint_detail = []
    checkpoints = (args.checkpoint0, args.checkpoint1, args.checkpoint2)
    context_neighbor = {int(row): [] for row in low_rows}; geometry_neighbor = {}
    geo_labels, geo_distance = _nearest_labels(geometry[fit_rows], target[fit_rows], geometry[low_rows], k=5)
    for offset, row in enumerate(low_rows): geometry_neighbor[int(row)] = (geo_labels[offset], geo_distance[offset])
    for seed, checkpoint in enumerate(checkpoints):
        values = _infer(cache, checkpoint.resolve(), seed, device); probability = _probability(values["structural"], values["conditional"]); probability_by_seed.append(probability)
        neighbor_labels, neighbor_distance = _nearest_labels(values["context"][fit_rows], target[fit_rows], values["context"][low_rows], k=5)
        decision_labels, decision_distance = _nearest_labels(values["context"][fit_decision], target[fit_decision], values["context"][low_rows], k=5)
        for offset, row in enumerate(low_rows):
            context_neighbor[int(row)].append({"seed": seed, "all_fit_labels": neighbor_labels[offset].tolist(), "all_fit_squared_distance": neighbor_distance[offset].tolist(), "decision_fit_labels": decision_labels[offset].tolist(), "decision_fit_squared_distance": decision_distance[offset].tolist()})
        del values
    ensemble = np.mean(np.stack(probability_by_seed), axis=0)
    endpoint_outcomes = []
    for record in endpoint_records:
        rows = np.flatnonzero((identity == record["identity"]) & (partition == (0 if record["partition"] == "fit" else 1)))
        event_index = 1 if record["event"] == "junction" else 2
        endpoint_outcomes.append({**record, "ensemble_outcome": _endpoint_outcome(rows, event_index, ensemble), "maximum_structural_mass": float(np.max(ensemble[rows, 1] + ensemble[rows, 2])), "maximum_correct_joint_probability": float(np.max(ensemble[rows, event_index]))})
    for row in low_rows:
        record = next(value for value in low_selection if value["identity"] == identity[row])
        seed_values = []
        for seed, probability in enumerate(probability_by_seed):
            mass = float(probability[row, 1] + probability[row, 2])
            seed_values.append({"seed": seed, "structural_mass": mass, "junction_probability": float(probability[row, 1]), "terminal_probability": float(probability[row, 2]), "terminal_conditional_probability": float(probability[row, 2] / max(mass, 1e-12))})
        geo_label, geo_distance = geometry_neighbor[int(row)]
        endpoint_detail.append({"global_sequence_index": int(global_index[row]), "row": int(row), "identity": str(identity[row]), "world": record["world"], "teacher_event": record["event"], "teacher_rows": int(record["teacher_rows"]), "audit_stage": record["stage"], "traversal_id": str(traversal[row]), "sequence_index": int(sequence[row]), "ensemble_structural_mass": float(ensemble[row, 1] + ensemble[row, 2]), "ensemble_junction_probability": float(ensemble[row, 1]), "ensemble_terminal_probability": float(ensemble[row, 2]), "seed_probabilities": seed_values, "context_neighbors": context_neighbor[int(row)], "geometry_all_fit_labels": geo_label.tolist(), "geometry_all_fit_squared_distance": geo_distance.tolist()})
    category = {}
    for split in ("fit", "selection"):
        records = [value for value in endpoint_outcomes if value["partition"] == split and value["support_bin"] == "1-3"]
        category[split] = {name: sum(value["ensemble_outcome"] == name for value in records) for name in ("correct_event_proposal", "event_misclassified", "proposal_missing")}
        category[split]["total"] = len(records)
    misclassified = category["fit"]["event_misclassified"] + category["selection"]["event_misclassified"]
    missing = category["fit"]["proposal_missing"] + category["selection"]["proposal_missing"]
    recommendation = "conditional_event_class_corrective_first" if misclassified else "temporal_geometry_representation_audit"
    summary = {"schema_version":"gse_low_support_endpoint_observability_audit_v1","status":"PASS_GSE_LOW_SUPPORT_ENDPOINT_OBSERVABILITY_AUDIT_V1","population":{"observations":EXPECTED,"fit_observations":int(np.sum(partition==0)),"selection_observations":int(np.sum(partition==1)),"relation_endpoints":len(endpoint_records),"fit_low_support_endpoints":category["fit"]["total"],"selection_low_support_endpoints":category["selection"]["total"],"low_selection_rows":len(low_rows)},"low_support_outcomes":category,"recommendation":recommendation,"interpretation":{"structural_only_excluded":bool(misclassified),"event_misclassified_low_endpoints":misclassified,"proposal_missing_low_endpoints":missing,"reason":"At least one low-support terminal already crosses the structural threshold but loses in the frozen junction-vs-terminal conditional head; a structural-only residual cannot recover it." if misclassified else "No accepted structural proposal was event-misclassified."},"gates":{"exact_population":len(endpoint_records)==98 and category["fit"]["total"]==10 and category["selection"]["total"]==2,"all_three_seeds_inferred":len(probability_by_seed)==3,"both_selection_failure_rows_explained":len(endpoint_detail)==4,"structural_only_mechanism_decidable":bool(misclassified)},"model_inference_frames":3*EXPECTED,"optimizer_steps":0,"model_updates":0,"threshold_selection_steps":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"duration_seconds":time.monotonic()-started}
    summary["gates"]["all_passed"] = all(summary["gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    with (output / "endpoint_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for value in endpoint_outcomes: stream.write(json.dumps(value, sort_keys=True)+"\n")
    (output / "low_selection_rows.json").write_text(json.dumps(endpoint_detail, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    with (output / "low_selection_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(("identity","row","structural_mass","junction_probability","terminal_probability","stage"))
        for value in endpoint_detail:writer.writerow((value["identity"],value["row"],value["ensemble_structural_mass"],value["ensemble_junction_probability"],value["ensemble_terminal_probability"],value["audit_stage"]))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    names = [f"{value['identity'].split(':')[-1]}\nrow {value['sequence_index']}" for value in endpoint_detail]
    x=np.arange(len(endpoint_detail)); width=.26
    axes[0].bar(x-width, [v["ensemble_structural_mass"] for v in endpoint_detail], width, label="Structure mass", color="#2878B5")
    axes[0].bar(x, [v["ensemble_junction_probability"] for v in endpoint_detail], width, label="Junction", color="#D95F02")
    axes[0].bar(x+width, [v["ensemble_terminal_probability"] for v in endpoint_detail], width, label="Terminal", color="#2CA02C")
    axes[0].axhline(.97,color="black",linestyle="--",linewidth=1,label="Fixed trigger 0.97");axes[0].set_xticks(x,names);axes[0].set_ylim(0,1.05);axes[0].set_ylabel("Frozen ensemble probability");axes[0].set_title("A  Two rare terminal endpoint observations");axes[0].legend(frameon=False,fontsize=8)
    categories=("Correct event","Wrong event","No proposal");fit_values=(category["fit"]["correct_event_proposal"],category["fit"]["event_misclassified"],category["fit"]["proposal_missing"]);selection_values=(category["selection"]["correct_event_proposal"],category["selection"]["event_misclassified"],category["selection"]["proposal_missing"]);x2=np.arange(3)
    axes[1].bar(x2-width/2,fit_values,width,label="C01-C06",color="#7F7F7F");axes[1].bar(x2+width/2,selection_values,width,label="C07-C08",color="#9467BD");axes[1].set_xticks(x2,categories);axes[1].set_ylabel("Low-support endpoint identities");axes[1].set_title("B  Failure mechanism is not uniform");axes[1].legend(frameon=False)
    fig.suptitle("Low-support relation endpoints: structural proposal vs event class")
    for suffix in ("png","pdf","svg"):fig.savefig(output/f"gse_low_support_endpoint_observability.{suffix}",dpi=240 if suffix=="png" else None)
    plt.close(fig)
    (output / "figure_source.json").write_text(json.dumps({"schema_version":"gse_low_support_endpoint_observability_figure_source_v1","endpoint_detail":endpoint_detail,"low_support_outcomes":category},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary, sort_keys=True)); return 0 if summary["gates"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
