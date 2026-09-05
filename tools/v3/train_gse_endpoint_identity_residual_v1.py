#!/usr/bin/env python3
"""Train one zero-initialized identity-balanced endpoint residual seed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_mass_triggers, extract_decision_mass_triggers,
)
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_endpoint_identity_residual import (
    EndpointIdentityStructuralResidual, identity_balanced_endpoint_mil_loss,
)


EXPECTED = 188_126


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _collate(samples: list[dict]) -> dict:
    import torch
    return {
        "tokens": torch.from_numpy(np.stack([sample["tokens"] for sample in samples])),
        "history_mask": torch.from_numpy(np.stack([sample["history_mask"] for sample in samples])),
        "observation_row": torch.from_numpy(np.asarray([sample["observation_row"] for sample in samples])),
    }


def _extract_base(cache: Path, checkpoint: Path, seed: int, device) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    dataset = ActionSetNodeDataset(cache, np.arange(EXPECTED, dtype=np.int64))
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, collate_fn=_collate)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("schema_version") != "gse_action_set_node_checkpoint_v1" or saved.get("seed") != seed:
        raise RuntimeError(f"endpoint residual base checkpoint drift: seed{seed}")
    model = ActionSetNodeDetector().to(device); model.load_state_dict(saved["model"], strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            values = model(
                batch["tokens"].to(device=device, dtype=torch.float32),
                batch["history_mask"].to(device=device, dtype=torch.bool),
            )
            output["context"].append(values["causal_context"].cpu().numpy())
            output["structural"].append(values["structural_logit"].cpu().numpy())
            output["conditional"].append(values["conditional_decision_logits"].cpu().numpy())
            output["row"].append(batch["observation_row"].numpy())
    result = {key: np.concatenate(value) for key, value in output.items()}
    if not np.array_equal(result.pop("row"), np.arange(EXPECTED)):
        raise RuntimeError("endpoint residual base extraction row drift")
    return result


def _partition_contract(
    *, code: int, partition: np.ndarray, target: np.ndarray, episode: np.ndarray,
    identity: np.ndarray, endpoint_names: list[str], device,
) -> dict:
    import torch
    rows = np.flatnonzero(partition == code)
    target_part = target[rows].astype(np.int64)
    episode_global = episode[rows].astype(np.int64)
    positive = sorted(set(episode_global[episode_global >= 0].tolist()))
    remap = {value: index for index, value in enumerate(positive)}
    episode_part = np.asarray([remap.get(int(value), -1) for value in episode_global], dtype=np.int64)
    endpoint_code = {value: index for index, value in enumerate(sorted(endpoint_names))}
    identity_by_episode = np.full(len(positive), -1, dtype=np.int64)
    for global_episode, compact in remap.items():
        local = np.flatnonzero(episode_global == global_episode)
        names = set(identity[rows[local]].tolist())
        if len(names) != 1:
            raise RuntimeError("endpoint residual episode identity drift")
        name = next(iter(names))
        identity_by_episode[compact] = endpoint_code.get(name, -1)
    present = sorted(set(identity_by_episode[identity_by_episode >= 0].tolist()))
    if present != list(range(len(endpoint_names))):
        raise RuntimeError("endpoint residual relation identity episode coverage drift")
    return {
        "rows": rows,
        "target_np": target_part, "episode_np": episode_part,
        "target": torch.from_numpy(target_part).to(device),
        "episode": torch.from_numpy(episode_part).to(device),
        "endpoint_identity": torch.from_numpy(identity_by_episode).to(device),
    }


def _probability(structural: np.ndarray, conditional: np.ndarray, residual_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logit = structural.astype(np.float64) + residual_value.astype(np.float64)
    mass = 1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0)))
    shifted = conditional.astype(np.float64) - conditional.max(axis=1, keepdims=True)
    classes = np.exp(shifted); classes /= classes.sum(axis=1, keepdims=True)
    action_probability = np.concatenate(((1.0 - mass)[:, None], mass[:, None] * classes), axis=1)
    # The action head models only no-node/junction/terminal, while every graph
    # evaluator consumes the canonical five-event interface. Turn and
    # geometry-transition mass is exactly zero, matching the frozen baseline.
    probability = np.zeros((len(action_probability), 5), dtype=np.float64)
    probability[:, :3] = action_probability
    uncertainty = -(probability * np.log(np.clip(probability, 1e-8, 1.0))).sum(axis=1) / np.log(5.0)
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("endpoint residual five-event probability drift")
    return probability.astype(np.float32), uncertainty.astype(np.float32)


def _endpoint_metrics(
    probability: np.ndarray, uncertainty: np.ndarray, rows: np.ndarray,
    traversal: np.ndarray, sequence: np.ndarray, identity: np.ndarray, event: np.ndarray,
    endpoint_rows: list[dict],
) -> dict:
    triggers = extract_decision_mass_triggers(
        probability[rows], traversal[rows], sequence[rows], uncertainty[rows], decision_threshold=.97,
    )
    correct = set()
    for trigger in triggers:
        row = int(rows[int(trigger.row)])
        predicted = {1: "junction", 2: "terminal"}[int(trigger.predicted_event_index)]
        if event[row] == predicted and identity[row]: correct.add(str(identity[row]))
    low = {row["identity"] for row in endpoint_rows if row["support_bin"] == "1-3"}
    high = {row["identity"] for row in endpoint_rows if int(row["teacher_rows"]) >= 11}
    all_names = {row["identity"] for row in endpoint_rows}
    return {"low_correct": len(correct & low), "low_total": len(low), "high_correct": len(correct & high), "high_total": len(high), "all_correct": len(correct & all_names), "all_total": len(all_names)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path); parser.add_argument("--endpoint-audit", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0,1,2)); parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=10); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args(); started=time.monotonic()
    if args.output_dir.exists(): raise RuntimeError("endpoint residual seed output exists; overwrite forbidden")
    args.output_dir.mkdir(parents=True)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("endpoint residual training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    device=torch.device("cuda"); cache=args.cache_dir.resolve()
    manifest=json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!="gse_action_set_node_cache_v1" or manifest.get("causal_observations")!=EXPECTED: raise RuntimeError("endpoint residual cache drift")
    partition=np.load(cache/"partition_code.npy").astype(np.uint8); target=np.load(cache/"decision_target.npy").astype(np.int64); episode=np.load(cache/"decision_episode_id.npy").astype(np.int64); identity=np.load(cache/"identity.npy").astype(str); traversal=np.load(cache/"traversal_id.npy").astype(str); sequence=np.load(cache/"sequence_index.npy").astype(np.int64); global_index=np.load(cache/"global_sequence_index.npy").astype(np.int64)
    event=np.asarray(["corridor" if value==0 else "junction" if value==1 else "terminal" for value in target])
    endpoint_records=_read_jsonl(args.endpoint_audit.resolve()); fit_records=[row for row in endpoint_records if row["partition"]=="fit"]; selection_records=[row for row in endpoint_records if row["partition"]=="selection"]
    if len(fit_records)!=72 or len(selection_records)!=26: raise RuntimeError("endpoint residual endpoint population drift")
    base=_extract_base(cache,args.base_checkpoint.resolve(),args.seed,device)
    fit=_partition_contract(code=0,partition=partition,target=target,episode=episode,identity=identity,endpoint_names=[row["identity"] for row in fit_records],device=device)
    selection=_partition_contract(code=1,partition=partition,target=target,episode=episode,identity=identity,endpoint_names=[row["identity"] for row in selection_records],device=device)
    if int(torch.sum(fit["endpoint_identity"]>=0))!=286 or int(torch.sum(selection["endpoint_identity"]>=0))!=110: raise RuntimeError("endpoint residual episode exposure drift")
    fit_rows=fit["rows"]; selection_rows=selection["rows"]
    fit_context=torch.from_numpy(base["context"][fit_rows]).to(device); fit_structural=torch.from_numpy(base["structural"][fit_rows]).to(device); fit_conditional=torch.from_numpy(base["conditional"][fit_rows]).to(device)
    residual=EndpointIdentityStructuralResidual().to(device); optimizer=torch.optim.AdamW(residual.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    history=[]; candidates=[]
    def evaluate(step:int):
        residual.eval()
        with torch.inference_mode():
            all_residual=[]
            for start in range(0,EXPECTED,8192): all_residual.append(residual(torch.from_numpy(base["context"][start:start+8192]).to(device)).cpu().numpy())
        value=np.concatenate(all_residual); probability,uncertainty=_probability(base["structural"],base["conditional"],value)
        decision=evaluate_decision_mass_triggers(probability[selection_rows],selection["target_np"],selection["episode_np"],traversal[selection_rows],sequence[selection_rows],uncertainty[selection_rows],decision_threshold=.97)
        endpoint_metric=_endpoint_metrics(probability,uncertainty,selection_rows,traversal,sequence,identity,event,selection_records)
        record={"step":step,"decision":decision,"endpoint":endpoint_metric}
        history.append(record); return record,probability,uncertainty
    baseline,_,_=evaluate(0); best_state={key:value.detach().cpu().clone() for key,value in residual.state_dict().items()}; best_key=None; best_step=0
    for step in range(1,args.steps+1):
        residual.train(); optimizer.zero_grad(set_to_none=True); corrected=fit_structural+residual(fit_context)
        losses=identity_balanced_endpoint_mil_loss(structural_logit=corrected,conditional_decision_logits=fit_conditional,decision_target=fit["target"],episode_id=fit["episode"],endpoint_identity_by_episode=fit["endpoint_identity"])
        losses["total"].backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(),5.0); optimizer.step()
        if step%args.evaluation_interval==0 or step==args.steps:
            record,_,_=evaluate(step); decision=record["decision"]; endpoint_metric=record["endpoint"]
            false_count=decision["predicted_decision_triggers"]-decision["correctly_classified_unique_decision_episodes"]; baseline_false=baseline["decision"]["predicted_decision_triggers"]-baseline["decision"]["correctly_classified_unique_decision_episodes"]
            safe=(false_count<=baseline_false and decision["correctly_classified_unique_decision_episodes"]>=baseline["decision"]["correctly_classified_unique_decision_episodes"] and endpoint_metric["high_correct"]>=baseline["endpoint"]["high_correct"])
            key=(endpoint_metric["low_correct"],endpoint_metric["all_correct"],decision["correctly_classified_unique_decision_episodes"],-false_count,-float(losses["total"].detach().cpu()),-step)
            record["safe_candidate"]=safe;record["fit_loss"]={name:float(value.detach().cpu()) for name,value in losses.items()}
            if safe and (best_key is None or key>best_key): best_key=key;best_step=step;best_state={name:value.detach().cpu().clone() for name,value in residual.state_dict().items()}
            print(json.dumps(record,sort_keys=True),flush=True)
    residual.load_state_dict(best_state,strict=True); residual.eval()
    _,probability,uncertainty=evaluate(best_step)
    torch.save({"schema_version":"gse_endpoint_identity_structural_residual_checkpoint_v1","seed":args.seed,"step":best_step,"residual":residual.state_dict(),"base_checkpoint_sha256":_sha(args.base_checkpoint.resolve()),"selection_baseline":baseline,"selection_selected":history[-1] if best_step==history[-1]["step"] else next(row for row in history if row["step"]==best_step)},args.output_dir/"best.pt")
    np.savez_compressed(args.output_dir/"all_outputs.npz",global_sequence_index=global_index,probability=probability,uncertainty=uncertainty)
    summary={"schema_version":"gse_endpoint_identity_residual_training_seed_v1","seed":args.seed,"steps":args.steps,"best_step":best_step,"trainable_parameters":sum(p.numel() for p in residual.parameters()),"base_model_optimizer_steps":0,"optimizer_steps":args.steps,"fit_observations":len(fit_rows),"selection_observations":len(selection_rows),"fit_endpoint_identities":72,"selection_endpoint_identities":26,"selection_baseline":baseline,"selection_selected":next(row for row in history if row["step"]==best_step),"duration_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0}
    (args.output_dir/"history.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n",encoding="utf-8");(args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(summary,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
