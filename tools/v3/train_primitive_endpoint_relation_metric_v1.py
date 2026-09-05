#!/usr/bin/env python3
"""Train one no-slot endpoint relation metric with the frozen observable backbone."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

import train_primitive_local_composition_slot_v1 as base
from mtare_topo.representation.primitive_endpoint_relation_metric_model import (
    FrozenObservableEndpointRelationMetricNet,
)
from mtare_topo.representation.primitive_endpoint_relation_metric_training import (
    endpoint_relation_metric_losses,
)
from mtare_topo.representation.primitive_local_composition_slot_training import (
    align_local_composition_slot_targets,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.representation.primitive_relation_observable_training import observable_numpy_batch_to_torch


HEAD_PARAMETERS = 426_818
HEAD_TENSORS = 34
TOTAL_PARAMETERS = 3_062_449
LOSS_NAMES = (
    "balanced_pair_logistic", "same_composition_compactness",
    "different_composition_separation", "disconnected_overlap_rejection", "total",
)


def load_frozen_model(checkpoint_path: Path, *, seed: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity") != "dual_endpoint_observed"
    ):
        raise RuntimeError("endpoint relation metric source checkpoint identity drift")
    backbone = ObservableSparsePortRelationNet(); backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.manual_seed(2_026_090_400 + seed)
    return FrozenObservableEndpointRelationMetricNet(backbone), checkpoint


def batch_losses(model, numpy_batch, *, device: torch.device):
    batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
    prediction = model(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg)
    assignments = match_primitives(prediction.primitive, batch.targets)
    targets = align_local_composition_slot_targets(batch.targets, assignments)
    losses = endpoint_relation_metric_losses(prediction.relation, targets)
    return losses, prediction, targets


def save_checkpoint(path: Path, *, model, optimizer, seed: int, epoch: int, source_checkpoint: Path, training_metrics: dict, selection_metrics: dict) -> None:
    payload = {
        "schema_version": "primitive_endpoint_relation_metric_checkpoint_v1",
        "seed": seed, "epoch": epoch,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": base._sha(source_checkpoint),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "training_metrics": training_metrics, "selection_metrics": selection_metrics,
        "training_contract": {
            "epochs": 3, "training_batch_size": 128, "evaluation_batch_size": 128,
            "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
            "fit_rows": 426552, "c07_rows": 64644,
            "selection_metric": "mean_C07_endpoint_relation_metric_total_loss",
            "teacher_attachment_validity": "dual_endpoint_observed",
            "teacher_relation_representation": "lossless_local_composition_cliques_without_slot_identity",
            "relation_embedding_dim": 64, "arbitrary_slots": False, "dustbin_class": False,
            "frozen_observable_backbone": True, "joint_metric_loss_from_epoch_zero": True,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, temporary); temporary.replace(path)


def main() -> int:
    base.EXPECTED_HEAD_PARAMETERS = HEAD_PARAMETERS
    base.EXPECTED_HEAD_TENSORS = HEAD_TENSORS
    base.EXPECTED_TOTAL_PARAMETERS = TOTAL_PARAMETERS
    base.LOSS_NAMES = LOSS_NAMES
    base.load_frozen_model = load_frozen_model
    base.batch_losses = batch_losses
    base._save_checkpoint = save_checkpoint
    result = base.main()
    if result == 0:
        output_index = base.sys.argv.index("--output-dir") + 1 if hasattr(base, "sys") else None
        if output_index is None:
            # argparse has already normalized the path; recover it from this wrapper's argv.
            import sys
            output_index = sys.argv.index("--output-dir") + 1
            output = Path(sys.argv[output_index]).resolve()
        else:
            output = Path(base.sys.argv[output_index]).resolve()
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["schema_version"] = "primitive_endpoint_relation_metric_seed_training_v1"
        summary["method"] = "no_slot_endpoint_relation_metric_complete_link"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
