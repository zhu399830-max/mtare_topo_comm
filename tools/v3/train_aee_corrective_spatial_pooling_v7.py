#!/usr/bin/env python3
"""V7: retain azimuth structure until after the nonlinear embedding transform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import train_aee_corrective_full_encoder_v3 as base
import train_aee_corrective_embedding_adaptation_v6 as v6
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


class SpatialPoolingStructuralSemanticNet(StructuralSemanticNet):
    """Use nonlinear-before-pooling readout without adding checkpoint tensors."""

    def forward(self, student: torch.Tensor) -> dict[str, torch.Tensor]:
        if student.ndim != 4 or tuple(student.shape[1:]) != (2, 16, 720):
            raise ValueError(f"expected [B,2,16,720], got {tuple(student.shape)}")
        feature_map = self.encoder(student)
        azimuth_feature = feature_map.mean(dim=2)
        direction_low = self.direction_head(azimuth_feature)
        if direction_low.shape[-1] != 180:
            raise RuntimeError(f"expected 180-column directional latent, got {direction_low.shape[-1]}")
        direction_logits = direction_low.repeat_interleave(4, dim=-1).squeeze(1)
        # The source model averaged azimuth before its nonlinear embedding,
        # erasing the distribution of local structural features. Applying the
        # same pointwise MLP first retains that distribution; the final mean is
        # still exactly invariant to circular azimuth shifts.
        pointwise = self.embedding_head(azimuth_feature.transpose(1, 2))
        z_role = F.normalize(pointwise.mean(dim=1), dim=1)
        return {
            "direction_logits": direction_logits,
            "count_logits": self.count_head(z_role),
            "role_logits": self.role_head(z_role),
            "z_role": z_role,
        }


def main() -> int:
    base.StructuralSemanticNet = SpatialPoolingStructuralSemanticNet
    code = v6.main()
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["mode"] = "M1D_AEE_CORRECTIVE_SPATIAL_POOLING_V7"
    checkpoint["config"].update(classification_readout="pointwise_embedding_then_circular_mean_v1")
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        schema_version="aee_corrective_spatial_pooling_seed_summary_v7",
        status="COMPLETED_AEE_CORRECTIVE_SPATIAL_POOLING_SEED_V7",
        best_checkpoint_sha256=sha256(checkpoint_path),
        representation_contract="encoder_frozen_spatial_pooling_embedding_adapted_direction_stage1_exact",
        classification_readout="pointwise_embedding_then_circular_mean_v1",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
