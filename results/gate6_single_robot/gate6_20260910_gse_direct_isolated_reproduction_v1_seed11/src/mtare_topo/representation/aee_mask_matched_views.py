"""Deterministic AEE-validity masking for Cano range-image training views."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Sequence

import numpy as np


def _permutation(size: int, seed: int, frame_ids: Sequence[str]) -> list[int]:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    for frame_id in sorted(frame_ids):
        digest.update(b"\0")
        digest.update(frame_id.encode("utf-8"))
    rng = random.Random(int.from_bytes(digest.digest()[:8], "big"))
    order = list(range(size))
    rng.shuffle(order)
    return order


def build_aee_mask_matched_cano_views(
    samples: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create one sparse Cano view per dense Cano sample without inventing returns.

    The function pairs equal-sized Cano and AEE subsets deterministically.  It
    applies only the paired AEE valid mask to Cano: a Cano miss remains a miss,
    and every newly invalid range is set to the canonical normalized max range.
    Input samples are never mutated.
    """

    cano = sorted(
        (item for item in samples if item.get("domain", "cano") == "cano"),
        key=lambda item: str(item["frame_id"]),
    )
    aee = sorted(
        (item for item in samples if item.get("domain") == "aee"),
        key=lambda item: str(item["frame_id"]),
    )
    if not cano or len(cano) != len(aee):
        raise ValueError("mask matching requires equal positive Cano and AEE sample counts")
    order = _permutation(len(aee), seed, [str(item["frame_id"]) for item in samples])
    views: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for cano_item, aee_index in zip(cano, order):
        aee_item = aee[aee_index]
        cano_student = np.asarray(cano_item["student"], dtype=np.float32)
        aee_student = np.asarray(aee_item["student"], dtype=np.float32)
        if cano_student.shape != (2, 16, 720) or aee_student.shape != (2, 16, 720):
            raise ValueError("student view must have shape [2,16,720]")
        if not np.isfinite(cano_student).all() or not np.isfinite(aee_student).all():
            raise ValueError("student view must be finite")
        cano_valid = cano_student[1] > 0.5
        aee_valid = aee_student[1] > 0.5
        matched_valid = cano_valid & aee_valid
        student = cano_student.copy()
        student[0, ~matched_valid] = 1.0
        student[1] = matched_valid.astype(np.float32)
        frame_id = str(cano_item["frame_id"])
        mask_frame_id = str(aee_item["frame_id"])
        view = dict(cano_item)
        view.update(
            student=student,
            domain="cano_mask_matched",
            view_id=f"{frame_id}:aee_mask:{mask_frame_id}",
            mask_source_frame_id=mask_frame_id,
            loss_weight=np.float32(0.5),
        )
        views.append(view)
        provenance.append(
            {
                "cano_frame_id": frame_id,
                "aee_mask_frame_id": mask_frame_id,
                "view_id": str(view["view_id"]),
            }
        )
    return views, provenance


def expand_domain_adaptation_views(
    samples: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return dense Cano, matched Cano, and unchanged AEE views.

    Dense and matched views each receive weight 0.5, so each independent Cano
    sample has total weight 1.0, equal to one independent AEE sample.
    """

    dense: list[dict[str, Any]] = []
    aee: list[dict[str, Any]] = []
    for item in samples:
        domain = item.get("domain", "cano")
        copied = dict(item)
        if domain == "cano":
            copied["domain"] = "cano_dense"
            copied["loss_weight"] = np.float32(0.5)
            copied["view_id"] = f"{item['frame_id']}:dense"
            dense.append(copied)
        elif domain == "aee":
            copied["loss_weight"] = np.float32(1.0)
            copied["view_id"] = f"{item['frame_id']}:raw"
            aee.append(copied)
        else:
            raise ValueError(f"unsupported source domain: {domain}")
    matched, provenance = build_aee_mask_matched_cano_views(samples, seed=seed)
    return dense + matched + aee, provenance


__all__ = ["build_aee_mask_matched_cano_views", "expand_domain_adaptation_views"]
