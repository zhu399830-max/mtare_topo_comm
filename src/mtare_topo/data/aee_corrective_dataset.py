"""Frozen sampling and shard contracts for the 11,000-frame AEE corrective set.

This module contains no ray casting, teacher query, model inference, or file
write.  It types objective development-geometry eligibility produced by the
executor and performs deterministic structural selection over that qualified
population.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.aee_domain_adaptation import (
    ENVIRONMENT_SEEDS,
    RAW_FRAMES_PER_TRAJECTORY,
)
from mtare_topo.data.aee_sensor_operator_parity import (
    AEE_GAZEBO_RAW_MAX_RANGE_M,
    AEE_HORIZONTAL_SAMPLES,
    resample_aee_gazebo_organized_azimuth,
)
from mtare_topo.data.cano_phase2_selector_v2 import (
    length_proportional_tunnel_quotas,
)
from mtare_topo.data.cano_sensor_smoke import AZIMUTH_COLUMNS, MAX_RANGE_M, NEAR_RANGE_M


AEE_CORRECTIVE_WORLDS = ("tunnel", "garage")
AEE_TRAJECTORIES = 10
AEE_FRAMES_PER_TRAJECTORY = 100
AEE_FRAMES_PER_WORLD = 500
AEE_CORRECTIVE_FRAMES = 1_000
CANO_CORRECTIVE_WORLDS = 20
CANO_CLUSTERS_PER_WORLD = 100
CANO_FRAMES_PER_CLUSTER = 5
CANO_FRAMES_PER_WORLD = 500
CANO_CORRECTIVE_FRAMES = 10_000
TOTAL_CORRECTIVE_FRAMES = 11_000


@dataclass(frozen=True)
class AEECorrectiveTrajectory:
    index: int
    world: str
    environment_seed: int
    split: str = "corrective_train"

    @property
    def trajectory_id(self) -> str:
        return f"{self.index:02d}_{self.world}_seed{self.environment_seed}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trajectory_id": self.trajectory_id}


@dataclass(frozen=True)
class CorrectiveCanoFrameEligibility:
    frame_index: int
    eligible: bool
    teacher_eligible: bool
    native_clearance_passed: bool
    minimum_horizontal_clearance_m: float
    branch_los_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectiveCanoClusterEligibility:
    cluster_id: str
    frames: tuple[CorrectiveCanoFrameEligibility, ...]

    @property
    def eligible(self) -> bool:
        return bool(
            len(self.frames) == CANO_FRAMES_PER_CLUSTER
            and tuple(item.frame_index for item in self.frames) == tuple(range(CANO_FRAMES_PER_CLUSTER))
            and all(item.eligible for item in self.frames)
        )

    @property
    def rejected_frame_count(self) -> int:
        return sum(not item.eligible for item in self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aee_corrective_cano_cluster_eligibility_v1",
            "cluster_id": self.cluster_id,
            "eligible": self.eligible,
            "rejected_frame_count": self.rejected_frame_count,
            "frames": [item.to_dict() for item in self.frames],
        }


@dataclass(frozen=True)
class AEEMonotonicStampPair:
    """One deterministic, order-preserving raw/registered scan association."""

    pair_index: int
    raw_message_index: int
    registered_message_index: int
    raw_stamp_ns: int
    registered_stamp_ns: int
    absolute_delta_ns: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def monotonic_nearest_stamp_pairs(
    raw_stamps_ns: Sequence[int],
    registered_stamps_ns: Sequence[int],
    *,
    maximum_delta_ns: int,
    required_pairs: int,
) -> tuple[AEEMonotonicStampPair, ...]:
    """Match scans by nearest timestamp without reuse or order reversal.

    Registered scans are visited in timestamp order.  Each is associated with
    the nearest raw scan strictly after the previously used raw scan.  Ties are
    resolved by the smaller raw message index.  Associations at or beyond the
    threshold are rejected, and the first ``required_pairs`` valid associations
    define the frozen trajectory population.
    """

    raw = tuple(int(value) for value in raw_stamps_ns)
    registered = tuple(int(value) for value in registered_stamps_ns)
    if maximum_delta_ns <= 0 or required_pairs <= 0:
        raise ValueError("pairing threshold and required pair count must be positive")
    if len(raw) != len(set(raw)) or any(right <= left for left, right in zip(raw, raw[1:])):
        raise ValueError("raw timestamps must be unique and strictly increasing")
    if len(registered) != len(set(registered)) or any(
        right <= left for left, right in zip(registered, registered[1:])
    ):
        raise ValueError("registered timestamps must be unique and strictly increasing")

    pairs: list[AEEMonotonicStampPair] = []
    previous_raw_index = -1
    for registered_index, registered_stamp in enumerate(registered):
        lower = previous_raw_index + 1
        if lower >= len(raw):
            break
        insertion = bisect_left(raw, registered_stamp, lo=lower)
        candidate_indices = []
        if insertion < len(raw):
            candidate_indices.append(insertion)
        if insertion - 1 >= lower:
            candidate_indices.append(insertion - 1)
        raw_index = min(
            candidate_indices,
            key=lambda index: (abs(raw[index] - registered_stamp), index),
        )
        delta = abs(raw[raw_index] - registered_stamp)
        if delta >= maximum_delta_ns:
            continue
        pairs.append(
            AEEMonotonicStampPair(
                pair_index=len(pairs),
                raw_message_index=raw_index,
                registered_message_index=registered_index,
                raw_stamp_ns=raw[raw_index],
                registered_stamp_ns=registered_stamp,
                absolute_delta_ns=delta,
            )
        )
        previous_raw_index = raw_index
        if len(pairs) == required_pairs:
            break
    if len(pairs) != required_pairs:
        raise RuntimeError(
            f"only {len(pairs)} valid monotonic scan pairs; {required_pairs} required"
        )
    return tuple(pairs)


def build_corrective_cano_cluster_eligibility(
    cluster_id: str,
    frame_results: Sequence[Mapping[str, Any]],
) -> CorrectiveCanoClusterEligibility:
    """Validate and freeze the five objective eligibility results for a cluster."""

    if not isinstance(cluster_id, str) or not cluster_id:
        raise ValueError("cluster_id must be a nonempty string")
    if len(frame_results) != CANO_FRAMES_PER_CLUSTER:
        raise ValueError(f"cluster eligibility requires exactly {CANO_FRAMES_PER_CLUSTER} frames")
    frames: list[CorrectiveCanoFrameEligibility] = []
    for expected_index, result in enumerate(frame_results):
        frame_index = int(result["frame_index"])
        clearance = float(result["minimum_horizontal_clearance_m"])
        teacher_eligible = bool(result["teacher_eligible"])
        native_clearance_passed = bool(result["native_clearance_passed"])
        eligible = bool(result["eligible"])
        branch_los = tuple(bool(value) for value in result["branch_los"])
        if frame_index != expected_index:
            raise ValueError("cluster eligibility frame indices must be exactly 0..4")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("cluster eligibility clearance must be finite and nonnegative")
        if eligible != bool(teacher_eligible and native_clearance_passed):
            raise ValueError("frame eligibility must equal teacher and native-clearance eligibility")
        frames.append(
            CorrectiveCanoFrameEligibility(
                frame_index=frame_index,
                eligible=eligible,
                teacher_eligible=teacher_eligible,
                native_clearance_passed=native_clearance_passed,
                minimum_horizontal_clearance_m=clearance,
                branch_los_passed=bool(branch_los and all(branch_los)),
            )
        )
    return CorrectiveCanoClusterEligibility(cluster_id=cluster_id, frames=tuple(frames))


def enumerate_aee_corrective_trajectories() -> tuple[AEECorrectiveTrajectory, ...]:
    result = tuple(
        AEECorrectiveTrajectory(index=index, world=world, environment_seed=seed)
        for index, (world, seed) in enumerate(
            (world, seed) for world in AEE_CORRECTIVE_WORLDS for seed in ENVIRONMENT_SEEDS
        )
    )
    if (
        len(result) != AEE_TRAJECTORIES
        or len({item.trajectory_id for item in result}) != AEE_TRAJECTORIES
        or Counter(item.world for item in result) != {"tunnel": 5, "garage": 5}
        or any(item.split != "corrective_train" for item in result)
    ):
        raise RuntimeError("AEE corrective trajectory registry drift")
    return result


def aee_corrective_frame_indices(
    raw_count: int = RAW_FRAMES_PER_TRAJECTORY,
) -> np.ndarray:
    """Freeze 100 outcome-independent frames from each 3,000-frame trajectory."""

    if raw_count != RAW_FRAMES_PER_TRAJECTORY:
        raise ValueError(f"raw trajectory must contain exactly {RAW_FRAMES_PER_TRAJECTORY} frames")
    indices = np.arange(0, raw_count, raw_count // AEE_FRAMES_PER_TRAJECTORY, dtype=np.int64)
    if (
        len(indices) != AEE_FRAMES_PER_TRAJECTORY
        or int(indices[0]) != 0
        or int(indices[-1]) != 2970
        or not np.array_equal(np.diff(indices), np.full(99, 30, dtype=np.int64))
    ):
        raise RuntimeError("AEE corrective frame selection drift")
    return indices


def _event_choices(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    choices: dict[str, dict[str, Any]] = {}
    for role, key, distance_key in (
        ("junction", "junction_event_ids", "junction_event_distance_m"),
        ("terminal", "terminal_event_ids", "terminal_event_distance_m"),
    ):
        events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in candidates:
            for event_id in item.get(key, []):
                events[str(event_id)].append(item)
        for event_id, rows in sorted(events.items()):
            chosen = min(
                rows,
                key=lambda item: (
                    float(item.get(distance_key, {}).get(event_id, float("inf"))),
                    str(item["cluster_id"]),
                ),
            )
            choices[f"{role}:{event_id}"] = dict(chosen)
    return choices


def select_corrective_cano_clusters(
    candidates: Sequence[Mapping[str, Any]],
    selected_count: int = CANO_CLUSTERS_PER_WORLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select event-covering, per-tunnel, farthest-spaced 5 m clusters.

    Selection depends only on the frozen graph/spline candidate lattice.  It
    never observes a scan, teacher label, clearance result, or model output.
    """

    records = [dict(item) for item in candidates]
    identities = [str(item["cluster_id"]) for item in records]
    if not records or len(identities) != len(set(identities)):
        raise ValueError("candidate cluster identities must be nonempty and unique")
    quotas = length_proportional_tunnel_quotas(records, selected_count)
    by_tunnel: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_tunnel[int(item["tunnel_id"])].append(item)
    for rows in by_tunnel.values():
        rows.sort(key=lambda item: (float(item["center_arc_m"]), str(item["cluster_id"])))

    event_choices = _event_choices(records)
    mandatory_ids = {str(item["cluster_id"]) for item in event_choices.values()}
    for tunnel_id, rows in sorted(by_tunnel.items()):
        midpoint = 0.5 * (float(rows[0]["center_arc_m"]) + float(rows[-1]["center_arc_m"]))
        mandatory_ids.add(
            str(min(rows, key=lambda item: (abs(float(item["center_arc_m"]) - midpoint), str(item["cluster_id"])))["cluster_id"])
        )
    by_id = {str(item["cluster_id"]): item for item in records}
    selected_by_tunnel: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for identity in sorted(mandatory_ids):
        item = by_id[identity]
        selected_by_tunnel[int(item["tunnel_id"])].append(item)
    overflow = {
        tunnel_id: len(selected_by_tunnel[tunnel_id]) - int(quotas[tunnel_id])
        for tunnel_id in sorted(quotas)
        if len(selected_by_tunnel[tunnel_id]) > int(quotas[tunnel_id])
    }
    if overflow:
        raise RuntimeError(f"mandatory structural coverage exceeds frozen tunnel quota: {overflow}")

    for tunnel_id, rows in sorted(by_tunnel.items()):
        chosen = selected_by_tunnel[tunnel_id]
        chosen_ids = {str(item["cluster_id"]) for item in chosen}
        while len(chosen) < int(quotas[tunnel_id]):
            remaining = [item for item in rows if str(item["cluster_id"]) not in chosen_ids]
            if not remaining:
                raise RuntimeError(f"tunnel {tunnel_id} cannot satisfy cluster quota")
            def score(item: Mapping[str, Any]) -> tuple[float, str]:
                arc = float(item["center_arc_m"])
                nearest = min(abs(arc - float(other["center_arc_m"])) for other in chosen)
                return nearest, str(item["cluster_id"])
            best_distance = max(score(item)[0] for item in remaining)
            best = min(
                (item for item in remaining if abs(score(item)[0] - best_distance) <= 1e-12),
                key=lambda item: str(item["cluster_id"]),
            )
            chosen.append(best)
            chosen_ids.add(str(best["cluster_id"]))

    selected = sorted(
        (dict(item) for rows in selected_by_tunnel.values() for item in rows),
        key=lambda item: str(item["cluster_id"]),
    )
    selected_ids = {str(item["cluster_id"]) for item in selected}
    uncovered_events = sorted(
        event_id
        for event_id, item in event_choices.items()
        if str(item["cluster_id"]) not in selected_ids
    )
    maximum_gap = 0.0
    for tunnel_id, rows in by_tunnel.items():
        arcs = [float(item["center_arc_m"]) for item in selected_by_tunnel[tunnel_id]]
        maximum_gap = max(
            maximum_gap,
            max(min(abs(float(item["center_arc_m"]) - arc) for arc in arcs) for item in rows),
        )
    observed_tunnels = Counter(int(item["tunnel_id"]) for item in selected)
    passed = bool(
        len(selected) == len(selected_ids) == selected_count
        and dict(sorted(observed_tunnels.items())) == dict(sorted(quotas.items()))
        and not uncovered_events
    )
    audit = {
        "schema_version": "aee_corrective_cano_cluster_selector_v1",
        "passed": passed,
        "selection_blind_to_observations": True,
        "candidate_clusters": len(records),
        "selected_clusters": len(selected),
        "selected_frames": len(selected) * CANO_FRAMES_PER_CLUSTER,
        "tunnel_quotas": {str(key): int(value) for key, value in sorted(quotas.items())},
        "mandatory_cluster_count": len(mandatory_ids),
        "junction_and_terminal_events": len(event_choices),
        "uncovered_events": uncovered_events,
        "maximum_candidate_to_selected_same_tunnel_arc_distance_m": maximum_gap,
    }
    if not passed:
        raise RuntimeError(f"corrective Cano selector audit failed: {audit}")
    return selected, audit


def select_qualified_corrective_cano_clusters(
    candidates: Sequence[Mapping[str, Any]],
    qualifications: Sequence[CorrectiveCanoClusterEligibility],
    selected_count: int = CANO_CLUSTERS_PER_WORLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply objective eligibility, then topology-only structural scoring.

    The eligibility stage may use the complete geometry of the current
    development world.  Structural scoring over the resulting eligible set
    consumes candidate metadata only and never reads model outputs.  Coverage
    is checked against the complete pre-filter candidate lattice.
    """

    records = [dict(item) for item in candidates]
    identities = [str(item["cluster_id"]) for item in records]
    qualification_by_id = {item.cluster_id: item for item in qualifications}
    if not records or len(identities) != len(set(identities)):
        raise ValueError("candidate cluster identities must be nonempty and unique")
    if len(qualification_by_id) != len(qualifications) or set(qualification_by_id) != set(identities):
        raise ValueError("qualification identities must match the complete candidate lattice exactly")

    eligible = [item for item in records if qualification_by_id[str(item["cluster_id"])].eligible]
    if len(eligible) < selected_count:
        raise RuntimeError("eligible cluster population cannot satisfy selected count")
    original_tunnels = {int(item["tunnel_id"]) for item in records}
    eligible_tunnels = {int(item["tunnel_id"]) for item in eligible}
    missing_tunnels = sorted(original_tunnels - eligible_tunnels)
    if missing_tunnels:
        raise RuntimeError(f"objective eligibility removed complete tunnels: {missing_tunnels}")

    selected, structural_audit = select_corrective_cano_clusters(eligible, selected_count)
    selected_ids = {str(item["cluster_id"]) for item in selected}
    selected_rows = [item for item in records if str(item["cluster_id"]) in selected_ids]
    original_events = set(_event_choices(records))
    covered_events: set[str] = set()
    for item in selected_rows:
        covered_events.update(f"junction:{event_id}" for event_id in item.get("junction_event_ids", []))
        covered_events.update(f"terminal:{event_id}" for event_id in item.get("terminal_event_ids", []))
    uncovered_original_events = sorted(original_events - covered_events)
    if uncovered_original_events:
        raise RuntimeError(
            "objective eligibility cannot preserve complete-lattice structural events: "
            f"{uncovered_original_events}"
        )

    original_lattice_maximum_gap = 0.0
    selected_by_tunnel: dict[int, list[float]] = defaultdict(list)
    for item in selected:
        selected_by_tunnel[int(item["tunnel_id"])].append(float(item["center_arc_m"]))
    for item in records:
        arcs = selected_by_tunnel[int(item["tunnel_id"])]
        original_lattice_maximum_gap = max(
            original_lattice_maximum_gap,
            min(abs(float(item["center_arc_m"]) - arc) for arc in arcs),
        )

    rejected = [item for item in qualifications if not item.eligible]
    structural_audit = dict(structural_audit)
    structural_audit.pop("selection_blind_to_observations", None)
    passed = bool(
        structural_audit.get("passed")
        and len(selected) == selected_count
        and not uncovered_original_events
        and not missing_tunnels
        and all(qualification_by_id[identity].eligible for identity in selected_ids)
    )
    audit = {
        "schema_version": "aee_corrective_cano_eligibility_first_selector_v1",
        "passed": passed,
        "selection_stage": "post_eligibility_structural_selection",
        "eligibility_uses_complete_development_geometry": True,
        "structural_scoring_blind_to_observations": True,
        "selection_blind_to_model_outputs": True,
        "selection_blind_to_strict_test": True,
        "candidate_lattice_clusters": len(records),
        "candidate_lattice_frames": len(records) * CANO_FRAMES_PER_CLUSTER,
        "eligible_clusters": len(eligible),
        "eligible_frames": len(eligible) * CANO_FRAMES_PER_CLUSTER,
        "ineligible_clusters": len(rejected),
        "ineligible_frames": sum(item.rejected_frame_count for item in rejected),
        "selected_clusters": len(selected),
        "selected_frames": len(selected) * CANO_FRAMES_PER_CLUSTER,
        "complete_lattice_events": len(original_events),
        "uncovered_complete_lattice_events": uncovered_original_events,
        "missing_complete_lattice_tunnels": missing_tunnels,
        "maximum_complete_lattice_to_selected_same_tunnel_arc_distance_m": original_lattice_maximum_gap,
        "structural_selector_audit": structural_audit,
    }
    if not passed:
        raise RuntimeError(f"qualified corrective Cano selector audit failed: {audit}")
    return selected, audit


def audit_corrective_sensor_shard(payload: Mapping[str, np.ndarray], expected_frames: int) -> dict[str, Any]:
    expected = {
        "raw_range_m": (expected_frames, 16, AEE_HORIZONTAL_SAMPLES),
        "raw_valid_mask": (expected_frames, 16, AEE_HORIZONTAL_SAMPLES),
        "range_m": (expected_frames, 16, AZIMUTH_COLUMNS),
        "valid_mask": (expected_frames, 16, AZIMUTH_COLUMNS),
        "sensor_xyz_m": (expected_frames, 3),
        "sensor_orientation_xyzw": (expected_frames, 4),
        "yaw_deg": (expected_frames,),
        "frame_id": (expected_frames,),
    }
    missing = sorted(set(expected) - set(payload))
    shape_mismatches = {
        key: {"actual": list(np.asarray(payload[key]).shape), "expected": list(shape)}
        for key, shape in expected.items()
        if key in payload and np.asarray(payload[key]).shape != shape
    }
    finite_failures = [
        key for key in expected if key in payload and key not in ("frame_id",) and not np.all(np.isfinite(payload[key]))
    ]
    raw_range = np.asarray(payload.get("raw_range_m", []), dtype=np.float32)
    raw_valid = np.asarray(payload.get("raw_valid_mask", []), dtype=bool)
    model_range = np.asarray(payload.get("range_m", []), dtype=np.float32)
    model_valid = np.asarray(payload.get("valid_mask", []), dtype=bool)
    raw_contract = bool(
        raw_range.shape == expected["raw_range_m"]
        and raw_valid.shape == expected["raw_valid_mask"]
        and np.all((raw_range >= 0.0) & (raw_range <= AEE_GAZEBO_RAW_MAX_RANGE_M))
        and np.all(raw_range[~raw_valid] == np.float32(AEE_GAZEBO_RAW_MAX_RANGE_M))
    )
    model_contract = bool(
        model_range.shape == expected["range_m"]
        and model_valid.shape == expected["valid_mask"]
        and np.all((model_range >= NEAR_RANGE_M) & (model_range <= MAX_RANGE_M))
        and np.all(model_range[~model_valid] == np.float32(MAX_RANGE_M))
        and np.all(model_valid.sum(axis=(1, 2)) > 0)
    )
    resample_exact = False
    if raw_contract and model_contract:
        source_model_valid = raw_valid & (raw_range >= NEAR_RANGE_M) & (raw_range <= MAX_RANGE_M)
        source_model_range = np.where(source_model_valid, raw_range, np.float32(MAX_RANGE_M))
        expected_range, expected_valid = resample_aee_gazebo_organized_azimuth(
            source_model_range, source_model_valid.astype(np.uint8)
        )
        resample_exact = bool(
            np.array_equal(model_range, expected_range) and np.array_equal(model_valid, expected_valid.astype(bool))
        )
    frame_ids = np.asarray(payload.get("frame_id", []))
    identities_unique = bool(frame_ids.shape == (expected_frames,) and len(set(frame_ids.tolist())) == expected_frames)
    passed = bool(
        not missing
        and not shape_mismatches
        and not finite_failures
        and raw_contract
        and model_contract
        and resample_exact
        and identities_unique
    )
    return {
        "schema_version": "aee_corrective_sensor_shard_audit_v1",
        "passed": passed,
        "expected_frames": expected_frames,
        "missing_arrays": missing,
        "shape_mismatches": shape_mismatches,
        "nonfinite_arrays": finite_failures,
        "raw_16x350_contract_passed": raw_contract,
        "model_16x720_contract_passed": model_contract,
        "raw_to_model_resample_bitwise_exact": resample_exact,
        "frame_identities_unique": identities_unique,
    }


__all__ = [
    "AEE_CORRECTIVE_FRAMES",
    "AEE_CORRECTIVE_WORLDS",
    "AEE_FRAMES_PER_TRAJECTORY",
    "AEE_FRAMES_PER_WORLD",
    "AEEMonotonicStampPair",
    "AEECorrectiveTrajectory",
    "CorrectiveCanoClusterEligibility",
    "CorrectiveCanoFrameEligibility",
    "CANO_CLUSTERS_PER_WORLD",
    "CANO_CORRECTIVE_FRAMES",
    "CANO_CORRECTIVE_WORLDS",
    "CANO_FRAMES_PER_CLUSTER",
    "CANO_FRAMES_PER_WORLD",
    "TOTAL_CORRECTIVE_FRAMES",
    "aee_corrective_frame_indices",
    "audit_corrective_sensor_shard",
    "build_corrective_cano_cluster_eligibility",
    "enumerate_aee_corrective_trajectories",
    "monotonic_nearest_stamp_pairs",
    "select_corrective_cano_clusters",
    "select_qualified_corrective_cano_clusters",
]
