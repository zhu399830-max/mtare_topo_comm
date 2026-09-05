"""Experiment governance for MASTER PLAN V3.

This module deliberately uses only the Python standard library so governance
checks remain available before the research environment is installed.
"""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_SPEC_VERSION = "v3_run_spec_v1"
DATA_CARD_VERSION = "v3_data_card_v1"
SENSOR_SMOKE_CARD_VERSION = "v3_sensor_smoke_card_v1"
SENSOR_CONTRACT_PILOT_CARD_VERSION = "v3_sensor_contract_pilot_card_v1"

GATE_RESULT_DIRECTORIES = {
    0: "gate0_baseline",
    1: "gate1_data",
    2: "gate2_representation",
    3: "gate3_semantics",
    4: "gate4_topology",
    5: "gate5_shadow",
    6: "gate6_single_robot",
    7: "gate7_multi_robot",
    8: "gate8_final",
}

DATA_APPROVAL_OPERATIONS = {
    "audit",
    "sensor_smoke",
    "sensor_contract_pilot",
    "annotation_pilot",
    "ai_annotation",
    "data_export",
    "teacher_generation",
    "teacher_calibration",
    "self_supervised_pretraining",
    "training",
    "normalization",
    "threshold_calibration",
    "augmentation_tuning",
    "checkpoint_selection",
    "topology_replay",
    "shadow",
}

OPERATION_GATE_RANGE = {
    "infrastructure": (0, 8),
    "audit": (0, 8),
    "interface_audit": (0, 0),
    "baseline": (0, 0),
    "sensor_smoke": (0, 1),
    "sensor_contract_pilot": (0, 1),
    "annotation_pilot": (1, 1),
    "ai_annotation": (1, 1),
    # Gate 3 may materialize an explicitly approved development-only Teacher
    # required by the current learned-semantics method.  The Data Card still
    # binds the exact operation, Gate and population; Gate 4+ remains closed.
    "data_export": (1, 3),
    "teacher_generation": (1, 3),
    "teacher_calibration": (1, 1),
    "self_supervised_pretraining": (2, 2),
    "training": (2, 3),
    "normalization": (1, 3),
    "threshold_calibration": (3, 4),
    "augmentation_tuning": (2, 3),
    "checkpoint_selection": (2, 3),
    "topology_replay": (4, 4),
    "shadow": (5, 5),
    "closed_loop_single": (6, 6),
    "closed_loop_multi": (7, 7),
    "final_benchmark": (8, 8),
}


def validate_sensor_smoke_card(card: Mapping[str, Any]) -> ValidationReport:
    """Validate a zero-dataset, fixed-world sensor diagnostic approval.

    This card exists so a small pre-dataset sensor feasibility run does not
    fabricate train/validation/test splits or trajectories merely to satisfy
    the formal dataset-card schema.
    """

    errors: list[str] = []
    if card.get("schema_version") != SENSOR_SMOKE_CARD_VERSION:
        errors.append(f"schema_version must be {SENSOR_SMOKE_CARD_VERSION!r}")
    if not _is_nonempty_string(card.get("card_id")):
        errors.append("card_id must be a non-empty string")
    if not _is_nonempty_string(card.get("purpose")):
        errors.append("purpose must be a non-empty string")

    approval = _required_mapping(card, "approval", errors)
    if approval.get("status") != "APPROVED":
        errors.append("approval.status must be 'APPROVED'")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not _is_nonempty_string(approval.get(key)):
            errors.append(f"approval.{key} must be a non-empty string")
    if approval.get("authorized_operations") != ["sensor_smoke"]:
        errors.append("approval.authorized_operations must be exactly ['sensor_smoke']")
    authorized_gates = approval.get("authorized_gates")
    if (
        not isinstance(authorized_gates, list)
        or not authorized_gates
        or not all(
            isinstance(gate, int)
            and not isinstance(gate, bool)
            and gate in (0, 1)
            for gate in authorized_gates
        )
    ):
        errors.append("approval.authorized_gates must be a non-empty subset of [0, 1]")

    source = _required_mapping(card, "source", errors)
    if not _is_nonempty_string_list(source.get("raw_sources")):
        errors.append("source.raw_sources must be a non-empty list")
    if not _is_nonempty_string(source.get("license_or_allowed_use")):
        errors.append("source.license_or_allowed_use must be documented")
    if not _is_nonempty_string(source.get("world_id")):
        errors.append("source.world_id must be documented")

    scope = _required_mapping(card, "scope", errors)
    if scope.get("formal_dataset") is not False:
        errors.append("scope.formal_dataset must be false")
    if scope.get("split") != "NONE_DIAGNOSTIC_SMOKE_ONLY":
        errors.append("scope.split must be 'NONE_DIAGNOSTIC_SMOKE_ONLY'")
    if scope.get("world_count") != 1:
        errors.append("scope.world_count must be exactly 1")
    pose_count = scope.get("pose_count")
    if not isinstance(pose_count, int) or isinstance(pose_count, bool) or pose_count <= 0:
        errors.append("scope.pose_count must be a positive integer")
    for key in ("training_samples", "validation_samples", "test_samples", "trajectories"):
        if scope.get(key) != 0:
            errors.append(f"scope.{key} must be 0")
    for key in (
        "no_training",
        "no_model_selection",
        "no_threshold_calibration",
        "benchmark_worlds_excluded",
    ):
        if scope.get(key) is not True:
            errors.append(f"scope.{key} must be true")

    sampling = _required_mapping(card, "sampling", errors)
    if sampling.get("planned_pose_count") != pose_count:
        errors.append("sampling.planned_pose_count must equal scope.pose_count")
    role_counts = sampling.get("role_counts")
    if not isinstance(role_counts, dict) or not role_counts:
        errors.append("sampling.role_counts must be a non-empty object")
    elif not all(
        _is_nonempty_string(key)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in role_counts.items()
    ):
        errors.append("sampling.role_counts values must be non-negative integers")
    elif isinstance(pose_count, int) and sum(role_counts.values()) != pose_count:
        errors.append("sampling.role_counts must sum to scope.pose_count")
    for key in ("selection_rule", "sensor_contract", "label_contract"):
        if not _is_nonempty_string(sampling.get(key)):
            errors.append(f"sampling.{key} must be documented")

    evidence = _required_mapping(card, "evidence", errors)
    for key in ("machine_metrics", "complete_visual_review", "failure_policy"):
        if not _is_nonempty_string(evidence.get(key)):
            errors.append(f"evidence.{key} must be documented")

    cost = _required_mapping(card, "estimated_cost", errors)
    for key in ("disk_gb", "wall_time_hours"):
        value = cost.get(key)
        if not _is_number(value) or value < 0:
            errors.append(f"estimated_cost.{key} must be a non-negative number")
    if not _is_nonempty_string(cost.get("compute")):
        errors.append("estimated_cost.compute must be documented")

    return ValidationReport(not errors, tuple(errors))


def validate_sensor_contract_pilot_card(card: Mapping[str, Any]) -> ValidationReport:
    """Validate an approved multi-world, zero-training sensor contract pilot."""

    errors: list[str] = []
    if card.get("schema_version") != SENSOR_CONTRACT_PILOT_CARD_VERSION:
        errors.append(
            f"schema_version must be {SENSOR_CONTRACT_PILOT_CARD_VERSION!r}"
        )
    if not _is_nonempty_string(card.get("card_id")):
        errors.append("card_id must be a non-empty string")
    if not _is_nonempty_string(card.get("purpose")):
        errors.append("purpose must be a non-empty string")

    approval = _required_mapping(card, "approval", errors)
    if approval.get("status") != "APPROVED":
        errors.append("approval.status must be 'APPROVED'")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not _is_nonempty_string(approval.get(key)):
            errors.append(f"approval.{key} must be a non-empty string")
    if approval.get("authorized_operations") != ["sensor_contract_pilot"]:
        errors.append(
            "approval.authorized_operations must be exactly ['sensor_contract_pilot']"
        )
    authorized_gates = approval.get("authorized_gates")
    if (
        not isinstance(authorized_gates, list)
        or not authorized_gates
        or not all(
            isinstance(gate, int)
            and not isinstance(gate, bool)
            and gate in (0, 1)
            for gate in authorized_gates
        )
    ):
        errors.append("approval.authorized_gates must be a non-empty subset of [0, 1]")

    source = _required_mapping(card, "source", errors)
    if not _is_nonempty_string_list(source.get("raw_sources")):
        errors.append("source.raw_sources must be a non-empty list")
    for key in ("license_or_allowed_use", "benchmark_exclusion"):
        if not _is_nonempty_string(source.get(key)):
            errors.append(f"source.{key} must be documented")

    scope = _required_mapping(card, "scope", errors)
    if scope.get("formal_dataset") is not False:
        errors.append("scope.formal_dataset must be false")
    if scope.get("split") != "NONE_DIAGNOSTIC_CONTRACT_PILOT_ONLY":
        errors.append(
            "scope.split must be 'NONE_DIAGNOSTIC_CONTRACT_PILOT_ONLY'"
        )
    positive_counts = (
        "topology_parent_count",
        "mesh_world_count",
        "canonical_anchor_count",
        "diagnostic_observation_count",
    )
    for key in positive_counts:
        value = scope.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"scope.{key} must be a positive integer")
    for key in ("training_samples", "validation_samples", "test_samples", "trajectories", "models"):
        if scope.get(key) != 0:
            errors.append(f"scope.{key} must be 0")
    for key in (
        "no_training",
        "no_model_selection",
        "no_threshold_calibration",
        "no_ai_labels",
        "benchmark_worlds_excluded",
    ):
        if scope.get(key) is not True:
            errors.append(f"scope.{key} must be true")

    generation = _required_mapping(card, "generation", errors)
    parent_ids = generation.get("parent_ids")
    topology_seeds = generation.get("topology_seeds")
    geometry_seeds = generation.get("geometry_seeds")
    parent_count = scope.get("topology_parent_count")
    if not _is_nonempty_string_list(parent_ids) or len(set(parent_ids)) != len(parent_ids):
        errors.append("generation.parent_ids must be a non-empty unique string list")
    elif isinstance(parent_count, int) and len(parent_ids) != parent_count:
        errors.append("generation.parent_ids count must equal scope.topology_parent_count")
    for key, values in (("topology_seeds", topology_seeds), ("geometry_seeds", geometry_seeds)):
        if (
            not isinstance(values, list)
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            or len(set(values)) != len(values)
        ):
            errors.append(f"generation.{key} must be a unique integer list")
        elif isinstance(parent_count, int) and len(values) != parent_count:
            errors.append(f"generation.{key} count must equal scope.topology_parent_count")
    for key in ("retry_policy", "variant_policy"):
        if not _is_nonempty_string(generation.get(key)):
            errors.append(f"generation.{key} must be documented")

    sampling = _required_mapping(card, "sampling", errors)
    anchors_per_parent = sampling.get("anchors_per_parent")
    views_per_anchor = sampling.get("views_per_anchor")
    if not isinstance(anchors_per_parent, int) or isinstance(anchors_per_parent, bool) or anchors_per_parent <= 0:
        errors.append("sampling.anchors_per_parent must be a positive integer")
    if not isinstance(views_per_anchor, int) or isinstance(views_per_anchor, bool) or views_per_anchor <= 0:
        errors.append("sampling.views_per_anchor must be a positive integer")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in (parent_count, anchors_per_parent, views_per_anchor)):
        if scope.get("canonical_anchor_count") != parent_count * anchors_per_parent:
            errors.append("scope.canonical_anchor_count must equal parents * anchors_per_parent")
        if scope.get("diagnostic_observation_count") != parent_count * anchors_per_parent * views_per_anchor:
            errors.append("scope.diagnostic_observation_count must equal parents * anchors * views")
    offsets = sampling.get("view_yaw_offsets_deg")
    if not isinstance(offsets, list) or len(offsets) != views_per_anchor or not all(_is_number(value) for value in offsets):
        errors.append("sampling.view_yaw_offsets_deg must contain one numeric offset per view")
    for key in ("rays_per_observation",):
        value = sampling.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"sampling.{key} must be a positive integer")
    for key in ("sensor_contract", "label_contract", "storage_contract"):
        if not _is_nonempty_string(sampling.get(key)):
            errors.append(f"sampling.{key} must be documented")

    evidence = _required_mapping(card, "evidence", errors)
    for key in ("machine_metrics", "complete_visual_review", "failure_policy"):
        if not _is_nonempty_string(evidence.get(key)):
            errors.append(f"evidence.{key} must be documented")
    cost = _required_mapping(card, "estimated_cost", errors)
    for key in ("disk_gb", "wall_time_hours"):
        value = cost.get(key)
        if not _is_number(value) or value < 0:
            errors.append(f"estimated_cost.{key} must be a non-negative number")
    if not _is_nonempty_string(cost.get("compute")):
        errors.append("estimated_cost.compute must be documented")

    return ValidationReport(not errors, tuple(errors))

DEVELOPMENT_WORLD_ROLES = (
    "train",
    "validation",
    "ssl",
    "normalization",
    "teacher_calibration",
    "threshold_calibration",
    "augmentation_tuning",
    "checkpoint_selection",
)

TEST_EXCLUSION_FLAGS = (
    "test_excluded_from_supervised_training",
    "test_excluded_from_ssl",
    "test_excluded_from_normalization",
    "test_excluded_from_teacher_calibration",
    "test_excluded_from_threshold_calibration",
    "test_excluded_from_augmentation_tuning",
    "test_excluded_from_checkpoint_selection",
)

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,80}$")
DATE_PATTERN = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable result of a governance validation."""

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PreflightReport:
    """Validation result plus the deterministic output directory."""

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    run_id: str | None
    output_dir: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "run_id": self.run_id,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_nonempty_string(item) for item in value)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _required_mapping(
    parent: Mapping[str, Any], key: str, errors: list[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _world_set(worlds: Mapping[str, Any], role: str, errors: list[str]) -> set[str]:
    value = worlds.get(role, [])
    if not isinstance(value, list) or not all(_is_nonempty_string(v) for v in value):
        errors.append(f"worlds.{role} must be a list of non-empty world names")
        return set()
    duplicates = {item for item in value if value.count(item) > 1}
    if duplicates:
        errors.append(f"worlds.{role} contains duplicates: {sorted(duplicates)}")
    return set(value)


def validate_data_card(card: Mapping[str, Any]) -> ValidationReport:
    """Validate data provenance, independence, approval, and test isolation."""

    errors: list[str] = []
    warnings: list[str] = []
    diagnostic = card.get("diagnostic_geometry_audit")
    diagnostic_enabled = isinstance(diagnostic, dict) and diagnostic.get("enabled") is True

    if card.get("schema_version") != DATA_CARD_VERSION:
        errors.append(f"schema_version must be {DATA_CARD_VERSION!r}")
    if not _is_nonempty_string(card.get("card_id")):
        errors.append("card_id must be a non-empty string")
    if not _is_nonempty_string(card.get("purpose")):
        errors.append("purpose must be a non-empty string")

    approval = _required_mapping(card, "approval", errors)
    if approval.get("status") != "APPROVED":
        errors.append("approval.status must be 'APPROVED'")
    for key in ("approved_by", "approved_at", "scope"):
        if not _is_nonempty_string(approval.get(key)):
            errors.append(f"approval.{key} must be a non-empty string")
    authorized_operations = approval.get("authorized_operations")
    if not _is_nonempty_string_list(authorized_operations):
        errors.append("approval.authorized_operations must be a non-empty list")
    elif not set(authorized_operations).issubset(DATA_APPROVAL_OPERATIONS):
        invalid = sorted(set(authorized_operations) - DATA_APPROVAL_OPERATIONS)
        errors.append(f"approval.authorized_operations contains invalid values: {invalid}")
    authorized_gates = approval.get("authorized_gates")
    if (
        not isinstance(authorized_gates, list)
        or not authorized_gates
        or not all(
            isinstance(gate, int)
            and not isinstance(gate, bool)
            and gate in GATE_RESULT_DIRECTORIES
            for gate in authorized_gates
        )
    ):
        errors.append("approval.authorized_gates must be a non-empty list of Gates 0-8")
    if not _is_nonempty_string(approval.get("confirmation_reference")):
        errors.append("approval.confirmation_reference must identify the user approval")

    source = _required_mapping(card, "source", errors)
    if not _is_nonempty_string_list(source.get("raw_sources")):
        errors.append("source.raw_sources must be a non-empty list")
    if not _is_nonempty_string(source.get("license_or_allowed_use")):
        errors.append("source.license_or_allowed_use must be documented")

    worlds = _required_mapping(card, "worlds", errors)
    world_sets = {
        role: _world_set(worlds, role, errors)
        for role in (*DEVELOPMENT_WORLD_ROLES, "strict_test")
    }
    if not world_sets["train"]:
        errors.append("worlds.train must contain at least one development world")
    if not world_sets["validation"]:
        errors.append("worlds.validation must contain at least one validation world")
    if not world_sets["strict_test"]:
        errors.append("worlds.strict_test must contain at least one frozen test world")

    strict_test = world_sets["strict_test"]
    for role in DEVELOPMENT_WORLD_ROLES:
        overlap = strict_test & world_sets[role]
        if overlap:
            errors.append(
                f"strict test leakage: worlds.{role} overlaps strict_test: {sorted(overlap)}"
            )

    train_validation_overlap = world_sets["train"] & world_sets["validation"]
    if train_validation_overlap:
        errors.append(
            "train/validation worlds must be disjoint: "
            f"{sorted(train_validation_overlap)}"
        )

    trajectories = card.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        errors.append("trajectories must be a non-empty list")
        trajectories = []
    trajectory_ids: set[str] = set()
    allowed_development_worlds = set().union(
        *(world_sets[role] for role in DEVELOPMENT_WORLD_ROLES)
    )
    for index, trajectory in enumerate(trajectories):
        prefix = f"trajectories[{index}]"
        if not isinstance(trajectory, dict):
            errors.append(f"{prefix} must be an object")
            continue
        trajectory_id = trajectory.get("id")
        if not _is_nonempty_string(trajectory_id):
            errors.append(f"{prefix}.id must be a non-empty string")
        elif trajectory_id in trajectory_ids:
            errors.append(f"duplicate trajectory id: {trajectory_id}")
        else:
            trajectory_ids.add(trajectory_id)
        world = trajectory.get("world")
        if world not in allowed_development_worlds:
            errors.append(f"{prefix}.world is not listed as a development world: {world!r}")
        if world in strict_test:
            errors.append(f"{prefix}.world uses strict test world: {world!r}")
        if trajectory.get("split") not in {"train", "validation", "ssl"}:
            errors.append(f"{prefix}.split must be train, validation, or ssl")
        elif world not in world_sets[trajectory["split"]]:
            errors.append(
                f"{prefix}.world {world!r} is not listed in worlds.{trajectory['split']}"
            )
        if trajectory.get("independent") is not True:
            if not diagnostic_enabled or trajectory.get("independent") is not False:
                errors.append(f"{prefix}.independent must be true")
        for key in ("duration_s", "distance_m", "spatial_coverage_m"):
            value = trajectory.get(key)
            if not _is_number(value) or value <= 0:
                errors.append(f"{prefix}.{key} must be a positive number")

    sampling = _required_mapping(card, "sampling", errors)
    for key in ("raw_frame_count", "effective_sample_count"):
        value = sampling.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"sampling.{key} must be a positive integer")
    event_count = sampling.get("effective_structure_event_count")
    minimum_event_count = 0 if diagnostic_enabled else 1
    if (
        not isinstance(event_count, int)
        or isinstance(event_count, bool)
        or event_count < minimum_event_count
    ):
        qualifier = "non-negative" if diagnostic_enabled else "positive"
        errors.append(f"sampling.effective_structure_event_count must be a {qualifier} integer")
    interval = sampling.get("spatial_interval_m")
    if not _is_number(interval) or interval <= 0:
        errors.append("sampling.spatial_interval_m must be a positive number")
    if not _is_nonempty_string(sampling.get("rule")):
        errors.append("sampling.rule must explain spatial/temporal sampling")
    event_counts = sampling.get("structure_event_counts")
    if not isinstance(event_counts, dict) or not event_counts:
        errors.append("sampling.structure_event_counts must be a non-empty object")
    elif not all(
        _is_nonempty_string(key)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in event_counts.items()
    ):
        errors.append("sampling.structure_event_counts values must be non-negative integers")

    teacher = _required_mapping(card, "teacher", errors)
    for key in ("source", "valid_mask", "planner_consistency_plan"):
        if not _is_nonempty_string(teacher.get(key)):
            errors.append(f"teacher.{key} must be documented")

    split = _required_mapping(card, "split", errors)
    if split.get("world_disjoint") is not True:
        errors.append("split.world_disjoint must be true")
    if split.get("trajectory_disjoint") is not True:
        errors.append("split.trajectory_disjoint must be true")
    if not _is_nonempty_string(split.get("historical_pollution_audit")):
        errors.append("split.historical_pollution_audit must be documented")

    leakage = _required_mapping(card, "leakage_audit", errors)
    for flag in TEST_EXCLUSION_FLAGS:
        if leakage.get(flag) is not True:
            errors.append(f"leakage_audit.{flag} must be true")

    cost = _required_mapping(card, "estimated_cost", errors)
    for key in ("disk_gb", "wall_time_hours"):
        value = cost.get(key)
        if not _is_number(value) or value < 0:
            errors.append(f"estimated_cost.{key} must be a non-negative number")
    if not _is_nonempty_string(cost.get("compute")):
        errors.append("estimated_cost.compute must be documented")

    raw_count = sampling.get("raw_frame_count")
    effective_count = sampling.get("effective_sample_count")
    if (
        isinstance(raw_count, int)
        and isinstance(effective_count, int)
        and effective_count > raw_count
    ):
        warnings.append(
            "effective_sample_count exceeds raw_frame_count; document whether augmentation "
            "is being counted, because augmented samples are not independent observations"
        )

    if diagnostic_enabled:
        if approval.get("authorized_operations") != ["topology_replay"]:
            errors.append(
                "diagnostic_geometry_audit requires approval.authorized_operations "
                "exactly ['topology_replay']"
            )
        required_true = (
            "formal_dataset_false",
            "source_trajectory_independent",
            "pose_subset_nonindependent",
            "parameter_candidates_not_samples",
            "zero_training",
            "zero_model_selection",
        )
        for key in required_true:
            if diagnostic.get(key) is not True:
                errors.append(f"diagnostic_geometry_audit.{key} must be true")
        parameter_count = diagnostic.get("parameter_evaluation_count")
        if (
            not isinstance(parameter_count, int)
            or isinstance(parameter_count, bool)
            or parameter_count <= 0
        ):
            errors.append(
                "diagnostic_geometry_audit.parameter_evaluation_count must be a positive integer"
            )
        if any(trajectory.get("independent") is not False for trajectory in trajectories):
            errors.append(
                "diagnostic_geometry_audit trajectories must explicitly set independent=false"
            )
        if isinstance(raw_count, int) and effective_count != raw_count:
            errors.append(
                "diagnostic_geometry_audit effective_sample_count must equal raw_frame_count; "
                "parameter candidates are not samples"
            )
        if isinstance(parameter_count, int) and sampling.get("parameter_evaluation_count") != parameter_count:
            errors.append(
                "sampling.parameter_evaluation_count must match "
                "diagnostic_geometry_audit.parameter_evaluation_count"
            )
        if event_count != 0:
            errors.append(
                "diagnostic_geometry_audit effective_structure_event_count must be exactly 0"
            )
        if split.get("trajectory_disjoint") is not True:
            errors.append(
                "diagnostic_geometry_audit still requires split.trajectory_disjoint=true"
            )
    elif diagnostic is not None:
        errors.append("diagnostic_geometry_audit.enabled must be true when the block is present")

    return ValidationReport(not errors, tuple(errors), tuple(warnings))


def validate_annotation_plan(card: Mapping[str, Any]) -> ValidationReport:
    """Validate the additional contract required for AI-assisted annotation."""

    errors: list[str] = []
    annotation = card.get("annotation")
    if not isinstance(annotation, dict):
        return ValidationReport(False, ("annotation must be an object for AI annotation",))
    for key in (
        "labeler_name",
        "labeler_version",
        "prompt_reference",
        "input_bundle_contract",
        "output_schema",
        "conflict_policy",
        "abstain_policy",
    ):
        if not _is_nonempty_string(annotation.get(key)):
            errors.append(f"annotation.{key} must be documented")
    for key in ("planned_ai_sample_count", "planned_human_gold_count"):
        value = annotation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"annotation.{key} must be a positive integer")
    if annotation.get("strict_test_excluded") is not True:
        errors.append("annotation.strict_test_excluded must be true")
    if annotation.get("raw_responses_preserved") is not True:
        errors.append("annotation.raw_responses_preserved must be true")
    return ValidationReport(not errors, tuple(errors))


def _project_relative_path(project_root: Path, raw_path: Any, label: str) -> tuple[Path | None, str | None]:
    if not _is_nonempty_string(raw_path):
        return None, f"{label} must be a non-empty project-relative path"
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, f"{label} must stay inside the project and cannot contain '..'"
    resolved_root = project_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None, f"{label} resolves outside the project"
    return resolved, None


def build_run_id(spec: Mapping[str, Any]) -> str:
    """Build the standard deterministic run identifier."""

    return (
        f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    )


def validate_run_spec(spec: Mapping[str, Any]) -> ValidationReport:
    """Validate the scientific metadata of a run spec."""

    errors: list[str] = []

    if spec.get("schema_version") != RUN_SPEC_VERSION:
        errors.append(f"schema_version must be {RUN_SPEC_VERSION!r}")
    gate = spec.get("gate")
    if not isinstance(gate, int) or isinstance(gate, bool) or gate not in GATE_RESULT_DIRECTORIES:
        errors.append("gate must be an integer from 0 to 8")
    slug = spec.get("slug")
    if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
        errors.append("slug must use 3-81 lowercase letters, digits, '_' or '-'")
    date = spec.get("date")
    if not isinstance(date, str) or not DATE_PATTERN.fullmatch(date):
        errors.append("date must use YYYYMMDD")
    seed = spec.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("seed must be a non-negative integer")
    for key in ("question", "method", "baseline"):
        if not _is_nonempty_string(spec.get(key)):
            errors.append(f"{key} must be a non-empty string")
    authorization = _required_mapping(spec, "user_authorization", errors)
    if authorization.get("status") != "APPROVED":
        errors.append("user_authorization.status must be 'APPROVED'")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not _is_nonempty_string(authorization.get(key)):
            errors.append(f"user_authorization.{key} must be a non-empty string")
    operation = spec.get("operation")
    if operation not in OPERATION_GATE_RANGE:
        errors.append(
            "operation must be one of: " + ", ".join(sorted(OPERATION_GATE_RANGE))
        )
    elif isinstance(gate, int) and gate in GATE_RESULT_DIRECTORIES:
        minimum, maximum = OPERATION_GATE_RANGE[operation]
        if not minimum <= gate <= maximum:
            errors.append(
                f"operation {operation!r} is allowed only in Gate {minimum}"
                + (f"-{maximum}" if maximum != minimum else "")
            )
    if not _is_nonempty_string_list(spec.get("command")):
        errors.append("command must be a non-empty argv list; shell strings are not allowed")
    if not _is_nonempty_string_list(spec.get("acceptance_criteria")):
        errors.append("acceptance_criteria must be a non-empty list")
    if not _is_nonempty_string_list(spec.get("expected_evidence")):
        errors.append("expected_evidence must be a non-empty list")
    cost = spec.get("estimated_cost")
    if not isinstance(cost, dict):
        errors.append("estimated_cost must be an object")
    else:
        for key in ("disk_gb", "wall_time_hours"):
            value = cost.get(key)
            if not _is_number(value) or value < 0:
                errors.append(f"estimated_cost.{key} must be a non-negative number")
        if not _is_nonempty_string(cost.get("compute")):
            errors.append("estimated_cost.compute must be documented")

    return ValidationReport(not errors, tuple(errors))


def preflight(
    spec: Mapping[str, Any],
    status: Mapping[str, Any],
    project_root: Path,
) -> PreflightReport:
    """Run all non-mutating checks required before creating a V3 run."""

    spec_report = validate_run_spec(spec)
    errors = list(spec_report.errors)
    warnings = list(spec_report.warnings)

    if status.get("schema_version") != "master_plan_v3_status_v1":
        errors.append("project status schema_version is not master_plan_v3_status_v1")

    gate = spec.get("gate")
    run_id: str | None = None
    output_dir: Path | None = None
    if isinstance(gate, int) and gate in GATE_RESULT_DIRECTORIES:
        if status.get("current_gate") != gate:
            errors.append(
                f"run Gate {gate} does not match current Gate {status.get('current_gate')!r}"
            )
        try:
            run_id = build_run_id(spec)
        except (KeyError, TypeError):
            pass
        if run_id:
            output_dir = (
                project_root.resolve()
                / "results"
                / GATE_RESULT_DIRECTORIES[gate]
                / run_id
            )
            if output_dir.exists():
                errors.append(f"run directory already exists; refusing to overwrite: {output_dir}")

    running = status.get("running_experiment")
    if running not in (None, ""):
        errors.append(f"another experiment is marked running: {running!r}")

    operation = spec.get("operation")
    data_card_path: Path | None = None
    if operation in DATA_APPROVAL_OPERATIONS:
        data_card_path, path_error = _project_relative_path(
            project_root, spec.get("data_card"), "data_card"
        )
        if path_error:
            errors.append(path_error)
        elif data_card_path is not None:
            if not data_card_path.is_file():
                errors.append(f"data_card does not exist: {data_card_path}")
            else:
                try:
                    card = load_json(data_card_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"cannot read data_card: {exc}")
                else:
                    if operation == "sensor_smoke":
                        card_report = validate_sensor_smoke_card(card)
                    elif operation == "sensor_contract_pilot":
                        card_report = validate_sensor_contract_pilot_card(card)
                    elif operation == "audit" and card.get("schema_version") == "v3_scoped_inventory_card_v1":
                        # Inventory determines currently unknown counts. This
                        # narrow card never authorizes training/export and
                        # cannot fabricate durations for static ray samples.
                        from mtare_topo.governance_inventory import validate_scoped_inventory_card
                        card_report = validate_scoped_inventory_card(card)
                    elif operation == "audit" and card.get("schema_version") == "v3_scoped_coordinate_audit_card_v1":
                        from mtare_topo.governance_inventory import validate_scoped_coordinate_audit_card
                        card_report = validate_scoped_coordinate_audit_card(card)
                    elif operation == "data_export" and card.get("schema_version") == "v3_scoped_field_recovery_card_v1":
                        from mtare_topo.governance_field_recovery import validate_scoped_field_recovery_card
                        card_report = validate_scoped_field_recovery_card(card)
                    elif operation == "training" and card.get("schema_version") == "v3_scoped_point_axis_training_card_v1":
                        from mtare_topo.governance_point_axis import validate_point_axis_training_card
                        card_report = validate_point_axis_training_card(card)
                    else:
                        card_report = validate_data_card(card)
                    errors.extend(card_report.errors)
                    warnings.extend(card_report.warnings)
                    if operation in {"annotation_pilot", "ai_annotation"}:
                        annotation_report = validate_annotation_plan(card)
                        errors.extend(annotation_report.errors)
                        warnings.extend(annotation_report.warnings)
                    approval = card.get("approval", {})
                    authorized_operations = approval.get("authorized_operations")
                    if not isinstance(authorized_operations, list):
                        authorized_operations = []
                    authorized_gates = approval.get("authorized_gates")
                    if not isinstance(authorized_gates, list):
                        authorized_gates = []
                    card_id = card.get("card_id")
                    if operation not in authorized_operations:
                        errors.append(
                            f"data card {card_id!r} does not authorize operation {operation!r}"
                        )
                    if gate not in authorized_gates:
                        errors.append(
                            f"data card {card_id!r} does not authorize Gate {gate!r}"
                        )
    elif spec.get("data_card") not in (None, ""):
        warnings.append(
            "this operation does not require a data card; the supplied card will be snapshotted"
        )

    config_path_raw = spec.get("config_path")
    if config_path_raw not in (None, ""):
        config_path, path_error = _project_relative_path(
            project_root, config_path_raw, "config_path"
        )
        if path_error:
            errors.append(path_error)
        elif config_path is not None and not config_path.is_file():
            errors.append(f"config_path does not exist: {config_path}")

    return PreflightReport(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        run_id=run_id,
        output_dir=output_dir,
    )


def _copy_optional_project_file(
    spec: Mapping[str, Any], project_root: Path, key: str, destination: Path
) -> None:
    raw_path = spec.get(key)
    if raw_path in (None, ""):
        return
    source, error = _project_relative_path(project_root, raw_path, key)
    if error or source is None:
        raise ValueError(error)
    shutil.copy2(source, destination)


def create_run(
    spec: Mapping[str, Any],
    status: Mapping[str, Any],
    project_root: Path,
) -> Path:
    """Create an immutable run skeleton after a successful preflight.

    This function never executes the experiment command.
    """

    report = preflight(spec, status, project_root)
    if not report.passed or report.output_dir is None or report.run_id is None:
        raise ValueError("preflight failed: " + "; ".join(report.errors))

    output_dir = report.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    for child in ("config", "logs", "metrics", "previews", "artifacts"):
        (output_dir / child).mkdir()

    write_json(output_dir / "config" / "run_spec.json", dict(spec))
    write_json(output_dir / "config" / "status_snapshot.json", dict(status))
    _copy_optional_project_file(
        spec, project_root, "config_path", output_dir / "config" / "source_config"
    )
    _copy_optional_project_file(
        spec, project_root, "data_card", output_dir / "config" / "data_card.json"
    )

    command = spec["command"]
    (output_dir / "config" / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    write_json(
        output_dir / "config" / "environment.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
    )
    write_json(
        output_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": report.run_id,
            "state": "CREATED_NOT_EXECUTED",
            "note": "create_run.py created evidence directories only; command has not run.",
        },
    )
    (output_dir / "previews" / "README.md").write_text(
        "# Preview provenance\n\n"
        "Every preview must record split, world, trajectory/sample ID, method, units, "
        "and the hypothesis it supports or contradicts.\n",
        encoding="utf-8",
    )
    return output_dir


def update_status_fields(
    status: Mapping[str, Any],
    *,
    running_experiment: str | None | object = ...,
    latest_result: str | None = None,
    blocking_issue: str | None = None,
    next_action: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Update operational status without allowing a Gate transition."""

    updated = dict(status)
    if running_experiment is not ...:
        updated["running_experiment"] = running_experiment
    if latest_result is not None:
        updated["latest_result"] = latest_result
    if blocking_issue is not None:
        updated["blocking_issue"] = blocking_issue
    if next_action is not None:
        updated["next_action"] = next_action
    if updated_at is not None:
        updated["updated_at"] = updated_at
    return updated


def format_errors(errors: Iterable[str]) -> str:
    """Format validation errors for CLI output."""

    return "\n".join(f"- {error}" for error in errors)
