"""Independent exact-budget coordinate intervention training authorization."""
from copy import deepcopy
import re

from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_point_axis import (
    SCHEMA as POPULATION_SCHEMA, TRAINING as POPULATION_TRAINING,
    RESTRICTIONS as POPULATION_RESTRICTIONS, validate_point_axis_training_card,
)

SCHEMA = "v3_scoped_coordinate_control_training_card_v1"
VARIANTS = ("raw_coordinates", "mean_broadcast_coordinates")
TRAINING = {**POPULATION_TRAINING, "variants": list(VARIANTS)}
RESTRICTIONS = (*POPULATION_RESTRICTIONS, "no_offset_training", "no_C02_C10",
                "same_initial_parameters", "coordinate_intervention_only",
                "no_equal_initial_output_requirement", "no_scientific_gate_promotion")


def validate_coordinate_control_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("coordinate control card must be an object",))
    if card.get("schema_version") != SCHEMA:
        errors.append("invalid independent coordinate control schema")
    training = card.get("training")
    if not isinstance(training, dict) or set(training) != set(TRAINING):
        errors.append("exact coordinate control training contract required")
        training = {}
    for key, expected in TRAINING.items():
        if type(training.get(key)) is not type(expected) or training[key] != expected:
            errors.append(f"coordinate control training.{key} is not frozen value")
    # Reuse exact-population/source/approval validation only. No original card
    # or approval is modified, and no operation is promoted: both are training.
    population = deepcopy(card)
    population["schema_version"] = POPULATION_SCHEMA
    population["training"] = deepcopy(POPULATION_TRAINING)
    errors.extend(validate_point_axis_training_card(population).errors)
    for restriction in RESTRICTIONS:
        if not isinstance(card.get("restrictions"), dict) or card["restrictions"].get(restriction) is not True:
            errors.append(f"coordinate control restriction missing: {restriction}")
    if type(card.get("visible_fragment_count")) is not int or card["visible_fragment_count"] != 1452:
        errors.append("exact same 1452 existing visible fragments required")
    if card.get("compute_device") != "cuda":
        errors.append("coordinate control is CUDA-first; no silent CPU experiment substitution")
    roots = card.get("source_roots")
    if not isinstance(roots, dict) or set(roots) != {"sensor", "teacher"}:
        errors.append("exact sensor and teacher source roots required")
    else:
        for root in roots.values():
            if (not _relative_source(root) or root.split("/")[-1] != "fit"
                    or re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)", root)):
                errors.append("source roots must be relative fit roots")
    seals = card.get("source_seals")
    sources = card.get("sealed_sources")
    sources = sources if isinstance(sources, dict) else {}
    if not isinstance(seals, dict) or set(seals) != {"sensor", "teacher", "reference"}:
        errors.append("sensor/teacher/reference seals required")
    else:
        for seal in seals.values():
            if not _relative_source(seal) or not _digest(sources.get(seal) if isinstance(seal, str) else None):
                errors.append("every source seal must be an exact hashed relative source")
    if not _relative_source(card.get("prediction_reference_root")):
        errors.append("exact relative frozen prediction reference root required")
    return ValidationReport(not errors, tuple(errors))
