from dataclasses import replace

import pytest

from mtare_topo.semantics.gse_local_structure_v2 import (
    LocalOpeningV2, LocalStructureObservationV2, SourceFrameV2, StructureInstanceV2,
)


def observation():
    structure = StructureInstanceV2(0, (1., 2., 3.), (.1, .7, .1, .1), .4, True)
    opening = LocalOpeningV2(0, None, (1., 0., 0.), None, None, .8, (.7, .3), .5, True)
    return LocalStructureObservationV2("runtime-session", 0, "current_sensor", tuple(
        SourceFrameV2(f"frame-{i}", i) for i in range(5)), 4, "a" * 64, (structure,), (opening,))


def test_unknown_is_not_filled_and_existence_has_one_authority():
    result = observation()
    assert result.openings[0].width_m is None
    assert result.openings[0].position_m is None
    assert result.structures[0].existence_probability == pytest.approx(.9)
    assert not result.uncertainty_calibrated


@pytest.mark.parametrize("change", [
    {"current_source_index": 3}, {"current_source_index": 5},
    {"input_binding_sha256": "guess"}, {"decision_index": True},
    {"coordinate_frame": ""}, {"schema_version": "v1"},
    {"source_frames": tuple(SourceFrameV2("same", i) for i in range(5))},
    {"source_frames": tuple(SourceFrameV2(str(i), 4-i) for i in range(5))},
    {"structures": []}, {"structures": observation().structures * 2},
])
def test_invalid_binding_or_population_rejected(change):
    with pytest.raises(ValueError):
        replace(observation(), **change)


@pytest.mark.parametrize("change", [
    {"width_m": 0.}, {"height_m": float("nan")}, {"direction": (0., 0., 0.)},
    {"query_index": 64}, {"existence_probability": True}, {"structure_membership": (.2, .2)},
])
def test_invalid_opening_rejected(change):
    with pytest.raises(ValueError):
        replace(observation().openings[0], **change)


def test_membership_population_cannot_silently_change():
    with pytest.raises(ValueError, match="membership"):
        replace(observation(), structures=())


def test_empty_output_is_not_a_detection():
    result = replace(observation(), structures=(), openings=())
    assert not result.structures and not result.openings
