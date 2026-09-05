from __future__ import annotations

import unittest

from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES,
    ExitGeometryToken,
    GeometricSemanticObservation,
    StructuralEvent,
)


class GeometricSemanticsTest(unittest.TestCase):
    def test_typed_observation_normalizes_axis_and_wraps_heading(self) -> None:
        token = ExitGeometryToken(heading_robot_deg=-10.0, opening_width_m=2.0, vertical_profile=(1.0, 2.0), descriptor=(0.1, 0.2), confidence=0.9)
        probabilities = {name: 0.0 for name in EVENT_NAMES}
        probabilities[StructuralEvent.JUNCTION.value] = 1.0
        observation = GeometricSemanticObservation(
            event_probabilities=probabilities,
            local_axis=(2.0, 0.0, 0.0),
            width_m=5.0,
            height_m=4.0,
            slope_deg=3.0,
            curvature_per_m=0.02,
            place_descriptor=(0.2, 0.3),
            exit_tokens=(token,),
            uncertainty=0.1,
        )
        self.assertEqual(token.heading_robot_deg, 350.0)
        self.assertEqual(observation.local_axis, (1.0, 0.0, 0.0))
        self.assertEqual(observation.event, StructuralEvent.JUNCTION)
        self.assertEqual(observation.to_dict()["exit_tokens"][0]["opening_width_m"], 2.0)

    def test_invalid_probability_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GeometricSemanticObservation(
                event_probabilities={"junction": 1.0},
                local_axis=(1.0, 0.0, 0.0),
                width_m=1.0,
                height_m=1.0,
                slope_deg=0.0,
                curvature_per_m=0.0,
                place_descriptor=(1.0,),
                exit_tokens=(),
                uncertainty=0.0,
            )


if __name__ == "__main__":
    unittest.main()
