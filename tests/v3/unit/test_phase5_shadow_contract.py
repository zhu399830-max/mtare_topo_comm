from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
for value in (ROOT / "src", ROOT / "tools/v3"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import execute_cano_c08_topological_planner_shadow_v1 as shadow


class Phase5ShadowContractTests(unittest.TestCase):
    def test_frozen_counts_and_methods(self):
        self.assertEqual(len(shadow.WORLDS), 3)
        self.assertEqual(len(shadow.METHODS), 5)
        self.assertEqual(shadow.EXPECTED_FRAMES, 4773)
        self.assertEqual(shadow.EXPECTED_CYCLES, 23865)

    def test_shadow_reuses_sealed_c08_and_excludes_c09_c10(self):
        self.assertTrue(shadow.SOURCE_RUN.is_dir())
        self.assertIn("c08", str(shadow.SOURCE_RUN).lower())
        self.assertNotIn("c09", str(shadow.SOURCE_RUN).lower())
        self.assertNotIn("c10", str(shadow.SOURCE_RUN).lower())

    def test_planner_constants_have_declared_sources(self):
        self.assertEqual(shadow.PLANNER_CONFIG.waypoint_lookahead_m, 4.0)
        self.assertEqual(shadow.PLANNER_CONFIG.graph_cost_scale_m, 20.0)
        self.assertEqual(shadow.PLANNER_CONFIG.minimum_frontier_confidence, 0.05)

    def test_proposal_preserves_scope_and_is_not_execution_approved(self):
        spec = json.loads((ROOT / "configs/v3/gate5/cano_c08_topological_planner_shadow_v1.proposal.json").read_text())
        card = json.loads((ROOT / "configs/v3/gate5/data_cards/cano_c08_topological_planner_shadow_v1.proposal.json").read_text())
        self.assertEqual(spec["gate"], 5)
        self.assertEqual(spec["operation"], "shadow")
        self.assertEqual(spec["user_authorization"]["status"], "PENDING")
        self.assertEqual(card["approval"]["status"], "PENDING")
        self.assertEqual(card["sampling"]["raw_frame_count"], 4773)
        self.assertEqual(card["sampling"]["effective_sample_count"], 4773)


if __name__ == "__main__":
    unittest.main()
