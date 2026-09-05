import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "contracts" / "isaac_generation_contract_draft_v1.json"


class IsaacGenerationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_design_does_not_authorize_external_or_compute_actions(self):
        auth = self.contract["authorization"]
        self.assertTrue(auth["design_only"])
        for key in ("third_party_download", "isaac_launch", "data_generation", "training"):
            self.assertFalse(auth[key])

    def test_generator_reuse_requires_verified_license_and_commit(self):
        acquisition = self.contract["generator_acquisition"]
        self.assertFalse(acquisition["legacy_repository_and_license_verified"])
        required = set(acquisition["required_before_reuse"])
        self.assertTrue({"pinned_commit", "license_file", "attribution_plan"}.issubset(required))
        self.assertEqual(
            acquisition["fallback"], "independent_reimplementation_from_published_method"
        )

    def test_common_geometry_and_sensor_parity_block_batch_generation(self):
        self.assertTrue(
            self.contract["generator_bundle"]["same_canonical_geometry_for_isaac_and_gazebo"]
        )
        sensor = self.contract["sensor_policy"]
        self.assertTrue(sensor["same_sensor_for_B1_and_M1"])
        self.assertGreaterEqual(sensor["isaac_gazebo_parity_fixed_pose_count_min"], 20)
        self.assertTrue(sensor["batch_generation_blocked_until_parity_pass"])

    def test_capacity_arithmetic_and_nested_world_curve(self):
        development = self.contract["data_capacity"]["development"]
        per_topology = (
            development["canonical_anchors_per_topology"]
            * development["geometry_variants_per_topology"]
            * development["sensor_views_per_geometry_anchor"]
        )
        self.assertEqual(
            development["train_observations"],
            development["train_topology_parents"] * per_topology,
        )
        self.assertEqual(
            development["validation_observations"],
            development["validation_topology_parents"] * per_topology,
        )
        self.assertEqual(
            development["development_test_observations"],
            development["development_test_topology_parents"] * per_topology,
        )
        self.assertEqual(
            development["total_observations"],
            development["train_observations"]
            + development["validation_observations"]
            + development["development_test_observations"],
        )
        self.assertEqual(
            development["nested_train_topology_parent_counts"], [20, 40, 80]
        )
        self.assertEqual(development["train_mesh_realizations"], 160)
        self.assertEqual(development["validation_mesh_realizations"], 20)
        self.assertEqual(development["development_test_mesh_realizations"], 20)

    def test_tng_seed_split_and_oracle_boundaries_are_frozen(self):
        tng = self.contract["tng_generation"]
        self.assertEqual(tng["split_atomic_unit"], "topology_parent_tng")
        self.assertTrue(tng["all_descendant_geometry_trajectory_and_observation_same_split"])
        self.assertEqual(
            set(tng["separate_seed_namespaces"]),
            {
                "topology_seed",
                "geometry_seed",
                "clutter_seed",
                "trajectory_seed",
                "sensor_seed",
            },
        )
        pair = self.contract["oracle_and_pair_policy"]
        self.assertFalse(pair["oracle_global_tng_student_visible"])
        self.assertEqual(pair["unobservable_connectivity_label"], "UNKNOWN_MASKED")
        self.assertTrue(pair["positive_requires_operational_connectivity_equivalence"])
        self.assertFalse(pair["connectivity_changing_variant_is_positive"])
        self.assertIn("planner_rollout", pair["operational_equivalence_checks"])

    def test_m1_does_not_learn_topology_or_planning_in_first_version(self):
        m1 = self.contract["learning"]["M1"]
        self.assertEqual(m1["trained_outputs"], ["R32", "D32", "U32"])
        self.assertFalse(m1["ai_auxiliary_enabled"])
        self.assertFalse(m1["contrastive_auxiliary_enabled"])
        self.assertTrue(
            {"G_local", "node_score", "waypoint", "planning_policy"}.issubset(
                set(m1["deferred_outputs"])
            )
        )
        self.assertFalse(self.contract["topology"]["learned_in_first_version"])
        self.assertFalse(self.contract["topology"]["role_similarity_alone_allows_merge"])


if __name__ == "__main__":
    unittest.main()
