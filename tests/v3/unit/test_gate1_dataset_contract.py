import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "contracts" / "gate1_dataset_contract_draft_v1.json"


class Gate1DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_design_does_not_authorize_gate1_operations(self):
        authorization = self.contract["authorization"]
        self.assertTrue(authorization["design_only"])
        for operation in ("data_generation", "annotation", "training", "simulation"):
            self.assertFalse(authorization[operation])
        self.assertEqual(self.contract["project_state"]["current_gate"], 0)
        self.assertEqual(self.contract["project_state"]["v3_dataset"], "NONE")

    def test_mtare_benchmark_is_development_prohibited(self):
        self.assertEqual(
            self.contract["pool_policy"]["MTARE_BENCHMARK_ONLY"],
            ["final_parity_benchmark_only"],
        )
        forbidden = set(self.contract["benchmark_forbidden_operations"])
        required = {
            "supervised_training",
            "self_supervised_training",
            "normalization",
            "ai_annotation",
            "human_development_annotation",
            "augmentation_tuning",
            "teacher_calibration",
            "threshold_selection",
            "checkpoint_selection",
            "failure_driven_model_revision",
        }
        self.assertEqual(forbidden, required)

    def test_pilot_is_five_world_contract_check_not_training(self):
        pilot = self.contract["pilot"]
        self.assertEqual(pilot["purpose"], "contract_validation_only")
        self.assertFalse(pilot["allowed_training"])
        self.assertEqual(len(pilot["topology_parents"]), 5)
        self.assertEqual(pilot["target_mesh_realizations_total"], 10)
        expected = (
            len(pilot["topology_parents"])
            * pilot["matched_canonical_anchors_per_topology"]
            * pilot["geometry_variants_per_topology"]
            * pilot["sensor_views_per_geometry_anchor"]
        )
        self.assertEqual(pilot["target_observations_total"], expected)

    def test_student_input_is_causal_and_oracle_free(self):
        student = self.contract["student_input_candidate"]
        self.assertEqual(student["route"], "terrain_relative_causal_2p5d_bev")
        self.assertEqual(len(student["channels_per_view"]), 8)
        self.assertIn("ray_free_evidence", student["channels_per_view"])
        self.assertEqual(
            set(student["forbidden_features"]),
            {
                "absolute_position",
                "world_id",
                "trajectory_id",
                "future_observation",
                "oracle_map",
                "teacher_tensor",
                "oracle_global_tng",
                "topology_parent_id",
                "geometry_variant_id",
                "unobservable_connectivity",
            },
        )

    def test_primary_learning_is_objective_supervision(self):
        learning = self.contract["learning_policy"]
        self.assertEqual(
            learning["primary"], "objective_supervised_lightweight_bev_polar_model"
        )
        self.assertFalse(learning["ai_core_teacher"])
        self.assertFalse(learning["contrastive_core_method"])
        targets = set(self.contract["objective_teacher"]["targets"])
        self.assertTrue(
            {
                "direction_traversable_R",
                "reachable_distance_D",
                "direction_valid_mask",
                "circular_exit_components",
                "local_exit_connectivity_G",
            }.issubset(targets)
        )
        self.assertFalse(self.contract["objective_teacher"]["invalid_is_negative"])

    def test_revised_capacity_and_b1_sequence_are_frozen(self):
        learning = self.contract["learning_policy"]
        self.assertEqual(learning["implementation_sequence"][0], "B1_cano_like_reproduction")
        capacity = self.contract["production_capacity_candidate"]
        self.assertEqual(capacity["train_topology_parents"], 80)
        self.assertEqual(capacity["validation_topology_parents"], 10)
        self.assertEqual(capacity["development_test_topology_parents"], 10)
        self.assertEqual(capacity["nested_train_topology_parent_counts"], [20, 40, 80])
        expected = (
            (
                capacity["train_topology_parents"]
                + capacity["validation_topology_parents"]
                + capacity["development_test_topology_parents"]
            )
            * capacity["matched_canonical_anchors_per_topology"]
            * capacity["geometry_variants_per_topology"]
            * capacity["sensor_views_per_geometry_anchor"]
        )
        self.assertEqual(capacity["estimated_observations"], expected)
        self.assertEqual(capacity["mesh_realizations_total"], 200)

    def test_tng_descendants_cannot_leak_or_form_false_positive_pairs(self):
        policy = self.contract["tng_pair_policy"]
        self.assertTrue(policy["all_descendants_of_topology_parent_same_split"])
        self.assertFalse(policy["complete_tng_student_visible"])
        self.assertEqual(policy["unobservable_edges"], "UNKNOWN_MASKED")
        self.assertTrue(
            policy["positive_requires_robot_footprint_and_planner_connectivity_equivalence"]
        )
        self.assertFalse(policy["connectivity_changing_geometry_is_positive"])
        self.assertEqual(policy["paired_contrastive_training"], "conditional_A1_not_primary_M1")


if __name__ == "__main__":
    unittest.main()
