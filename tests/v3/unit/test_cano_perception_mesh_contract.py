import unittest

import numpy as np

from mtare_topo.data.cano_perception_mesh_contract import floor_support_contract


def test_floor_support_contract_reports_nonfinite_and_height_failures():
    result = floor_support_contract([0.0, 1.0, 2.0], [0.1, np.inf, 2.4], 0.25)
    assert result["supported_count"] == 1
    assert result["unsupported_indices"] == [1, 2]
    assert result["passed"] is False


def test_floor_support_contract_passes_boundary():
    result = floor_support_contract([0.0, 1.0], [0.25, 0.75], 0.25)
    assert result["passed"] is True

from mtare_topo.data.cano_perception_mesh_contract import (
    SENTINEL_PARENT_IDS,
    batch_mesh_audit,
    geometric_replay_pair_audit,
    immutable_asset_batch_audit,
    immutable_full_parent_asset_batch_audit,
    zero_area_triangle_sanitation,
    mesh_array_audit,
    replay_pair_audit,
    select_train_sentinels,
)


class CanoPerceptionMeshContractTest(unittest.TestCase):
    def _parents(self):
        return [
            {
                "parent_id": parent_id,
                "recipe_stratum_id": f"R{index:02d}",
                "accepted_rank_in_recipe": 1,
                "split": "train",
                "reserved_geometry_seed": 720000 + 100 * index + 1,
            }
            for index, parent_id in enumerate(SENTINEL_PARENT_IDS, start=1)
        ]

    def test_selects_only_fixed_rank_one_train_sentinels(self):
        parents = self._parents() + [
            {
                "parent_id": "S01_flat_tree_small_C02",
                "recipe_stratum_id": "R01",
                "accepted_rank_in_recipe": 2,
                "split": "train",
                "reserved_geometry_seed": 720102,
            }
        ]
        self.assertEqual(
            [item["parent_id"] for item in select_train_sentinels(parents)],
            list(SENTINEL_PARENT_IDS),
        )

    def test_rejects_validation_sentinel(self):
        parents = self._parents()
        parents[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "train-only"):
            select_train_sentinels(parents)

    def test_mesh_array_contract(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        axis = np.asarray([[0, 0, 0, 1, 0, 0, 5, 1, 0], [10, 0, 0, 1, 0, 0, 5, 1, 1]])
        splines = np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
        audit = mesh_array_audit(vertices, triangles, [2], axis, [0, 1], splines)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["largest_component_fraction"], 1.0)

    def test_mesh_array_rejects_missing_tunnel_axis(self):
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        axis = np.asarray([[0, 0, 0, 1, 0, 0, 5, 1, 0]], dtype=np.float64)
        audit = mesh_array_audit(vertices, triangles, [1], axis, [0, 1], vertices)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["every_source_tunnel_has_axis"])

    def test_replay_and_batch_contract(self):
        mesh_audit = {"passed": True}
        records = []
        for parent_id in SENTINEL_PARENT_IDS:
            primary = {
                "parent_id": parent_id,
                "graph_identity": "g",
                "spline_identity": "s",
                "effective_geometry_parameter_sha256": "p",
                "mesh_sha256": "m",
                "mesh_audit": mesh_audit,
            }
            replay = dict(primary)
            pair = replay_pair_audit(primary, replay)
            self.assertTrue(pair["passed"])
            records.append(
                {
                    "parent_id": parent_id,
                    "split": "train",
                    "primary": primary,
                    "replay": replay,
                    "replay_audit": pair,
                }
            )
        self.assertTrue(batch_mesh_audit(records)["passed"])

    def test_noise_bounded_geometric_replay_accepts_different_obj_hash(self):
        base = {
            "parent_id": "S01",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_audit": {"passed": True},
        }
        primary = dict(base, mesh_sha256="primary")
        replay = dict(base, mesh_sha256="replay")
        audit = geometric_replay_pair_audit(
            primary,
            replay,
            axis_exact_equal=True,
            primary_vertex_count=100,
            replay_vertex_count=100,
            primary_triangle_count=200,
            replay_triangle_count=200,
            primary_to_replay_nearest_vertex_maximum_m=0.60,
            replay_to_primary_nearest_vertex_maximum_m=0.58,
            aabb_endpoint_maximum_coordinate_difference_m=0.12,
            surface_area_relative_difference=0.0005,
        )
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["metrics"]["obj_sha256_equal_recorded_not_required"])

    def test_noise_bounded_geometric_replay_rejects_distance_above_bound(self):
        base = {
            "parent_id": "S01",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base,
            base,
            axis_exact_equal=True,
            primary_vertex_count=100,
            replay_vertex_count=100,
            primary_triangle_count=200,
            replay_triangle_count=200,
            primary_to_replay_nearest_vertex_maximum_m=0.751,
            replay_to_primary_nearest_vertex_maximum_m=0.58,
            aabb_endpoint_maximum_coordinate_difference_m=0.12,
            surface_area_relative_difference=0.0005,
        )
        self.assertFalse(audit["passed"])

    def test_m0r2_accepts_triangle_count_difference_within_0_01_percent(self):
        base = {
            "parent_id": "S02",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base,
            base,
            axis_exact_equal=True,
            primary_vertex_count=78650,
            replay_vertex_count=78650,
            primary_triangle_count=157298,
            replay_triangle_count=157296,
            primary_to_replay_nearest_vertex_maximum_m=0.692104,
            replay_to_primary_nearest_vertex_maximum_m=0.554699,
            aabb_endpoint_maximum_coordinate_difference_m=0.084740,
            surface_area_relative_difference=0.00100623,
            triangle_count_relative_limit=0.0001,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["metrics"]["triangle_count_absolute_difference"], 2)
        self.assertLessEqual(
            audit["metrics"]["triangle_count_relative_difference"], 0.0001
        )

    def test_m0r2_rejects_triangle_count_difference_above_0_01_percent(self):
        base = {
            "parent_id": "S02",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base,
            base,
            axis_exact_equal=True,
            primary_vertex_count=1000,
            replay_vertex_count=1000,
            primary_triangle_count=10000,
            replay_triangle_count=9998,
            primary_to_replay_nearest_vertex_maximum_m=0.60,
            replay_to_primary_nearest_vertex_maximum_m=0.58,
            aabb_endpoint_maximum_coordinate_difference_m=0.12,
            surface_area_relative_difference=0.0005,
            triangle_count_relative_limit=0.0001,
        )
        self.assertFalse(audit["passed"])

    def test_m0r3_uses_point_to_surface_instead_of_nearest_vertex_for_pass(self):
        base = {
            "parent_id": "S08",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base,
            base,
            axis_exact_equal=True,
            primary_vertex_count=185205,
            replay_vertex_count=185205,
            primary_triangle_count=370418,
            replay_triangle_count=370416,
            primary_to_replay_nearest_vertex_maximum_m=0.576404,
            replay_to_primary_nearest_vertex_maximum_m=0.844986,
            primary_to_replay_surface_maximum_m=0.494647,
            replay_to_primary_surface_maximum_m=0.502614,
            aabb_endpoint_maximum_coordinate_difference_m=0.012009,
            surface_area_relative_difference=0.00050641,
            triangle_count_relative_limit=0.0001,
        )
        self.assertTrue(audit["passed"])
        self.assertNotIn(
            "replay_to_primary_nearest_vertex_maximum_within_0_75m",
            audit["checks"],
        )
        self.assertEqual(
            audit["thresholds"]["pass_fail_distance_metric"],
            "vertex_to_opposite_triangle_surface_maximum",
        )

    def test_m0r3_rejects_point_to_surface_above_unchanged_bound(self):
        base = {
            "parent_id": "S08",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base,
            base,
            axis_exact_equal=True,
            primary_vertex_count=100,
            replay_vertex_count=100,
            primary_triangle_count=200,
            replay_triangle_count=200,
            primary_to_replay_nearest_vertex_maximum_m=0.50,
            replay_to_primary_nearest_vertex_maximum_m=0.50,
            primary_to_replay_surface_maximum_m=0.751,
            replay_to_primary_surface_maximum_m=0.50,
            aabb_endpoint_maximum_coordinate_difference_m=0.10,
            surface_area_relative_difference=0.001,
            triangle_count_relative_limit=0.0001,
        )
        self.assertFalse(audit["passed"])

    def test_m0r4_accepts_unified_vertex_and_triangle_discretization(self):
        base = {
            "parent_id": "S07",
            "graph_identity": "g",
            "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p",
            "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base, base,
            axis_exact_equal=True,
            primary_vertex_count=156110,
            replay_vertex_count=156109,
            primary_triangle_count=312234,
            replay_triangle_count=312232,
            primary_to_replay_nearest_vertex_maximum_m=0.598829,
            replay_to_primary_nearest_vertex_maximum_m=0.600387,
            primary_to_replay_surface_maximum_m=0.462879,
            replay_to_primary_surface_maximum_m=0.511900,
            aabb_endpoint_maximum_coordinate_difference_m=0.092010,
            surface_area_relative_difference=0.00013279,
            vertex_count_relative_limit=0.0001,
            triangle_count_relative_limit=0.0001,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["metrics"]["vertex_count_absolute_difference"], 1)

    def test_m0r4_rejects_vertex_difference_above_0_01_percent(self):
        base = {
            "parent_id": "S07", "graph_identity": "g", "spline_identity": "s",
            "operation_trace_sha256": "o",
            "effective_geometry_parameter_sha256": "p", "mesh_sha256": "m",
            "mesh_audit": {"passed": True},
        }
        audit = geometric_replay_pair_audit(
            base, base,
            axis_exact_equal=True,
            primary_vertex_count=10000,
            replay_vertex_count=9998,
            primary_triangle_count=20000,
            replay_triangle_count=20000,
            primary_to_replay_nearest_vertex_maximum_m=0.5,
            replay_to_primary_nearest_vertex_maximum_m=0.5,
            primary_to_replay_surface_maximum_m=0.5,
            replay_to_primary_surface_maximum_m=0.5,
            aabb_endpoint_maximum_coordinate_difference_m=0.1,
            surface_area_relative_difference=0.001,
            vertex_count_relative_limit=0.0001,
            triangle_count_relative_limit=0.0001,
        )
        self.assertFalse(audit["passed"])

    def test_m0f_accepts_ten_primary_assets_and_zero_replay(self):
        records = []
        for index, parent_id in enumerate(SENTINEL_PARENT_IDS):
            records.append({
                "parent_id": parent_id,
                "split": "train",
                "primary": {
                    "mesh_sha256": f"{index + 1:064x}",
                    "mesh_audit": {"passed": True},
                    "identity_checks": {"graph": True, "splines": True},
                },
            })
        audit = immutable_asset_batch_audit(records)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["checks"]["zero_replay_records"])

    def test_m0f_rejects_any_replay_record(self):
        records = []
        for index, parent_id in enumerate(SENTINEL_PARENT_IDS):
            records.append({
                "parent_id": parent_id,
                "split": "train",
                "primary": {
                    "mesh_sha256": f"{index + 1:064x}",
                    "mesh_audit": {"passed": True},
                    "identity_checks": {"graph": True},
                },
            })
        records[0]["replay"] = {}
        self.assertFalse(immutable_asset_batch_audit(records)["passed"])

    def test_m1_accepts_exact_100_parent_manifest(self):
        expected = []
        records = []
        for recipe in range(10):
            for candidate in range(10):
                split = "train" if candidate < 8 else ("validation" if candidate == 8 else "development_test")
                parent = {"parent_id": f"R{recipe:02d}_C{candidate:02d}", "split": split, "recipe_stratum_id": f"R{recipe:02d}"}
                expected.append(parent)
                records.append({
                    **parent,
                    "primary": {
                        "mesh_sha256": f"{recipe * 10 + candidate + 1:064x}",
                        "mesh_audit": {"passed": True},
                        "identity_checks": {
                            "graph_matches_sealed_source": True,
                            "operation_trace_matches_sealed_source": True,
                            "parent_identity_matches_v2r": True,
                            "splines_match_sealed_source": True,
                        },
                    },
                })
        audit = immutable_full_parent_asset_batch_audit(records, expected)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["split_counts"], {"train": 80, "validation": 10, "development_test": 10})

    def test_m1_rejects_order_drift_replay_and_duplicate_hash(self):
        expected = []
        records = []
        for recipe in range(10):
            for candidate in range(10):
                split = "train" if candidate < 8 else ("validation" if candidate == 8 else "development_test")
                parent = {"parent_id": f"R{recipe:02d}_C{candidate:02d}", "split": split, "recipe_stratum_id": f"R{recipe:02d}"}
                expected.append(parent)
                records.append({**parent, "primary": {"mesh_sha256": f"{recipe * 10 + candidate + 1:064x}", "mesh_audit": {"passed": True}, "identity_checks": {"graph_matches_sealed_source": True, "operation_trace_matches_sealed_source": True, "parent_identity_matches_v2r": True, "splines_match_sealed_source": True}}})
        records[0], records[1] = records[1], records[0]
        records[0]["replay"] = {}
        records[1]["primary"]["mesh_sha256"] = records[0]["primary"]["mesh_sha256"]
        self.assertFalse(immutable_full_parent_asset_batch_audit(records, expected)["passed"])

    def test_m1r_sanitation_removes_only_collinear_triangle(self):
        vertices = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        triangles = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        keep, evidence = zero_area_triangle_sanitation(vertices, triangles)
        self.assertEqual(keep.tolist(), [False, True])
        self.assertEqual(evidence["removed_triangle_count"], 1)
        self.assertEqual(evidence["removed_surface_area_sum_m2"], 0.0)
        self.assertEqual(evidence["triangle_count_after"], 1)

    def test_m1r_sanitation_preserves_valid_triangles(self):
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        keep, evidence = zero_area_triangle_sanitation(vertices, triangles)
        self.assertEqual(keep.tolist(), [True])
        self.assertEqual(evidence["removed_triangle_count"], 0)


if __name__ == "__main__":
    unittest.main()
