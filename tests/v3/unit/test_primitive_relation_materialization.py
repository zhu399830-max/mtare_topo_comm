from __future__ import annotations

import unittest

from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction


class PrimitiveRelationMaterializationTest(unittest.TestCase):
    def _document(self) -> dict:
        endpoint = lambda primitive, index: {
            "primitive_id": primitive, "endpoint_index": index, "node_id": "n",
            "xyz_m": [float(index), 0.0, 0.0], "composition_anchor_xyz_m": [0.5, 0.0, 0.0],
        }
        primitive = lambda identity: {
            "primitive_id": identity, "centerline_xyz_m": [[0, 0, 0], [1, 0, 0]],
            "endpoint_half_axes_m": [[2, 1], [2, 1]], "endpoint_shape_exponent": [2, 6],
        }
        return {
            "schema_version": "primitive_relation_realized_construction_v1",
            "base_construction": {
                "schema_version": "primitive_construction_graph_v1", "coordinate_frame": "world",
                "endpoint_attachment_mode": "free_space_overlap", "node_degree_source": "edge_incidence",
                "primitives": [{"primitive_id": "p0"}, {"primitive_id": "p1"}],
                "composition_operations": [{
                    "node_id": "n", "anchor_xyz_m": [0.5, 0, 0],
                    "member_endpoints": [endpoint("p0", 1), endpoint("p1", 0)],
                }],
            },
            "realized_primitives": [primitive("p0"), primitive("p1")],
        }

    def test_load_preserves_primitive_order_and_endpoint_identity(self) -> None:
        construction, primitives = load_p1a_realized_construction(self._document())
        self.assertEqual(tuple(value.primitive_id for value in primitives), ("p0", "p1"))
        self.assertEqual(construction.compositions[0].member_endpoints[1].primitive_id, "p1")
        self.assertEqual(construction.endpoint_attachment_mode, "free_space_overlap")

    def test_identity_order_drift_fails(self) -> None:
        document = self._document()
        document["realized_primitives"].reverse()
        with self.assertRaisesRegex(ValueError, "identity/order"):
            load_p1a_realized_construction(document)


if __name__ == "__main__":
    unittest.main()
