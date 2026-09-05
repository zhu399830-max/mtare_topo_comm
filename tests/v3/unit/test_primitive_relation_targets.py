from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.primitive_frame_support import PrimitiveFrameSupport
from mtare_topo.data.primitive_relation_targets import (
    primitive_relation_targets_from_frame_support,
    primitive_relation_window_targets,
)
from mtare_topo.teacher.primitive_construction_supervisor import (
    EndpointComposition,
    PrimitiveConstructionGraph,
    PrimitiveEndpoint,
)


def _endpoint(primitive: str, endpoint: int, node: str) -> PrimitiveEndpoint:
    return PrimitiveEndpoint(primitive, endpoint, node, (float(endpoint), 0., 0.))


class PrimitiveRelationTargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        p0e1 = _endpoint("p0", 1, "n")
        p1e0 = _endpoint("p1", 0, "n")
        self.construction = PrimitiveConstructionGraph(
            coordinate_frame="world", primitives=(),
            compositions=(EndpointComposition("n", (p0e1, p1e0)),),
        )

    def test_attachment_overlap_and_dustbin_are_distinct(self) -> None:
        empty = [()] * 1440
        frames = []
        for frame in range(5):
            row = list(empty)
            row[10] = (0,)
            if frame < 4: row[730] = (2,)  # same azimuth column 10, past-only
            if frame >= 2: row[20] = (1,)
            frames.append(row)
        target = primitive_relation_window_targets(
            construction=self.construction, primitive_ids=("p0", "p1", "p2"),
            source_sets_by_frame=frames, maximum_slots=4,
        )
        np.testing.assert_array_equal(target.primitive_index, [0, 1, 2, -1])
        self.assertEqual(target.endpoint_attachment[0, 1, 1, 0], 1)
        self.assertEqual(target.disconnected_angular_overlap[0, 2], 1)
        self.assertEqual(target.disconnected_angular_overlap[0, 1], 0)
        # Frame 0 local slot1 is primitive2, absent from current frame -> dustbin.
        self.assertEqual(target.temporal_correspondence[0, 1, 4], 1)
        # Current frame primitives 0/1 map to current local slots 0/1.
        self.assertEqual(target.temporal_correspondence[4, 0, 0], 1)
        self.assertEqual(target.temporal_correspondence[4, 1, 1], 1)
        supports = []
        for sources in frames:
            counts = np.zeros(3, dtype=np.int32); low = np.full(3, -1, dtype=np.int32); high = np.full(3, -1, dtype=np.int32); azimuth = np.zeros((3,720), dtype=np.uint8)
            for ray, source in enumerate(sources):
                for primitive in source:
                    counts[primitive] += 1; low[primitive] = 0; high[primitive] = 1; azimuth[primitive, ray % 720] = 1
            supports.append(PrimitiveFrameSupport(counts, low, high, np.packbits(azimuth, axis=1, bitorder="little")))
        cached = primitive_relation_targets_from_frame_support(
            construction=self.construction, primitive_ids=("p0", "p1", "p2"),
            frame_supports=supports, maximum_slots=4,
        )
        for name in (
            "primitive_index", "primitive_mask", "frame_primitive_index",
            "frame_primitive_mask", "temporal_correspondence",
            "endpoint_attachment", "disconnected_angular_overlap",
        ):
            np.testing.assert_array_equal(getattr(cached, name), getattr(target, name))

    def test_capacity_overflow_fails_without_truncation(self) -> None:
        sources = [[(index,) for index in range(5)]] * 5
        with self.assertRaisesRegex(OverflowError, "exceeds"):
            primitive_relation_window_targets(
                construction=self.construction,
                primitive_ids=tuple(f"p{index}" for index in range(5)),
                source_sets_by_frame=sources, maximum_slots=4,
            )


if __name__ == "__main__": unittest.main()
