from __future__ import annotations

import unittest

import numpy as np

from mtare_topo.data.primitive_relation_storage import (
    pack_primitive_relation_targets,
    unpack_disconnected_overlap,
    unpack_endpoint_attachment,
    unpack_temporal_correspondence,
)
from mtare_topo.data.primitive_relation_targets import PrimitiveRelationWindowTargets


class PrimitiveRelationStorageTest(unittest.TestCase):
    def test_dense_relations_round_trip_losslessly(self) -> None:
        primitive_index = np.full(32, -1, dtype=np.int32); primitive_index[:4] = np.arange(4)
        primitive_mask = (primitive_index >= 0).astype(np.uint8)
        frame_index = np.full((5,32), -1, dtype=np.int32); frame_index[:, :4] = np.arange(4)
        frame_mask = (frame_index >= 0).astype(np.uint8)
        temporal = np.zeros((5,32,33), dtype=np.uint8)
        for frame in range(5):
            for slot in range(4): temporal[frame,slot,slot] = 1
        attachment = np.zeros((32,2,32,2), dtype=np.uint8)
        for other in range(1,4): attachment[0,1,other,0] = 1; attachment[other,0,0,1] = 1
        endpoint_mask = np.zeros_like(attachment); endpoint_mask[:4,:, :4,:] = 1
        for slot in range(4):endpoint_mask[slot,:,slot,:]=0
        overlap = np.zeros((32,32),dtype=np.uint8); overlap[1,2]=overlap[2,1]=1
        pair_mask=np.zeros((32,32),dtype=np.uint8);pair_mask[:4,:4]=1;np.fill_diagonal(pair_mask,0)
        target=PrimitiveRelationWindowTargets(primitive_index,primitive_mask,frame_index,frame_mask,temporal,frame_mask.copy(),attachment,endpoint_mask,overlap,pair_mask)
        packed=pack_primitive_relation_targets(target)
        restored_temporal, restored_mask=unpack_temporal_correspondence(packed)
        np.testing.assert_array_equal(restored_temporal,temporal);np.testing.assert_array_equal(restored_mask,frame_mask)
        np.testing.assert_array_equal(unpack_endpoint_attachment(packed),attachment)
        np.testing.assert_array_equal(unpack_disconnected_overlap(packed),overlap)


if __name__ == "__main__": unittest.main()
