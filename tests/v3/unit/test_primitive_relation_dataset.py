import numpy as np
import pytest

from mtare_topo.data.primitive_relation_dataset import (
    PrimitiveMembershipCodebook,
    visible_primitive_window_targets,
)
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipsePrimitive,
    SweptSuperellipseProvenanceField,
)


def _hit(distance, point, sources):
    return PrimitiveRayHit(distance, tuple(point), tuple(sources), len(sources) == 1)


def _primitive(identity, y):
    return SweptSuperellipsePrimitive(
        identity, np.asarray([[0, y, 0], [4, y, 0]], dtype=float),
        ((1, .8), (1.2, .9)), (2, 6),
    )


def test_membership_codebook_round_trip_retains_ambiguity():
    codebook = PrimitiveMembershipCodebook(("a", "b"))
    hits = [None, _hit(1, [0,0,0], ["a"]), _hit(2, [0,0,0], ["b", "a"])]
    codes = codebook.encode(hits)
    assert codes.dtype == np.uint16
    assert codebook.decode(codes) == ((), (0,), (0, 1))
    assert codebook.cardinality(codes).tolist() == [0, 1, 2]


def test_membership_codebook_rejects_unknown_identity():
    codebook = PrimitiveMembershipCodebook(("a",))
    with pytest.raises(ValueError, match="unknown primitive"):
        codebook.encode([_hit(1, [0,0,0], ["b"])])


def test_visible_window_crops_axes_and_preserves_temporal_identity():
    field = SweptSuperellipseProvenanceField((_primitive("a", 0), _primitive("b", 2)), spacing_m=.1)
    frames = [[], [], [_hit(1, [1,0.8,0], ["a"])], [_hit(1, [2,0.8,0], ["a"])], [_hit(1, [3,2.8,0], ["b"])]]
    target = visible_primitive_window_targets(
        field=field, hits_by_frame=frames, current_sensor_xyz_m=np.zeros(3), current_yaw_deg=0,
    )
    assert target.primitive_index[:2].tolist() == [0, 1]
    assert target.mask.sum() == 2
    assert target.support_ray_count[:2].tolist() == [2, 1]
    assert target.temporal_visibility[:, 0].tolist() == [0, 0, 1, 1, 0]
    assert target.temporal_visibility[:, 1].tolist() == [0, 0, 0, 0, 1]
    assert target.axis_control_current_sensor_m[0, 0, 0] == pytest.approx(1, abs=.11)
    assert target.axis_control_current_sensor_m[0, -1, 0] == pytest.approx(2, abs=.11)


def test_visible_window_fails_instead_of_truncating_slot_overflow():
    primitives = tuple(_primitive(str(index), 2*index) for index in range(9))
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.1)
    hits = [_hit(1, [1,2*index+.8,0], [str(index)]) for index in range(9)]
    with pytest.raises(OverflowError, match="exceeds 8"):
        visible_primitive_window_targets(
            field=field, hits_by_frame=[hits, [], [], [], []],
            current_sensor_xyz_m=np.zeros(3), current_yaw_deg=0,
        )
