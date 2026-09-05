from mtare_topo.data.c08_causal_replay import merge_clearance_risk_segments


def record(frame, clearance, edge="e0", tunnel=1):
    return {
        "frame_index": frame, "route_arc_m": frame * 2.0,
        "minimum_horizontal_clearance_m": clearance,
        "clearance_passed": clearance >= .8, "edge_id": edge,
        "tunnel_id": tunnel, "axis_xyz_m": [float(frame), 0.0, 0.0],
    }


def test_risk_segments_merge_only_consecutive_frames():
    rows = [record(0, 1.0), record(1, .7), record(2, .6, "e1", 2), record(3, 1.0), record(4, .5)]
    segments = merge_clearance_risk_segments(rows)
    assert len(segments) == 2
    assert segments[0]["start_frame"] == 1
    assert segments[0]["end_frame"] == 2
    assert segments[0]["minimum_frame"] == 2
    assert segments[0]["edge_ids"] == ["e0", "e1"]
    assert segments[1]["frame_count"] == 1


def test_no_risk_has_no_segments():
    assert merge_clearance_risk_segments([record(0, .8), record(1, 1.0)]) == []
