import hashlib

import pytest

from integration.native_structure_bridge import prepare_scan_source_overlay as overlay


def _source(**repeats):
    source = (
        '#include <std_msgs/Bool.h>\n'
        'bool use_gazebo_time = false;\n'
        '  nhPrivate.getParam("use_gazebo_time", use_gazebo_time);\n'
        '  float pointX3 = pointX2 + vehicleRecX;\n'
        '  pubScanPointer->publish(scanData2);\n'
    )
    for needle, count in repeats.items():
        source += needle * (count - 1)
    return source


def test_transform_rejects_source_sha_drift(monkeypatch):
    source = _source()
    monkeypatch.setattr(overlay, 'EXPECTED', hashlib.sha256(source.encode()).hexdigest())
    with pytest.raises(ValueError, match='PINNED_VEHICLE_SIMULATOR_SOURCE_CHANGED'):
        overlay.transform(source + 'drift')


def test_transform_rejects_non_unique_anchor(monkeypatch):
    source = _source().replace('bool use_gazebo_time = false;\n', 'bool use_gazebo_time = false;\n' * 2)
    monkeypatch.setattr(overlay, 'EXPECTED', hashlib.sha256(source.encode()).hexdigest())
    with pytest.raises(ValueError, match='PINNED_SCAN_SOURCE_ANCHOR_CHANGED'):
        overlay.transform(source)


def test_transform_preserves_publish_and_only_adds_source_topic(monkeypatch):
    source = _source()
    monkeypatch.setattr(overlay, 'EXPECTED', hashlib.sha256(source.encode()).hexdigest())
    changed = overlay.transform(source)
    assert changed.count('pubScanPointer->publish(scanData2);') == 1
    assert 'recordNativeScanSources = false' in changed
    assert '/native_structure/scan_sources' in changed
    assert 'pubWaypoint' not in changed
    assert changed.count('  pubScanPointer->publish(scanData2);\n') == 1
    assert changed.count('pubNativeScanSources.publish(evidence);') == 1
    assert '  float pointX3 = pointX2 + vehicleRecX;\n' in changed
